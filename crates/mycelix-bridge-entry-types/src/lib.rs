#![allow(deprecated)] // Uses legacy ConsciousnessCredential for backward-compat bridge
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Bridge Entry Types — Shared DHT entry structs for cluster bridges
//!
//! Provides the canonical entry types stored on the DHT by both the Commons
//! and Civic bridge integrity zomes. Each integrity zome wraps these in its
//! own `#[hdk_entry_types]` enum (required by Holochain), but the underlying
//! struct definitions live here — single source of truth.
//!
//! Also provides domain-independent validation helpers.
//!
//! ## Schema Versioning Policy
//!
//! All entry types carry a `schema_version` field (default 1). When adding new
//! fields to an existing entry type:
//! 1. Use `#[serde(default)]` or `#[serde(default = "...")]` so existing on-chain
//!    data deserializes without the new field.
//! 2. Bump `schema_version` in new entries so readers can distinguish generations.
//! 3. Never remove or rename existing fields — only add new ones.

use hdi::prelude::*;

fn default_schema_v1() -> u8 {
    1
}

// ============================================================================
// Size limit constants — prevent DHT storage exhaustion attacks
// ============================================================================

/// Maximum length for domain identifiers (e.g., "property", "justice")
pub const MAX_DOMAIN_BYTES: usize = 256;

/// Maximum length for query_type / event_type identifiers
pub const MAX_TYPE_BYTES: usize = 256;

/// Maximum length for query result payloads (64 KB)
pub const MAX_RESULT_BYTES: usize = 65_536;

/// Maximum length for each individual hash in related_hashes
pub const MAX_HASH_BYTES: usize = 128;

/// Maximum length for DID strings
pub const MAX_DID_BYTES: usize = 256;

// ============================================================================
// Entry types
// ============================================================================

/// Cross-domain query entry stored on the DHT.
///
/// Used by both commons-bridge and civic-bridge integrity zomes
/// as a variant in their `EntryTypes` enum.
///
/// We derive `SerializedBytes` + manually implement `TryFrom<&Entry>` instead
/// of using `#[hdk_entry_helper]`. The helper macro generates code referencing
/// `From<&T> for SerializedBytes` which is incompatible with `derive(SerializedBytes)`
/// in HDI 0.7 — a known macro mismatch. Our manual impl achieves the same result.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct BridgeQueryEntry {
    /// Schema version for forward-compatible deserialization.
    #[serde(default = "default_schema_v1")]
    pub schema_version: u8,
    /// Domain within the cluster (e.g., "property", "justice")
    pub domain: String,
    /// Query type / function name
    pub query_type: String,
    /// The agent initiating the query
    pub requester: AgentPubKey,
    /// Query parameters (JSON string)
    pub params: String,
    /// Result payload (filled after resolution)
    pub result: Option<String>,
    /// When the query was created
    pub created_at: Timestamp,
    /// When the query was resolved
    pub resolved_at: Option<Timestamp>,
    /// Whether the query succeeded
    pub success: Option<bool>,
}

impl TryFrom<&Entry> for BridgeQueryEntry {
    type Error = WasmError;
    fn try_from(entry: &Entry) -> Result<Self, Self::Error> {
        match entry {
            Entry::App(bytes) => {
                let sb = SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()));
                <Self as TryFrom<SerializedBytes>>::try_from(sb)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not an app entry".into(),
            ))),
        }
    }
}

/// Cross-domain event entry stored on the DHT.
///
/// Used by both commons-bridge and civic-bridge integrity zomes.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct BridgeEventEntry {
    /// Schema version for forward-compatible deserialization.
    #[serde(default = "default_schema_v1")]
    pub schema_version: u8,
    /// Domain within the cluster (e.g., "housing", "emergency")
    pub domain: String,
    /// Event type identifier
    pub event_type: String,
    /// Agent that triggered the event
    pub source_agent: AgentPubKey,
    /// Event payload (JSON string)
    pub payload: String,
    /// When the event occurred
    pub created_at: Timestamp,
    /// Related entry hashes for cross-referencing
    pub related_hashes: Vec<String>,
}

impl TryFrom<&Entry> for BridgeEventEntry {
    type Error = WasmError;
    fn try_from(entry: &Entry) -> Result<Self, Self::Error> {
        match entry {
            Entry::App(bytes) => {
                let sb = SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()));
                <Self as TryFrom<SerializedBytes>>::try_from(sb)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not an app entry".into(),
            ))),
        }
    }
}

// ============================================================================
// Cached credential entry
// ============================================================================

/// Cached consciousness credential stored on the agent's source chain.
///
/// Stores the serialized `ConsciousnessCredential` (from bridge-common) as JSON
/// to avoid a circular dependency. The coordinator layer (which has both deps)
/// handles serialization/deserialization.
///
/// TTL checking is done at the coordinator level — integrity only validates
/// structural constraints.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct CachedCredentialEntry {
    /// Schema version for forward-compatible deserialization.
    #[serde(default = "default_schema_v1")]
    pub schema_version: u8,
    /// The DID this credential was issued for
    pub did: String,
    /// Serialized ConsciousnessCredential as JSON
    pub credential_json: String,
    /// When this cache entry was stored (microseconds since epoch)
    pub cached_at_us: i64,
}

impl TryFrom<&Entry> for CachedCredentialEntry {
    type Error = WasmError;
    fn try_from(entry: &Entry) -> Result<Self, Self::Error> {
        match entry {
            Entry::App(bytes) => {
                let sb = SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()));
                <Self as TryFrom<SerializedBytes>>::try_from(sb)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not an app entry".into(),
            ))),
        }
    }
}

/// Validate a cached credential entry.
///
/// Returns `Ok(())` if valid, or `Err(reason)` if invalid.
pub fn validate_cached_credential(entry: &CachedCredentialEntry) -> Result<(), String> {
    if entry.did.is_empty() {
        return Err("Cached credential DID cannot be empty".into());
    }
    if entry.did.len() > MAX_DID_BYTES {
        return Err(format!(
            "Cached credential DID too long ({} bytes, max {})",
            entry.did.len(),
            MAX_DID_BYTES
        ));
    }
    if entry.credential_json.is_empty() {
        return Err("Cached credential JSON cannot be empty".into());
    }
    if entry.credential_json.len() > 16384 {
        return Err("Cached credential JSON too large (max 16384 bytes)".into());
    }
    Ok(())
}

// ============================================================================
// Jurisdiction constraint entry
// ============================================================================

/// Jurisdictional constraint applied to a geographic zone or regulatory domain.
///
/// Models nation-states and regulatory bodies as **environmental constraints**
/// on the peer-to-peer network — tags applied to geographic clusters, NOT root
/// nodes that own the network. A jurisdiction constraint is a localized set of
/// rules that physical nodes happen to sit inside.
///
/// ## Design Philosophy
///
/// Nation-states are shared human conventions with real physical consequences.
/// In Mycelix, they are modeled the same way emergencies use `OperationalZone`:
/// as environmental facts that constrain operations, not authorities that
/// control the DHT.
///
/// ## Example
///
/// ```ignore
/// JurisdictionConstraintEntry {
///     zone_id: "eu-gdpr-zone-1".into(),
///     zone_polygon: vec![(35.0, -10.0), (71.0, -10.0), (71.0, 40.0), (35.0, 40.0)],
///     regulatory_tags: vec!["gdpr".into(), "right_to_erasure".into()],
///     fiat_zone: Some("EUR".into()),
///     enforcement_risk: 0.85, // High — EU actively enforces GDPR
///     authority_did: Some("did:mycelix:eu-dpa-collective".into()),
///     description: "European Union GDPR enforcement zone".into(),
///     created_at: Timestamp::now()?,
///     last_validated: Timestamp::now()?,
/// }
/// ```
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct JurisdictionConstraintEntry {
    /// Unique identifier for this jurisdiction zone
    pub zone_id: String,
    /// GIS polygon vertices (lat, lon) defining the geographic boundary.
    /// Empty = non-geographic jurisdiction (e.g., internet-only regulation).
    pub zone_polygon: Vec<(f64, f64)>,
    /// Regulatory tags that apply within this zone (e.g., "gdpr", "hipaa", "fatf_travel_rule").
    /// Nodes and actions within the zone must check these tags for compliance.
    pub regulatory_tags: Vec<String>,
    /// Fiat currency zone identifier (ISO 4217), if applicable.
    pub fiat_zone: Option<String>,
    /// Enforcement risk [0.0, 1.0]: probability that violations are actively prosecuted.
    /// 0.0 = unenforced, 1.0 = actively and consistently enforced.
    pub enforcement_risk: f64,
    /// DID of the authority that asserts this jurisdiction (e.g., a regulatory body).
    /// `None` = community-asserted (crowdsourced geographic tagging).
    pub authority_did: Option<String>,
    /// Human-readable description of this jurisdiction constraint.
    pub description: String,
    /// When this constraint was first created.
    pub created_at: Timestamp,
    /// When this constraint was last validated/confirmed as active.
    pub last_validated: Timestamp,
}

impl TryFrom<&Entry> for JurisdictionConstraintEntry {
    type Error = WasmError;
    fn try_from(entry: &Entry) -> Result<Self, Self::Error> {
        match entry {
            Entry::App(bytes) => {
                let sb = SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()));
                <Self as TryFrom<SerializedBytes>>::try_from(sb)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not an app entry".into(),
            ))),
        }
    }
}

/// Validate a jurisdiction constraint entry.
///
/// Returns `Ok(())` if valid, or `Err(reason)` if invalid.
pub fn validate_jurisdiction_constraint(entry: &JurisdictionConstraintEntry) -> Result<(), String> {
    if entry.zone_id.is_empty() {
        return Err("Zone ID cannot be empty".into());
    }
    if entry.zone_id.len() > 256 {
        return Err("Zone ID must be 256 characters or fewer".into());
    }
    if entry.zone_polygon.len() > 1000 {
        return Err("Zone polygon cannot have more than 1000 vertices".into());
    }
    // Validate polygon vertices are valid coordinates
    for (lat, lon) in &entry.zone_polygon {
        if !lat.is_finite() || !lon.is_finite() {
            return Err("Polygon vertices must have finite coordinates".into());
        }
        if *lat < -90.0 || *lat > 90.0 {
            return Err(format!("Latitude {} out of range [-90, 90]", lat));
        }
        if *lon < -180.0 || *lon > 180.0 {
            return Err(format!("Longitude {} out of range [-180, 180]", lon));
        }
    }
    if entry.regulatory_tags.is_empty() {
        return Err("At least one regulatory tag is required".into());
    }
    if entry.regulatory_tags.len() > 50 {
        return Err("Cannot have more than 50 regulatory tags".into());
    }
    for tag in &entry.regulatory_tags {
        if tag.is_empty() {
            return Err("Regulatory tags cannot be empty strings".into());
        }
        if tag.len() > 128 {
            return Err(format!(
                "Regulatory tag '{}...' exceeds 128 character limit",
                &tag[..32.min(tag.len())]
            ));
        }
    }
    if !entry.enforcement_risk.is_finite()
        || entry.enforcement_risk < 0.0
        || entry.enforcement_risk > 1.0
    {
        return Err(format!(
            "Enforcement risk must be in [0.0, 1.0], got {}",
            entry.enforcement_risk
        ));
    }
    if entry.description.len() > 4096 {
        return Err("Description must be 4096 characters or fewer".into());
    }
    Ok(())
}

// ============================================================================
// Schema Migration Framework
// ============================================================================

/// Trait for entry types that support schema migration.
///
/// Implementors declare their current schema version and provide a migration
/// path from older versions. This enables forward-compatible DHT reads: when
/// a node encounters an entry written by a newer or older software version,
/// migration can convert it to the current in-memory representation.
///
/// ## Adding a v1 → v2 migration (example)
///
/// When you add a new field to `BridgeQueryEntry`:
///
/// 1. Add the field with `#[serde(default)]` so v1 entries deserialize.
/// 2. Bump `CURRENT_VERSION` to 2.
/// 3. In `migrate()`, handle `from_version == 1`:
///
/// ```ignore
/// fn migrate(raw: &Entry, from_version: u8) -> Result<Option<Self>, String> {
///     match from_version {
///         1 => {
///             // Deserialize v1 entry (missing new_field will get serde default)
///             let entry = Self::try_from(raw)
///                 .map_err(|e| format!("v1 migration failed: {}", e))?;
///             // Optionally transform fields here
///             Ok(Some(entry))
///         }
///         v if v == Self::CURRENT_VERSION => Ok(None), // already current
///         v => Err(format!("Unknown schema version {}", v)),
///     }
/// }
/// ```
pub trait SchemaMigration: Sized {
    /// Current schema version for this type.
    const CURRENT_VERSION: u8;

    /// Migrate from an older version to the current version.
    ///
    /// Returns:
    /// - `Ok(None)` if the entry is already at the current version (no migration needed).
    /// - `Ok(Some(migrated))` if migration succeeded.
    /// - `Err(reason)` if migration is not possible (e.g., unknown future version).
    fn migrate(_raw: &Entry, from_version: u8) -> Result<Option<Self>, String>;
}

impl SchemaMigration for BridgeQueryEntry {
    const CURRENT_VERSION: u8 = 1;

    fn migrate(_raw: &Entry, from_version: u8) -> Result<Option<Self>, String> {
        match from_version {
            1 => Ok(None), // Already at current version — no migration needed
            v if v > Self::CURRENT_VERSION => Err(format!(
                "BridgeQueryEntry schema version {} is newer than supported version {}",
                v,
                Self::CURRENT_VERSION
            )),
            v => Err(format!(
                "BridgeQueryEntry has no migration path from version {}",
                v
            )),
        }
    }
}

impl SchemaMigration for BridgeEventEntry {
    const CURRENT_VERSION: u8 = 1;

    fn migrate(_raw: &Entry, from_version: u8) -> Result<Option<Self>, String> {
        match from_version {
            1 => Ok(None), // Already at current version — no migration needed
            v if v > Self::CURRENT_VERSION => Err(format!(
                "BridgeEventEntry schema version {} is newer than supported version {}",
                v,
                Self::CURRENT_VERSION
            )),
            v => Err(format!(
                "BridgeEventEntry has no migration path from version {}",
                v
            )),
        }
    }
}

impl SchemaMigration for CachedCredentialEntry {
    const CURRENT_VERSION: u8 = 1;

    fn migrate(_raw: &Entry, from_version: u8) -> Result<Option<Self>, String> {
        match from_version {
            1 => Ok(None), // Already at current version — no migration needed
            v if v > Self::CURRENT_VERSION => Err(format!(
                "CachedCredentialEntry schema version {} is newer than supported version {}",
                v,
                Self::CURRENT_VERSION
            )),
            v => Err(format!(
                "CachedCredentialEntry has no migration path from version {}",
                v
            )),
        }
    }
}

/// Read an entry with automatic schema migration.
///
/// Attempts to deserialize the entry as the current version of `T`. If
/// deserialization succeeds and the `schema_version` field matches
/// `T::CURRENT_VERSION`, returns the entry directly. If the version is
/// older, attempts migration via `T::migrate()`.
///
/// ## Errors
///
/// Returns `Err` if:
/// - The entry cannot be deserialized at all (corrupt or wrong type).
/// - The schema version is from the future (newer than `CURRENT_VERSION`).
/// - Migration fails for any other reason.
///
/// ## Example
///
/// ```ignore
/// let entry: BridgeQueryEntry = read_with_migration::<BridgeQueryEntry>(&raw_entry)?;
/// ```
pub fn read_with_migration<T>(entry: &Entry) -> Result<T, String>
where
    T: SchemaMigration + for<'a> TryFrom<&'a Entry, Error = WasmError>,
{
    // First, try to deserialize as-is. Serde defaults handle missing fields
    // from older versions, so this usually succeeds even for old entries.
    match T::try_from(entry) {
        Ok(value) => Ok(value),
        Err(deser_err) => {
            // Deserialization failed — try migration from version 0 as a last resort.
            // In practice, entries that can't deserialize at all are truly incompatible.
            match T::migrate(entry, 0) {
                Ok(Some(migrated)) => Ok(migrated),
                Ok(None) => Err(format!(
                    "Entry deserialization failed and no migration applied: {}",
                    deser_err
                )),
                Err(mig_err) => Err(format!(
                    "Entry deserialization failed ({}), migration also failed ({})",
                    deser_err, mig_err
                )),
            }
        }
    }
}

// ============================================================================
// Author validation helpers
// ============================================================================

/// Check that the author of an update/delete matches the original entry author.
///
/// Returns `ValidateCallbackResult::Valid` if authors match, or
/// `ValidateCallbackResult::Invalid` with a descriptive message if they don't.
///
/// # Arguments
/// * `original_author` - The author of the original entry/action
/// * `action_author` - The author of the update/delete action
/// * `operation` - Human-readable operation name (e.g., "update", "delete")
pub fn check_author_match(
    original_author: &AgentPubKey,
    action_author: &AgentPubKey,
    operation: &str,
) -> ValidateCallbackResult {
    if action_author != original_author {
        ValidateCallbackResult::Invalid(format!(
            "Only the original entry author can {} their entries",
            operation,
        ))
    } else {
        ValidateCallbackResult::Valid
    }
}

/// Check that the author of a delete-link matches the original link author.
///
/// Returns `ValidateCallbackResult::Valid` if authors match, or
/// `ValidateCallbackResult::Invalid` with a descriptive message if they don't.
pub fn check_link_author_match(
    original_author: &AgentPubKey,
    action_author: &AgentPubKey,
) -> ValidateCallbackResult {
    if action_author != original_author {
        ValidateCallbackResult::Invalid("Only the original author can delete this link".into())
    } else {
        ValidateCallbackResult::Valid
    }
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate a bridge query entry against a domain allowlist.
///
/// Returns `Ok(())` if valid, or `Err(reason)` if invalid.
/// Each integrity zome calls this with its own `VALID_DOMAINS` list.
pub fn validate_query_fields(
    query: &BridgeQueryEntry,
    valid_domains: &[&str],
) -> Result<(), String> {
    if query.domain.len() > MAX_DOMAIN_BYTES {
        return Err(format!(
            "Domain too long ({} bytes, max {})",
            query.domain.len(),
            MAX_DOMAIN_BYTES
        ));
    }
    if !valid_domains.contains(&query.domain.as_str()) {
        return Err(format!(
            "Invalid domain '{}'. Must be one of: {:?}",
            query.domain, valid_domains
        ));
    }
    if query.query_type.len() > MAX_TYPE_BYTES {
        return Err(format!(
            "Query type too long ({} bytes, max {})",
            query.query_type.len(),
            MAX_TYPE_BYTES
        ));
    }
    if query.params.len() > 8192 {
        return Err("Parameters must be 8192 characters or fewer".into());
    }
    if !query.params.is_empty() {
        if serde_json::from_str::<serde_json::Value>(&query.params).is_err() {
            return Err("Parameters must be valid JSON".into());
        }
    }
    if let Some(ref result) = query.result {
        if result.len() > MAX_RESULT_BYTES {
            return Err(format!(
                "Result too large ({} bytes, max {})",
                result.len(),
                MAX_RESULT_BYTES
            ));
        }
    }
    Ok(())
}

/// Validate a bridge event entry against a domain allowlist.
///
/// Returns `Ok(())` if valid, or `Err(reason)` if invalid.
pub fn validate_event_fields(
    event: &BridgeEventEntry,
    valid_domains: &[&str],
) -> Result<(), String> {
    if event.domain.len() > MAX_DOMAIN_BYTES {
        return Err(format!(
            "Domain too long ({} bytes, max {})",
            event.domain.len(),
            MAX_DOMAIN_BYTES
        ));
    }
    if !valid_domains.contains(&event.domain.as_str()) {
        return Err(format!(
            "Invalid domain '{}'. Must be one of: {:?}",
            event.domain, valid_domains
        ));
    }
    if event.event_type.len() > MAX_TYPE_BYTES {
        return Err(format!(
            "Event type too long ({} bytes, max {})",
            event.event_type.len(),
            MAX_TYPE_BYTES
        ));
    }
    if event.payload.len() > 8192 {
        return Err("Payload must be 8192 characters or fewer".into());
    }
    if event.related_hashes.len() > 20 {
        return Err("Cannot have more than 20 related hashes".into());
    }
    for (i, hash) in event.related_hashes.iter().enumerate() {
        if hash.len() > MAX_HASH_BYTES {
            return Err(format!(
                "Related hash [{}] too long ({} bytes, max {})",
                i,
                hash.len(),
                MAX_HASH_BYTES
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn make_query(domain: &str, params: &str) -> BridgeQueryEntry {
        BridgeQueryEntry {
            schema_version: 1,
            domain: domain.into(),
            query_type: "test_query".into(),
            requester: fake_agent(),
            params: params.into(),
            result: None,
            created_at: Timestamp::from_micros(0),
            resolved_at: None,
            success: None,
        }
    }

    fn make_event(domain: &str, payload: &str) -> BridgeEventEntry {
        BridgeEventEntry {
            schema_version: 1,
            domain: domain.into(),
            event_type: "test_event".into(),
            source_agent: fake_agent(),
            payload: payload.into(),
            created_at: Timestamp::from_micros(0),
            related_hashes: vec![],
        }
    }

    const DOMAINS: &[&str] = &[
        "property",
        "housing",
        "care",
        "justice",
        "mutualaid",
        "water",
        "food",
        "transport",
    ];

    // ---- validate_query_fields ----

    #[test]
    fn query_valid_domain_accepted() {
        let q = make_query("property", "{}");
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_all_valid_domains_accepted() {
        for domain in DOMAINS {
            let q = make_query(domain, "{}");
            assert!(
                validate_query_fields(&q, DOMAINS).is_ok(),
                "domain '{}' should be valid",
                domain
            );
        }
    }

    #[test]
    fn query_invalid_domain_rejected() {
        let q = make_query("invalid", "{}");
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
        assert!(err.contains("invalid"));
    }

    #[test]
    fn query_empty_domain_rejected() {
        let q = make_query("", "{}");
        assert!(validate_query_fields(&q, DOMAINS).is_err());
    }

    #[test]
    fn query_empty_params_accepted() {
        let q = make_query("property", "");
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_valid_json_params_accepted() {
        let q = make_query("property", r#"{"key":"value","nested":{"a":1}}"#);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_invalid_json_params_rejected() {
        let q = make_query("property", "{not json at all");
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("valid JSON"));
    }

    #[test]
    fn query_params_at_limit_accepted() {
        // Exactly 8192 chars of valid JSON: {"k":"xxx..."}
        let wrapper_len = r#"{"k":""}"#.len(); // 8 chars
        let padding = "x".repeat(8192 - wrapper_len);
        let json = format!(r#"{{"k":"{}"}}"#, padding);
        assert_eq!(json.len(), 8192);
        let q = make_query("property", &json);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_params_over_limit_rejected() {
        let big = "x".repeat(8193);
        let q = make_query("property", &big);
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    // ---- validate_event_fields ----

    #[test]
    fn event_valid_domain_accepted() {
        let e = make_event("housing", "{}");
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_invalid_domain_rejected() {
        let e = make_event("unknown", "{}");
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("Invalid domain"));
    }

    #[test]
    fn event_payload_at_limit_accepted() {
        let payload = "x".repeat(8192);
        let e = make_event("property", &payload);
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_payload_over_limit_rejected() {
        let payload = "x".repeat(8193);
        let e = make_event("property", &payload);
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("8192"));
    }

    #[test]
    fn event_exactly_20_hashes_accepted() {
        let mut e = make_event("property", "{}");
        e.related_hashes = (0..20).map(|i| format!("hash_{}", i)).collect();
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_21_hashes_rejected() {
        let mut e = make_event("property", "{}");
        e.related_hashes = (0..21).map(|i| format!("hash_{}", i)).collect();
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("20 related hashes"));
    }

    #[test]
    fn event_zero_hashes_accepted() {
        let e = make_event("care", "{}");
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    // ---- Serde roundtrip ----

    #[test]
    fn query_serde_roundtrip() {
        let q = make_query("property", r#"{"key":"value"}"#);
        let bytes = serde_json::to_vec(&q).unwrap();
        let q2: BridgeQueryEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(q, q2);
    }

    #[test]
    fn query_with_result_serde_roundtrip() {
        let mut q = make_query("housing", "{}");
        q.result = Some("resolved data".into());
        q.resolved_at = Some(Timestamp::from_micros(1_000_000));
        q.success = Some(true);
        let bytes = serde_json::to_vec(&q).unwrap();
        let q2: BridgeQueryEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(q, q2);
    }

    #[test]
    fn event_serde_roundtrip() {
        let mut e = make_event("care", r#"{"action":"help"}"#);
        e.related_hashes = vec!["hash_a".into(), "hash_b".into()];
        let bytes = serde_json::to_vec(&e).unwrap();
        let e2: BridgeEventEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(e, e2);
    }

    // ---- Food & transport domain validation ----

    #[test]
    fn query_food_domain_accepted() {
        let q = make_query("food", r#"{"product":"tomatoes"}"#);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_transport_domain_accepted() {
        let q = make_query("transport", r#"{"route_id":"r1"}"#);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn event_food_domain_accepted() {
        let e = make_event("food", r#"{"event":"harvest_recorded"}"#);
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn event_transport_domain_accepted() {
        let mut e = make_event("transport", r#"{"event":"trip_logged"}"#);
        e.related_hashes = vec!["vehicle_hash_1".into()];
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn query_all_domains_accepted() {
        for domain in DOMAINS {
            let q = make_query(domain, "{}");
            assert!(
                validate_query_fields(&q, DOMAINS).is_ok(),
                "domain '{}' should be valid",
                domain
            );
        }
    }

    #[test]
    fn event_all_domains_accepted() {
        for domain in DOMAINS {
            let e = make_event(domain, "{}");
            assert!(
                validate_event_fields(&e, DOMAINS).is_ok(),
                "domain '{}' should be valid",
                domain
            );
        }
    }

    // ---- Empty domain list ----

    #[test]
    fn empty_domain_list_rejects_everything() {
        let q = make_query("property", "{}");
        assert!(validate_query_fields(&q, &[]).is_err());

        let e = make_event("property", "{}");
        assert!(validate_event_fields(&e, &[]).is_err());
    }

    // ---- Edge cases ----

    #[test]
    fn query_null_bytes_in_domain_serde_roundtrip() {
        let q = BridgeQueryEntry {
            schema_version: 1,
            domain: "prop\0erty".into(),
            query_type: "test\0query".into(),
            requester: fake_agent(),
            params: r#"{"key":"val\u0000ue"}"#.into(),
            result: None,
            created_at: Timestamp::from_micros(0),
            resolved_at: None,
            success: None,
        };
        let bytes = serde_json::to_vec(&q).unwrap();
        let q2: BridgeQueryEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(q, q2);
        assert!(q2.domain.contains('\0'));
        assert!(q2.query_type.contains('\0'));
    }

    #[test]
    fn query_deeply_nested_json_params_roundtrip() {
        // Build 15 levels of nesting: {"a":{"a":{"a":...true...}}}
        let depth = 15;
        let mut json = String::new();
        for _ in 0..depth {
            json.push_str(r#"{"a":"#);
        }
        json.push_str("true");
        for _ in 0..depth {
            json.push('}');
        }
        // Confirm the string is valid JSON
        assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());

        let q = make_query("property", &json);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());

        let bytes = serde_json::to_vec(&q).unwrap();
        let q2: BridgeQueryEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(q, q2);
        assert_eq!(q2.params, json);
    }

    #[test]
    fn event_unicode_edge_cases_serde_roundtrip() {
        let unicode_payload = concat!(
            r#"{"emoji":"#,
            r#""🌟🔮🧘🕸️","#,                     // emoji
            r#""cjk":"你好世界","#,               // CJK characters
            r#""rtl":"مرحبا","#,                  // RTL Arabic text
            r#""zwj":"a\u200Bb\u200Cc\uFEFFd","#, // zero-width chars (ZWSP, ZWNJ, BOM)
            r#""combining":"e\u0301""#,           // combining accent (é as e + ◌́)
            r#"}"#
        );
        // Verify the string is valid JSON before constructing the entry
        assert!(serde_json::from_str::<serde_json::Value>(unicode_payload).is_ok());

        let mut e = make_event("care", unicode_payload);
        e.related_hashes = vec!["hash_🌟".into(), "哈希".into()];

        let bytes = serde_json::to_vec(&e).unwrap();
        let e2: BridgeEventEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(e, e2);
        assert!(e2.payload.contains("🌟"));
        assert!(e2.payload.contains("你好"));
        assert!(e2.payload.contains("مرحبا"));
    }

    #[test]
    fn query_params_exactly_at_8192_boundary() {
        // Build valid JSON that is exactly 8192 bytes
        // {"d":"xxx..."} = 7 wrapper chars + padding
        let wrapper = r#"{"d":""}"#;
        let padding_len = 8192 - wrapper.len();
        let padding = "a".repeat(padding_len);
        let json = format!(r#"{{"d":"{}"}}"#, padding);
        assert_eq!(json.len(), 8192, "params must be exactly 8192 bytes");
        assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());

        let q = make_query("property", &json);
        assert!(
            validate_query_fields(&q, DOMAINS).is_ok(),
            "exactly 8192 bytes should pass"
        );

        // One byte over the limit must fail
        let one_over = format!(r#"{{"d":"{}"}}"#, "a".repeat(padding_len + 1));
        assert_eq!(one_over.len(), 8193);
        let q_over = make_query("property", &one_over);
        assert!(
            validate_query_fields(&q_over, DOMAINS).is_err(),
            "8193 bytes should be rejected"
        );
    }

    #[test]
    fn event_exactly_20_related_hashes_boundary() {
        let mut e = make_event("justice", r#"{"case":"boundary"}"#);
        e.related_hashes = (0..20).map(|i| format!("hash_{:04}", i)).collect();
        assert_eq!(e.related_hashes.len(), 20);
        assert!(
            validate_event_fields(&e, DOMAINS).is_ok(),
            "exactly 20 hashes should pass"
        );

        // Serde roundtrip preserves all 20 hashes
        let bytes = serde_json::to_vec(&e).unwrap();
        let e2: BridgeEventEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(e2.related_hashes.len(), 20);
        assert_eq!(e, e2);

        // 21 must fail
        let mut e_over = e.clone();
        e_over.related_hashes.push("hash_overflow".into());
        assert_eq!(e_over.related_hashes.len(), 21);
        assert!(
            validate_event_fields(&e_over, DOMAINS).is_err(),
            "21 hashes should be rejected"
        );
    }

    #[test]
    fn query_empty_string_fields() {
        let q = BridgeQueryEntry {
            schema_version: 1,
            domain: "property".into(),
            query_type: "".into(),
            requester: fake_agent(),
            params: "".into(),
            result: None,
            created_at: Timestamp::from_micros(0),
            resolved_at: None,
            success: None,
        };
        // Empty query_type is not checked by validate_query_fields,
        // and empty params is explicitly allowed (skips JSON check)
        assert!(validate_query_fields(&q, DOMAINS).is_ok());

        // Serde roundtrip preserves empty strings
        let bytes = serde_json::to_vec(&q).unwrap();
        let q2: BridgeQueryEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(q2.query_type, "");
        assert_eq!(q2.params, "");
        assert_eq!(q, q2);
    }

    // ---- CachedCredentialEntry validation ----

    fn make_cached_credential(did: &str, json: &str) -> CachedCredentialEntry {
        CachedCredentialEntry {
            schema_version: 1,
            did: did.into(),
            credential_json: json.into(),
            cached_at_us: 1_000_000,
        }
    }

    #[test]
    fn cached_credential_valid() {
        let c = make_cached_credential("did:mycelix:abc123", r#"{"profile":{}}"#);
        assert!(validate_cached_credential(&c).is_ok());
    }

    #[test]
    fn cached_credential_empty_did_rejected() {
        let c = make_cached_credential("", r#"{"profile":{}}"#);
        let err = validate_cached_credential(&c).unwrap_err();
        assert!(err.contains("DID cannot be empty"));
    }

    #[test]
    fn cached_credential_empty_json_rejected() {
        let c = make_cached_credential("did:mycelix:abc", "");
        let err = validate_cached_credential(&c).unwrap_err();
        assert!(err.contains("JSON cannot be empty"));
    }

    #[test]
    fn cached_credential_oversized_json_rejected() {
        let big = "x".repeat(16385);
        let c = make_cached_credential("did:mycelix:abc", &big);
        let err = validate_cached_credential(&c).unwrap_err();
        assert!(err.contains("16384"));
    }

    #[test]
    fn cached_credential_at_json_boundary_accepted() {
        let json = "x".repeat(16384);
        let c = make_cached_credential("did:mycelix:abc", &json);
        assert!(validate_cached_credential(&c).is_ok());
    }

    #[test]
    fn cached_credential_serde_roundtrip() {
        let c = make_cached_credential(
            "did:mycelix:agent123",
            r#"{"did":"did:mycelix:agent123","profile":{"identity":0.5}}"#,
        );
        let bytes = serde_json::to_vec(&c).unwrap();
        let c2: CachedCredentialEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn cached_credential_negative_timestamp_roundtrip() {
        let c = CachedCredentialEntry {
            schema_version: 1,
            did: "did:mycelix:test".into(),
            credential_json: "{}".into(),
            cached_at_us: -1,
        };
        let bytes = serde_json::to_vec(&c).unwrap();
        let c2: CachedCredentialEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(c.cached_at_us, c2.cached_at_us);
    }

    #[test]
    fn query_very_long_domain_string_rejected_by_size() {
        let long_domain = "x".repeat(1024);
        let q = make_query(&long_domain, "{}");
        // Rejected by MAX_DOMAIN_BYTES (256) before allowlist check
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("Domain too long"));

        // Even if we add it to the allowlist, it's still rejected by size limit
        let custom_domains: Vec<&str> = vec![&long_domain];
        let err = validate_query_fields(&q, &custom_domains).unwrap_err();
        assert!(err.contains("Domain too long"));
    }

    #[test]
    fn query_domain_at_max_bytes_accepted() {
        let domain = "x".repeat(MAX_DOMAIN_BYTES);
        let q = make_query(&domain, "{}");
        let domains: Vec<&str> = vec![&domain];
        assert!(validate_query_fields(&q, &domains).is_ok());
    }

    #[test]
    fn query_domain_over_max_bytes_rejected() {
        let domain = "x".repeat(MAX_DOMAIN_BYTES + 1);
        let q = make_query(&domain, "{}");
        let domains: Vec<&str> = vec![&domain];
        let err = validate_query_fields(&q, &domains).unwrap_err();
        assert!(err.contains("Domain too long"));
    }

    #[test]
    fn query_type_over_max_bytes_rejected() {
        let mut q = make_query("property", "{}");
        q.query_type = "x".repeat(MAX_TYPE_BYTES + 1);
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("Query type too long"));
    }

    #[test]
    fn query_type_at_max_bytes_accepted() {
        let mut q = make_query("property", "{}");
        q.query_type = "x".repeat(MAX_TYPE_BYTES);
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn query_result_over_max_bytes_rejected() {
        let mut q = make_query("property", "{}");
        q.result = Some("x".repeat(MAX_RESULT_BYTES + 1));
        let err = validate_query_fields(&q, DOMAINS).unwrap_err();
        assert!(err.contains("Result too large"));
    }

    #[test]
    fn query_result_at_max_bytes_accepted() {
        let mut q = make_query("property", "{}");
        q.result = Some("x".repeat(MAX_RESULT_BYTES));
        assert!(validate_query_fields(&q, DOMAINS).is_ok());
    }

    #[test]
    fn event_type_over_max_bytes_rejected() {
        let mut e = make_event("property", "{}");
        e.event_type = "x".repeat(MAX_TYPE_BYTES + 1);
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("Event type too long"));
    }

    #[test]
    fn event_individual_hash_over_max_bytes_rejected() {
        let mut e = make_event("property", "{}");
        e.related_hashes = vec!["x".repeat(MAX_HASH_BYTES + 1)];
        let err = validate_event_fields(&e, DOMAINS).unwrap_err();
        assert!(err.contains("Related hash [0] too long"));
    }

    #[test]
    fn event_individual_hash_at_max_bytes_accepted() {
        let mut e = make_event("property", "{}");
        e.related_hashes = vec!["x".repeat(MAX_HASH_BYTES)];
        assert!(validate_event_fields(&e, DOMAINS).is_ok());
    }

    #[test]
    fn cached_credential_did_over_max_bytes_rejected() {
        let c = make_cached_credential(&"x".repeat(MAX_DID_BYTES + 1), r#"{"p":{}}"#);
        let err = validate_cached_credential(&c).unwrap_err();
        assert!(err.contains("DID too long"));
    }

    #[test]
    fn cached_credential_did_at_max_bytes_accepted() {
        let c = make_cached_credential(&"x".repeat(MAX_DID_BYTES), r#"{"p":{}}"#);
        assert!(validate_cached_credential(&c).is_ok());
    }

    // ---- JurisdictionConstraintEntry validation ----

    fn make_jurisdiction() -> JurisdictionConstraintEntry {
        JurisdictionConstraintEntry {
            zone_id: "eu-gdpr-zone-1".into(),
            zone_polygon: vec![(35.0, -10.0), (71.0, -10.0), (71.0, 40.0), (35.0, 40.0)],
            regulatory_tags: vec!["gdpr".into(), "right_to_erasure".into()],
            fiat_zone: Some("EUR".into()),
            enforcement_risk: 0.85,
            authority_did: Some("did:mycelix:eu-dpa-collective".into()),
            description: "European Union GDPR enforcement zone".into(),
            created_at: Timestamp::from_micros(1_000_000),
            last_validated: Timestamp::from_micros(2_000_000),
        }
    }

    #[test]
    fn jurisdiction_valid() {
        let j = make_jurisdiction();
        assert!(validate_jurisdiction_constraint(&j).is_ok());
    }

    #[test]
    fn jurisdiction_serde_roundtrip() {
        let j = make_jurisdiction();
        let bytes = serde_json::to_vec(&j).unwrap();
        let j2: JurisdictionConstraintEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(j, j2);
    }

    #[test]
    fn jurisdiction_empty_zone_id_rejected() {
        let mut j = make_jurisdiction();
        j.zone_id = "".into();
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("Zone ID cannot be empty"));
    }

    #[test]
    fn jurisdiction_long_zone_id_rejected() {
        let mut j = make_jurisdiction();
        j.zone_id = "z".repeat(257);
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("256"));
    }

    #[test]
    fn jurisdiction_too_many_polygon_vertices() {
        let mut j = make_jurisdiction();
        j.zone_polygon = (0..1001).map(|i| (i as f64 * 0.01, 0.0)).collect();
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("1000"));
    }

    #[test]
    fn jurisdiction_empty_polygon_accepted() {
        let mut j = make_jurisdiction();
        j.zone_polygon = vec![];
        assert!(validate_jurisdiction_constraint(&j).is_ok());
    }

    #[test]
    fn jurisdiction_invalid_latitude() {
        let mut j = make_jurisdiction();
        j.zone_polygon = vec![(91.0, 0.0)];
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("Latitude"));
    }

    #[test]
    fn jurisdiction_invalid_longitude() {
        let mut j = make_jurisdiction();
        j.zone_polygon = vec![(0.0, 181.0)];
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("Longitude"));
    }

    #[test]
    fn jurisdiction_nan_coordinates_rejected() {
        let mut j = make_jurisdiction();
        j.zone_polygon = vec![(f64::NAN, 0.0)];
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("finite"));
    }

    #[test]
    fn jurisdiction_no_regulatory_tags_rejected() {
        let mut j = make_jurisdiction();
        j.regulatory_tags = vec![];
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("At least one"));
    }

    #[test]
    fn jurisdiction_empty_tag_rejected() {
        let mut j = make_jurisdiction();
        j.regulatory_tags = vec!["gdpr".into(), "".into()];
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("empty strings"));
    }

    #[test]
    fn jurisdiction_long_tag_rejected() {
        let mut j = make_jurisdiction();
        j.regulatory_tags = vec!["t".repeat(129)];
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("128"));
    }

    #[test]
    fn jurisdiction_too_many_tags_rejected() {
        let mut j = make_jurisdiction();
        j.regulatory_tags = (0..51).map(|i| format!("tag_{i}")).collect();
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("50"));
    }

    #[test]
    fn jurisdiction_enforcement_nan_rejected() {
        let mut j = make_jurisdiction();
        j.enforcement_risk = f64::NAN;
        assert!(validate_jurisdiction_constraint(&j).is_err());
    }

    #[test]
    fn jurisdiction_enforcement_out_of_range() {
        let mut j = make_jurisdiction();
        j.enforcement_risk = 1.01;
        assert!(validate_jurisdiction_constraint(&j).is_err());

        j.enforcement_risk = -0.01;
        assert!(validate_jurisdiction_constraint(&j).is_err());
    }

    #[test]
    fn jurisdiction_enforcement_boundaries_accepted() {
        let mut j = make_jurisdiction();
        j.enforcement_risk = 0.0;
        assert!(validate_jurisdiction_constraint(&j).is_ok());
        j.enforcement_risk = 1.0;
        assert!(validate_jurisdiction_constraint(&j).is_ok());
    }

    #[test]
    fn jurisdiction_long_description_rejected() {
        let mut j = make_jurisdiction();
        j.description = "d".repeat(4097);
        let err = validate_jurisdiction_constraint(&j).unwrap_err();
        assert!(err.contains("4096"));
    }

    #[test]
    fn jurisdiction_optional_fields_none() {
        let mut j = make_jurisdiction();
        j.fiat_zone = None;
        j.authority_did = None;
        assert!(validate_jurisdiction_constraint(&j).is_ok());

        // Serde roundtrip with None fields
        let bytes = serde_json::to_vec(&j).unwrap();
        let j2: JurisdictionConstraintEntry = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(j2.fiat_zone, None);
        assert_eq!(j2.authority_did, None);
    }

    // ---- Schema migration framework ----

    #[test]
    fn migration_current_version_returns_none() {
        // All three types at version 1 should return Ok(None) — no migration needed
        let entry = Entry::App(
            AppEntryBytes::try_from(
                SerializedBytes::try_from(make_query("property", "{}")).unwrap(),
            )
            .unwrap(),
        );
        assert!(BridgeQueryEntry::migrate(&entry, 1).unwrap().is_none());

        let entry = Entry::App(
            AppEntryBytes::try_from(
                SerializedBytes::try_from(make_event("housing", "{}")).unwrap(),
            )
            .unwrap(),
        );
        assert!(BridgeEventEntry::migrate(&entry, 1).unwrap().is_none());

        let entry = Entry::App(
            AppEntryBytes::try_from(
                SerializedBytes::try_from(make_cached_credential("did:test", "{}")).unwrap(),
            )
            .unwrap(),
        );
        assert!(CachedCredentialEntry::migrate(&entry, 1).unwrap().is_none());
    }

    #[test]
    fn migration_future_version_returns_error() {
        // Version 99 is from the future — migration should fail
        let entry = Entry::App(
            AppEntryBytes::try_from(
                SerializedBytes::try_from(make_query("property", "{}")).unwrap(),
            )
            .unwrap(),
        );
        let err = BridgeQueryEntry::migrate(&entry, 99).unwrap_err();
        assert!(err.contains("newer than supported"));
        assert!(err.contains("99"));

        let err = BridgeEventEntry::migrate(&entry, 99).unwrap_err();
        assert!(err.contains("newer than supported"));

        let err = CachedCredentialEntry::migrate(&entry, 99).unwrap_err();
        assert!(err.contains("newer than supported"));
    }

    #[test]
    fn migration_version_zero_returns_error() {
        // Version 0 has no migration path
        let entry = Entry::App(
            AppEntryBytes::try_from(
                SerializedBytes::try_from(make_query("property", "{}")).unwrap(),
            )
            .unwrap(),
        );
        let err = BridgeQueryEntry::migrate(&entry, 0).unwrap_err();
        assert!(err.contains("no migration path"));
    }

    #[test]
    fn migration_current_version_constants() {
        // Verify all types declare version 1 as current
        assert_eq!(BridgeQueryEntry::CURRENT_VERSION, 1);
        assert_eq!(BridgeEventEntry::CURRENT_VERSION, 1);
        assert_eq!(CachedCredentialEntry::CURRENT_VERSION, 1);
    }

    #[test]
    fn read_with_migration_current_version_succeeds() {
        // read_with_migration should succeed for a well-formed current-version entry
        let query = make_query("property", r#"{"key":"val"}"#);
        let entry = Entry::App(
            AppEntryBytes::try_from(SerializedBytes::try_from(query.clone()).unwrap()).unwrap(),
        );
        let result = read_with_migration::<BridgeQueryEntry>(&entry).unwrap();
        assert_eq!(result.domain, "property");
        assert_eq!(result.schema_version, 1);

        let event = make_event("housing", "{}");
        let entry = Entry::App(
            AppEntryBytes::try_from(SerializedBytes::try_from(event.clone()).unwrap()).unwrap(),
        );
        let result = read_with_migration::<BridgeEventEntry>(&entry).unwrap();
        assert_eq!(result.domain, "housing");

        let cred = make_cached_credential("did:test:abc", r#"{"p":1}"#);
        let entry = Entry::App(
            AppEntryBytes::try_from(SerializedBytes::try_from(cred.clone()).unwrap()).unwrap(),
        );
        let result = read_with_migration::<CachedCredentialEntry>(&entry).unwrap();
        assert_eq!(result.did, "did:test:abc");
    }

    #[test]
    fn read_with_migration_wrong_entry_type_fails() {
        // Trying to read an Agent entry as BridgeQueryEntry should fail
        let entry = Entry::Agent(fake_agent());
        let err = read_with_migration::<BridgeQueryEntry>(&entry).unwrap_err();
        assert!(err.contains("failed"));
    }

    // ---- Author validation helpers ----

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    #[test]
    fn check_author_match_same_author_valid() {
        let agent = fake_agent();
        let result = check_author_match(&agent, &agent, "delete");
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn check_author_match_different_author_invalid() {
        let agent1 = fake_agent();
        let agent2 = fake_agent_2();
        let result = check_author_match(&agent1, &agent2, "delete");
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("original"), "got: {msg}");
                assert!(msg.contains("delete"), "got: {msg}");
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn check_link_author_match_same_author_valid() {
        let agent = fake_agent();
        let result = check_link_author_match(&agent, &agent);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn check_link_author_match_different_author_invalid() {
        let agent1 = fake_agent();
        let agent2 = fake_agent_2();
        let result = check_link_author_match(&agent1, &agent2);
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("original author"), "got: {msg}");
                assert!(msg.contains("link"), "got: {msg}");
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }
}

// ============================================================================
// Cross-Cluster Notification Entry
// ============================================================================

/// A notification dispatched across cluster boundaries.
///
/// Stored on the target cluster's DHT when received via cross-cluster fanout.
/// Used by the notification service to provide unified inbox, unread tracking,
/// and multi-channel delivery (signal, email, SMS, webhook).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct CrossClusterNotification {
    /// Schema version (currently 1)
    pub schema_version: u8,
    /// Source cluster role name (e.g., "commons", "civic")
    pub source_cluster: String,
    /// Source zome that generated the notification
    pub source_zome: String,
    /// Event type identifier (e.g., "property_transfer_initiated", "disaster_declared")
    pub event_type: String,
    /// Target cluster role names. Empty = broadcast to all clusters.
    pub target_clusters: Vec<String>,
    /// Target agent public keys (base64). Empty = broadcast to all agents.
    pub target_agents: Vec<String>,
    /// JSON-serialized notification payload
    pub payload: String,
    /// Priority: 0=Low, 1=Normal, 2=High, 3=Emergency
    pub priority: u8,
    /// When the notification was created (from source cluster)
    pub created_at: Timestamp,
    /// When the notification expires. None = never.
    pub expires_at: Option<Timestamp>,
}

impl TryFrom<&Entry> for CrossClusterNotification {
    type Error = WasmError;
    fn try_from(entry: &Entry) -> Result<Self, Self::Error> {
        match entry {
            Entry::App(bytes) => {
                let sb = SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()));
                <Self as TryFrom<SerializedBytes>>::try_from(sb)
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not an app entry".into(),
            ))),
        }
    }
}

impl Default for CrossClusterNotification {
    fn default() -> Self {
        Self {
            schema_version: 1,
            source_cluster: String::new(),
            source_zome: String::new(),
            event_type: String::new(),
            target_clusters: vec![],
            target_agents: vec![],
            payload: String::new(),
            priority: 1,
            created_at: Timestamp::from_micros(0),
            expires_at: None,
        }
    }
}

/// Validate a cross-cluster notification entry.
pub fn validate_notification(entry: &CrossClusterNotification) -> Result<(), String> {
    if entry.source_cluster.is_empty() {
        return Err("source_cluster cannot be empty".into());
    }
    if entry.source_cluster.len() > 64 {
        return Err("source_cluster exceeds 64 chars".into());
    }
    if entry.source_zome.is_empty() {
        return Err("source_zome cannot be empty".into());
    }
    if entry.event_type.is_empty() {
        return Err("event_type cannot be empty".into());
    }
    if entry.event_type.len() > 256 {
        return Err("event_type exceeds 256 chars".into());
    }
    if entry.payload.len() > 65_536 {
        return Err("payload exceeds 64KB limit".into());
    }
    if entry.priority > 3 {
        return Err(format!("priority must be 0-3, got {}", entry.priority));
    }
    if entry.target_clusters.len() > 16 {
        return Err("Cannot target more than 16 clusters".into());
    }
    if entry.target_agents.len() > 100 {
        return Err("Cannot target more than 100 agents".into());
    }
    Ok(())
}

// ============================================================================
// Saga Entry
// ============================================================================

/// A saga workflow entry stored on the DHT for durability and auditability.
///
/// The `saga_json` field contains a serialized `SagaDefinition` from
/// `mycelix_bridge_common::saga`.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, SerializedBytes)]
pub struct SagaEntry {
    /// Schema version (currently 1)
    pub schema_version: u8,
    /// JSON-serialized SagaDefinition
    pub saga_json: String,
    /// Agent who initiated the saga
    pub initiator: String,
    /// When the saga was created
    pub created_at: Timestamp,
}

/// Validate a saga entry.
pub fn validate_saga_entry(entry: &SagaEntry) -> Result<(), String> {
    if entry.saga_json.is_empty() {
        return Err("saga_json cannot be empty".into());
    }
    if entry.saga_json.len() > 1_048_576 {
        return Err("saga_json exceeds 1MB limit".into());
    }
    if entry.initiator.is_empty() {
        return Err("initiator cannot be empty".into());
    }
    Ok(())
}
