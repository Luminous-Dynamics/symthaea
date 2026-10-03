// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Durable shared authorization-consumption domain.
//!
//! SQLite is the authoritative reservation/consumption state machine. The
//! external effect remains a separate boundary: an uncertain effect becomes
//! Indeterminate and requires explicit reconciliation.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::{Duration, DateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};
use url::Url;

use super::{
    ActionAuthorizationWitness, AuthorizationConsumptionError, AuthorizationLease,
    AuthorizationLeaseState, EpistemicAction, ExecutionOutcome, ExecutionReceipt,
};

#[derive(Debug)]
pub enum AuthorizationStoreError {
    Sqlite(rusqlite::Error),
    Consumption(AuthorizationConsumptionError),
    NotFound(String),
    InvalidState(String),
}
impl std::fmt::Display for AuthorizationStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "SQLite error: {e}"),
            Self::Consumption(e) => write!(f, "authorization consumption error: {e:?}"),
            Self::NotFound(id) => write!(f, "authorization lease not found: {id}"),
            Self::InvalidState(s) => write!(f, "invalid persisted authorization state: {s}"),
        }
    }
}
impl std::error::Error for AuthorizationStoreError {}
impl From<rusqlite::Error> for AuthorizationStoreError {
    fn from(e: rusqlite::Error) -> Self { Self::Sqlite(e) }
}
impl From<AuthorizationConsumptionError> for AuthorizationStoreError {
    fn from(e: AuthorizationConsumptionError) -> Self { Self::Consumption(e) }
}

/// Relying-party authorization for recovering one exact attempt.
///
/// The authority/authentication layer is responsible for validating the issuer
/// and policy. The durable store enforces only the non-negotiable structural
/// binding: one authorization instance, one attempt, one boundary, one action
/// digest, and one authority epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryAuthorizationWitness {
    pub authorization_instance: String,
    pub attempt_id: String,
    pub boundary_id: String,
    pub action_digest: String,
    pub policy: String,
    pub authority_epoch: u64,
    pub issued_at: String,
}

impl RecoveryAuthorizationWitness {
    pub fn is_bound_to(&self, lease: &AuthorizationLease) -> bool {
        !self.authorization_instance.is_empty()
            && !self.attempt_id.is_empty()
            && !self.boundary_id.is_empty()
            && !self.policy.is_empty()
            && !self.issued_at.is_empty()
            && self.authorization_instance == lease.authorization_instance
            && self.action_digest == lease.action_digest
            && self.policy == lease.policy
            && self.authority_epoch == lease.authority_epoch
    }
}

/// A durable shared consumption domain. Each operation uses a fresh connection,
/// allowing independent processes to contend on the same SQLite state machine.
/// The purpose of provider evidence presented to the verifier. A pre-entry lookup
/// is intentionally not interchangeable with terminal outcome evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderEvidenceKind {
    TerminalOutcome,
    PreEntryLookup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderVerificationPurpose {
    TerminalOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorizationClockPolicy {
    pub max_age_seconds: u64,
    pub allowed_skew_seconds: u64,
    pub require_expiry: bool,
}

impl Default for AuthorizationClockPolicy {
    fn default() -> Self {
        Self {
            max_age_seconds: 48 * 60 * 60,
            allowed_skew_seconds: 5 * 60,
            require_expiry: true,
        }
    }
}

impl AuthorizationClockPolicy {
    const CLOCK_SOURCE_ID: &'static str = "system-utc-wall-clock-v1";

    fn validate_configuration(&self) -> Result<(), AuthorizationStoreError> {
        if self.max_age_seconds == 0 || self.allowed_skew_seconds > self.max_age_seconds {
            return Err(AuthorizationStoreError::InvalidState(
                "authorization clock policy must have a non-zero max age and skew no greater than max age".into(),
            ));
        }
        Ok(())
    }

    fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:gis:authorization-clock-policy:v1\n");
        hasher.update(Self::CLOCK_SOURCE_ID.as_bytes());
        hasher.update(self.max_age_seconds.to_be_bytes());
        hasher.update(self.allowed_skew_seconds.to_be_bytes());
        hasher.update([self.require_expiry as u8]);
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    fn validate(
        &self,
        issued_at: &str,
        expires_at: Option<&str>,
        now: DateTime<Utc>,
    ) -> Result<(String, Option<String>), AuthorizationStoreError> {
        let issued = DateTime::parse_from_rfc3339(issued_at)
            .map_err(|_| AuthorizationConsumptionError::AuthorizationValidityWindowFailed)?
            .with_timezone(&Utc);

        let expiry = match expires_at {
            Some(value) => Some(
                DateTime::parse_from_rfc3339(value)
                    .map_err(|_| AuthorizationConsumptionError::AuthorizationValidityWindowFailed)?
                    .with_timezone(&Utc),
            ),
            None if self.require_expiry => {
                return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
            }
            None => None,
        };

        let skew = Duration::seconds(self.allowed_skew_seconds as i64);
        if issued > now + skew
            || now > issued + Duration::seconds(self.max_age_seconds as i64) + skew
        {
            return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
        }

        if let Some(expiry) = expiry {
            if expiry <= issued || now > expiry + skew {
                return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
            }
            if expiry > issued + Duration::from_secs(self.max_age_seconds) {
                return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
            }
        }

        let normalize = |value: DateTime<Utc>| {
            value.to_rfc3339_opts(SecondsFormat::Secs, true)
        };
        Ok((normalize(issued), expiry.map(normalize)))
    }
}

fn trusted_utc_now() -> Result<DateTime<Utc>, AuthorizationStoreError> {
    // This explicit UTC wall-clock source is committed into the policy digest.
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| AuthorizationStoreError::InvalidState(
            "system clock is before UNIX epoch".into(),
        ))?;
    DateTime::<Utc>::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())
        .ok_or_else(|| AuthorizationStoreError::InvalidState(
            "system clock timestamp is out of range".into(),
        ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderVerifierConfiguration {
    /// Stable relying-party control-domain identifier.
    pub relying_party_id: String,
    /// Stable relying-party-selected verifier implementation/profile identifier.
    pub verifier_id: String,
    /// Digest of the exact verifier configuration used for provider evidence.
    pub verifier_config_digest: String,
    /// Digest of the trust anchors/status inputs selected by the relying party.
    pub trust_anchor_digest: String,
    /// Digest of the evidence profile used to classify terminal provider evidence.
    pub evidence_profile_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderTerminalEvidence {
    pub kind: ProviderEvidenceKind,
    pub outcome: ExecutionOutcome,
    /// Stable provider-side evidence identifier, if the provider exposes one.
    pub evidence_id: String,
    /// Digest of the authenticated provider evidence payload.
    pub evidence_digest: String,
    pub action_id: String,
    pub action_digest: String,
    pub attempt_id: String,
    /// Logical operation identity; distinct from the execution attempt.
    pub operation_id: String,
    /// Native authorization replay unit supplied by the native authorization path.
    pub native_replay_identity: String,
    pub provider_idempotency_key: String,
    pub target_identity: String,
    pub audience: String,
    pub adapter: String,
    pub boundary_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedProviderOutcome {
    pub evidence: ProviderTerminalEvidence,
    /// Relying-party-pinned verifier implementation, trust anchors, and evidence profile.
    pub configuration: ProviderVerifierConfiguration,
    /// Digest of the verifier's authenticated verification statement.
    pub verification_digest: String,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderVerificationError {
    InvalidEvidenceKind,
    InvalidBinding,
    IndeterminateNotTerminal,
    VerificationFailed,
}

/// The verifier owns provider authentication and semantic authority. The durable
/// store deliberately does not pretend that an adapter's local enum is provider
/// truth; it only accepts a verifier result that explicitly affirms terminal
/// evidence for the exact frozen dispatch record.
pub trait ProviderEvidenceVerifier {
    fn verify(
        &self,
        purpose: ProviderVerificationPurpose,
        record: &DurableDispatchRecord,
        evidence: &ProviderTerminalEvidence,
    ) -> Result<VerifiedProviderOutcome, ProviderVerificationError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableDispatchRecord {
    pub authorization_instance: String,
    pub attempt_id: String,
    /// Logical operation identity supplied by the effect boundary.
    pub operation_id: String,
    /// Native replay identity derived by the native authorization path.
    pub native_replay_identity: String,
    pub action_id: String,
    pub action_digest: String,
    pub provider_idempotency_key: String,
    pub target_identity: String,
    pub audience: String,
    pub adapter: String,
    /// Stable boundary identity. It scopes attempt ownership without entering
    /// the shared action key, so separate boundary instances cannot claim one
    /// another's dispatch records.
    pub boundary_id: String,
}

impl DurableDispatchRecord {
    pub fn new(
        authorization_instance: impl Into<String>,
        attempt_id: impl Into<String>,
        operation_id: impl Into<String>,
        native_replay_identity: impl Into<String>,
        action_id: impl Into<String>,
        action_digest: impl Into<String>,
        provider_idempotency_key: impl Into<String>,
        effect: &super::ActionEffectBinding,
        boundary_id: impl Into<String>,
    ) -> Self {
        Self {
            authorization_instance: authorization_instance.into(),
            attempt_id: attempt_id.into(),
            operation_id: operation_id.into(),
            native_replay_identity: native_replay_identity.into(),
            action_id: action_id.into(),
            action_digest: action_digest.into(),
            provider_idempotency_key: provider_idempotency_key.into(),
            target_identity: effect.target_identity.clone(),
            audience: effect.audience.clone(),
            adapter: effect.adapter.clone(),
            boundary_id: boundary_id.into(),
        }
    }
}

pub struct SqliteAuthorizationStore {
    path: PathBuf,
    relying_party_id: String,
    clock_policy: AuthorizationClockPolicy,
}

impl SqliteAuthorizationStore {
    const LEGACY_RELYING_PARTY_ID: &'static str = "legacy-local";

    pub fn open(path: impl AsRef<Path>) -> Result<Self, AuthorizationStoreError> {
        Self::open_with_relying_party(path, Self::LEGACY_RELYING_PARTY_ID)
    }

    /// Open a durable authorization-consumption domain pinned to one relying party.
    pub fn open_with_relying_party(
        path: impl AsRef<Path>,
        relying_party_id: impl Into<String>,
    ) -> Result<Self, AuthorizationStoreError> {
        Self::open_with_relying_party_and_clock_policy(
            path,
            relying_party_id,
            AuthorizationClockPolicy::default(),
        )
    }

    pub fn open_with_relying_party_and_clock_policy(
        path: impl AsRef<Path>,
        relying_party_id: impl Into<String>,
        clock_policy: AuthorizationClockPolicy,
    ) -> Result<Self, AuthorizationStoreError> {
        clock_policy.validate_configuration()?;
        let path = path.as_ref().to_path_buf();
        let relying_party_id = relying_party_id.into();
        if relying_party_id.is_empty() {
            return Err(AuthorizationStoreError::InvalidState(
                "relying party id must not be empty".into(),
            ));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AuthorizationStoreError::InvalidState(e.to_string()))?;
        }
        let store = Self { path, relying_party_id, clock_policy };
        let mut connection = store.connection()?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;",
        )?;
        migrate_legacy_schema(&mut connection)?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS authorization_store_metadata (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS authorization_native_authority_pins (
               issuer TEXT PRIMARY KEY,
               authority_namespace TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS authorization_native_authority_pin_sets (
               pin_set_id TEXT NOT NULL,
               pin_set_digest TEXT PRIMARY KEY,
               snapshot TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS authorization_leases (
               authorization_instance TEXT PRIMARY KEY,
               action_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               support_digest TEXT NOT NULL,
               policy TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               remaining_executions INTEGER NOT NULL,
               state TEXT NOT NULL,
               attempt_id TEXT,
               boundary_id TEXT,
               validity_issued_at TEXT,
               validity_expires_at TEXT,
               validity_policy_digest TEXT
             );
             CREATE TABLE IF NOT EXISTS authorization_receipts (
               authorization_instance TEXT NOT NULL,
               action_id TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               phase TEXT NOT NULL,
               outcome TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               boundary_id TEXT,
               PRIMARY KEY(authorization_instance, attempt_id, phase)
             );
             CREATE TABLE IF NOT EXISTS authorization_recovery_markers (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               boundary_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               marker TEXT NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id, marker)
             );
             CREATE TABLE IF NOT EXISTS authorization_terminal_evidence (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               operation_id TEXT NOT NULL,
               native_replay_identity TEXT NOT NULL,
               native_authority_namespace TEXT,
               native_authorization_id TEXT,
               native_replay_derivation_digest TEXT,
               native_authority_pin_set_id TEXT,
               native_authority_pin_set_digest TEXT,
               relying_party_id TEXT,
               boundary_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               provider_idempotency_key TEXT NOT NULL,
               target_identity TEXT NOT NULL,
               audience TEXT NOT NULL,
               adapter TEXT,
               outcome TEXT NOT NULL,
               evidence_id TEXT NOT NULL,
               evidence_digest TEXT NOT NULL,
               verifier_id TEXT NOT NULL,
               verifier_config_digest TEXT NOT NULL,
               trust_anchor_digest TEXT NOT NULL,
               evidence_profile_digest TEXT NOT NULL,
               verification_digest TEXT NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id)
             );
             CREATE TABLE IF NOT EXISTS authorization_dispatches (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               operation_id TEXT NOT NULL,
               native_replay_identity TEXT NOT NULL,
               native_authority_namespace TEXT,
               native_authorization_id TEXT,
               native_replay_derivation_digest TEXT,
               native_authority_pin_set_id TEXT,
               native_authority_pin_set_digest TEXT,
               validity_issued_at TEXT,
               validity_expires_at TEXT,
               validity_policy_digest TEXT,
               relying_party_id TEXT,
               action_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               provider_idempotency_key TEXT NOT NULL,
               target_identity TEXT NOT NULL,
               audience TEXT NOT NULL,
               adapter TEXT NOT NULL,
               boundary_id TEXT NOT NULL,
               state TEXT NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id)
             );",
        )?;
        ensure_column(&mut connection, "authorization_leases", "boundary_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_leases", "validity_issued_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_leases", "validity_expires_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_leases", "validity_policy_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "operation_id", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_dispatches", "native_replay_identity", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_dispatches", "native_issuer", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "native_authority_namespace", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "native_authorization_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "native_replay_derivation_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "native_authority_pin_set_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "native_authority_pin_set_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "validity_issued_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "validity_expires_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "validity_policy_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "relying_party_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "operation_id", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "native_replay_identity", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "native_issuer", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "native_authority_namespace", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "native_authorization_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "native_replay_derivation_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "native_authority_pin_set_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "native_authority_pin_set_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "validity_issued_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "validity_expires_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "validity_policy_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "adapter", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "relying_party_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_receipts", "boundary_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "verifier_config_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "trust_anchor_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "evidence_profile_digest", "TEXT NOT NULL DEFAULT ''")?;
        connection.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS authorization_lease_attempt_id_uq
               ON authorization_leases(attempt_id)
               WHERE boundary_id IS NOT NULL AND attempt_id IS NOT NULL;
             CREATE UNIQUE INDEX IF NOT EXISTS authorization_dispatch_attempt_id_uq
               ON authorization_dispatches(attempt_id)
               WHERE boundary_id IS NOT NULL AND attempt_id IS NOT NULL;
             CREATE UNIQUE INDEX IF NOT EXISTS authorization_dispatch_native_replay_uq
               ON authorization_dispatches(native_replay_identity)
               WHERE native_replay_identity <> '';
             CREATE UNIQUE INDEX IF NOT EXISTS authorization_dispatch_operation_uq
               ON authorization_dispatches(operation_id)
               WHERE operation_id <> '';
             DROP INDEX IF EXISTS authorization_dispatch_action_fence_idx;
             CREATE INDEX authorization_dispatch_action_fence_idx
               ON authorization_dispatches(relying_party_id, target_identity, action_digest, state);",
        )?;

        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        validate_native_authority_pin_set(&tx)?;
        let clock_policy_digest = store.clock_policy.digest();
        let configured_policy: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata WHERE key='authorization_clock_policy_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match configured_policy {
            Some(existing) if existing != clock_policy_digest => {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "authorization clock policy mismatch: store is pinned to {existing}"
                )));
            }
            Some(_) => {}
            None => {
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value) VALUES('authorization_clock_policy_digest',?1)",
                    params![clock_policy_digest.as_str()],
                )?;
            }
        }

        let configured: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata WHERE key='relying_party_id'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match configured {
            Some(existing) if existing != store.relying_party_id => {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "relying party mismatch: store is pinned to {existing}"
                )));
            }
            Some(_) => {}
            None => {
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value) VALUES('relying_party_id',?1)",
                    params![store.relying_party_id.as_str()],
                )?;
                tx.execute(
                    "UPDATE authorization_dispatches
                     SET relying_party_id=?1
                     WHERE relying_party_id IS NULL OR relying_party_id=''",
                    params![store.relying_party_id.as_str()],
                )?;
                tx.execute(
                    "UPDATE authorization_terminal_evidence
                     SET relying_party_id=?1
                     WHERE relying_party_id IS NULL OR relying_party_id=''",
                    params![store.relying_party_id.as_str()],
                )?;
            }
        }
        tx.commit()?;
        Ok(store)
    }

    fn connection(&self) -> Result<Connection, AuthorizationStoreError> {
        Ok(Connection::open(&self.path)?)
    }

    pub fn relying_party_id(&self) -> &str {
        &self.relying_party_id
    }

    /// Pin one issuer to one authority namespace for this relying-party domain.
    /// The mapping is write-once; attempting to change an established pin fails closed.
    pub fn pin_native_authority_namespace(
        &self,
        issuer: &str,
        authority_namespace: &str,
    ) -> Result<(), AuthorizationStoreError> {
        if issuer.is_empty() || authority_namespace.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT authority_namespace
                 FROM authorization_native_authority_pins
                 WHERE issuer=?1",
                params![issuer],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(namespace) = existing.as_deref() {
            if namespace != authority_namespace {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "authority namespace pin for {issuer} is already established as {namespace}"
                )));
            }
            tx.commit()?;
            return Ok(());
        }

        let normalized_issuer = normalize_native_issuer(issuer);
        let mut stmt = tx.prepare(
            "SELECT issuer,authority_namespace
             FROM authorization_native_authority_pins",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let existing_issuer: String = row.get(0)?;
            let existing_namespace: String = row.get(1)?;
            if normalize_native_issuer(&existing_issuer) == normalized_issuer
                && existing_namespace != authority_namespace
            {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "authority namespace collision: {issuer} normalizes with {existing_issuer},                      but the established namespaces differ"
                )));
            }
        }

        tx.execute(
            "INSERT INTO authorization_native_authority_pins(issuer,authority_namespace)
             VALUES (?1,?2)",
            params![issuer, authority_namespace],
        )?;
        tx.commit()?;
        Ok(())
    }

fn normalize_native_issuer(issuer: &str) -> String {
    let candidate = issuer
        .find(':')
        .map(|colon| format!("{}{}", issuer[..colon].to_ascii_lowercase(), &issuer[colon..]))
        .unwrap_or_else(|| issuer.to_owned());

    let candidate = if candidate.len() > 5 {
        let prefix = candidate.as_bytes();
        let http = prefix[..5].eq_ignore_ascii_case(b"http:");
        let https = candidate.len() > 6 && prefix[..6].eq_ignore_ascii_case(b"https:");
        if (http || https)
            && !candidate[candidate.find(':').unwrap_or(0) + 1..].starts_with("//")
        {
            let colon = candidate.find(':').unwrap_or(0);
            format!("{}//{}", &candidate[..colon + 1], &candidate[colon + 1..])
        } else {
            candidate
        }
    } else {
        candidate
    };

    let Ok(mut url) = Url::parse(&candidate) else {
        return issuer.to_owned();
    };

    if (url.scheme() == "http" && url.port() == Some(80))
        || (url.scheme() == "https" && url.port() == Some(443))
    {
        let _ = url.set_port(None);
    }

    if let Some(host) = url.host_str().map(str::to_owned) {
        let normalized_host = host.strip_suffix('.').unwrap_or(&host);
        if normalized_host != host {
            if url.set_host(Some(normalized_host)).is_err() {
                return issuer.to_owned();
            }
        }
        let path = url.path().trim_end_matches('/');
        url.set_path(path);
        return url.to_string();
    }

    url.to_string()
}

fn validate_native_authority_pin_set(
    tx: &Transaction<'_>,
) -> Result<(), AuthorizationStoreError> {
    let mut stmt = tx.prepare(
        "SELECT issuer,authority_namespace
         FROM authorization_native_authority_pins
         ORDER BY issuer ASC",
    )?;
    let mut rows = stmt.query([])?;
    let mut seen: Vec<(String, String)> = Vec::new();
    while let Some(row) = rows.next()? {
        let issuer: String = row.get(0)?;
        let namespace: String = row.get(1)?;
        let normalized = normalize_native_issuer(&issuer);
        if let Some((existing_issuer, existing_namespace)) =
            seen.iter().find(|(existing, _)| normalize_native_issuer(existing) == normalized)
        {
            if existing_namespace != &namespace {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "invalid native authority pin set: {issuer} normalizes with {existing_issuer} but namespaces differ"
                )));
            }
        } else {
            seen.push((issuer, namespace));
        }
    }
    Ok(())
}

    fn native_authority_pin_set_id(&self) -> String {
        format!("{}:native-authority-pins:v1", self.relying_party_id)
    }

    fn persist_native_authority_pin_set_snapshot(
        &self,
        tx: &Transaction<'_>,
    ) -> Result<(String, String), AuthorizationStoreError> {
        let mut stmt = tx.prepare(
            "SELECT issuer,authority_namespace
             FROM authorization_native_authority_pins
             ORDER BY issuer ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut canonical = Vec::with_capacity(64);
        canonical.extend_from_slice(b"symthaea:gis:native-pin-set:v1\n");
        for row in rows {
            let (issuer, namespace) = row?;
            append_len_prefixed(&mut canonical, issuer.as_bytes());
            append_len_prefixed(&mut canonical, namespace.as_bytes());
        }

        let pin_set_id = self.native_authority_pin_set_id();
        let pin_set_digest = format!("sha256:{}", hex::encode(Sha256::digest(&canonical)));
        let snapshot = hex::encode(&canonical);

        tx.execute(
            "INSERT OR IGNORE INTO authorization_native_authority_pin_sets
             (pin_set_id,pin_set_digest,snapshot)
             VALUES (?1,?2,?3)",
            params![pin_set_id.as_str(), pin_set_digest.as_str(), snapshot.as_str()],
        )?;

        let stored: String = tx.query_row(
            "SELECT snapshot
             FROM authorization_native_authority_pin_sets
             WHERE pin_set_id=?1 AND pin_set_digest=?2",
            params![pin_set_id.as_str(), pin_set_digest.as_str()],
            |row| row.get(0),
        )?;
        if stored != snapshot {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }

        Ok((pin_set_id, pin_set_digest))
    }

    fn validate_native_authority_pin_set_snapshot(
        &self,
        tx: &Transaction<'_>,
        pin_set_id: Option<&str>,
        pin_set_digest: Option<&str>,
    ) -> Result<(), AuthorizationStoreError> {
        let (Some(pin_set_id), Some(pin_set_digest)) = (pin_set_id, pin_set_digest) else {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        };
        if pin_set_id != self.native_authority_pin_set_id() {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }
        let snapshot: String = tx.query_row(
            "SELECT snapshot
             FROM authorization_native_authority_pin_sets
             WHERE pin_set_id=?1 AND pin_set_digest=?2",
            params![pin_set_id, pin_set_digest],
            |row| row.get(0),
        ).optional()?
            .ok_or_else(|| AuthorizationConsumptionError::InvalidNativeReplayProvenance)?;
        let canonical = hex::decode(&snapshot)
            .map_err(|_| AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            ))?;
        let derived = format!("sha256:{}", hex::encode(Sha256::digest(&canonical)));
        if derived != pin_set_digest {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }
        Ok(())
    }

    fn pinned_native_authority_namespace(
        &self,
        issuer: &str,
    ) -> Result<String, AuthorizationStoreError> {
        if issuer.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }
        let normalized = normalize_native_issuer(issuer);
        let connection = self.connection()?;
        let mut stmt = connection.prepare(
            "SELECT issuer,authority_namespace
             FROM authorization_native_authority_pins",
        )?;
        let mut rows = stmt.query([])?;
        let mut matched_namespace: Option<String> = None;
        while let Some(row) = rows.next()? {
            let pinned_issuer: String = row.get(0)?;
            if normalize_native_issuer(&pinned_issuer) != normalized {
                continue;
            }
            let namespace: String = row.get(1)?;
            match matched_namespace.as_deref() {
                None => matched_namespace = Some(namespace),
                Some(existing) if existing == namespace => {}
                Some(_) => {
                    return Err(
                        AuthorizationConsumptionError::InvalidNativeReplayProvenance.into()
                    );
                }
            }
        }
        matched_namespace
            .ok_or_else(|| AuthorizationConsumptionError::InvalidNativeReplayProvenance.into())
    }

    pub fn register_lease(&self, lease: &AuthorizationLease) -> Result<(), AuthorizationStoreError> {
        let connection = self.connection()?;
        connection.execute(
            "INSERT INTO authorization_leases
             (authorization_instance, action_id, action_digest, support_digest, policy, authority_epoch,
              remaining_executions, state, attempt_id, boundary_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,NULL)",
            params![
                lease.authorization_instance, lease.action_id, lease.action_digest,
                lease.support_digest, lease.policy, lease.authority_epoch as i64,
                lease.remaining_executions as i64, encode_state(&lease.state),
                state_attempt(&lease.state)
            ],
        )?;
        Ok(())
    }

    pub fn prepare_for_execution(
        &self, witness: &ActionAuthorizationWitness, action: &EpistemicAction,
        current_frame: &str, attempt_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        // Effect-bound actions must use the boundary-owned lifecycle so the
        // durable dispatch record, native replay provenance, same-action fence,
        // and provider verifier controls cannot be bypassed by the legacy API.
        if action.effect_binding().is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let bound: Option<String> = tx
            .query_row(
                "SELECT boundary_id FROM authorization_leases
                 WHERE authorization_instance=?1",
                params![witness.authorization_instance.as_str()],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        if bound.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;
        lease.prepare_for_execution(witness, action, current_frame, attempt_id)?;
        tx.execute(
            "UPDATE authorization_leases SET state='prepared', attempt_id=?2
             WHERE authorization_instance=?1 AND state='ready' AND remaining_executions>0
               AND boundary_id IS NULL",
            params![witness.authorization_instance.as_str(), attempt_id],
        )?;
        if tx.changes() != 1 {
            return Err(AuthorizationConsumptionError::NotReady.into());
        }
        tx.commit()?;
        Ok(())
    }

    /// Prepare an effectful attempt with durable boundary ownership.
    ///
    /// Persisting the boundary while the lease is still Prepared means a crash
    /// before DispatchPending cannot leave the reservation ownerless.
    /// Bound attempt IDs are unique within this durable state domain.
    pub fn prepare_for_execution_bound(
        &self,
        witness: &ActionAuthorizationWitness,
        action: &EpistemicAction,
        current_frame: &str,
        attempt_id: &str,
        boundary_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        if boundary_id.is_empty() || attempt_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let (validity_issued_at, validity_expires_at) = self.clock_policy.validate(
            &witness.issued_at,
            witness.expires_at.as_deref(),
            trusted_utc_now()?,
        )?;
        let validity_policy_digest = self.clock_policy.digest();

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let prior_owner: Option<(String, String)> = tx
            .query_row(
                "SELECT authorization_instance,boundary_id
                 FROM authorization_leases
                 WHERE attempt_id=?1 AND boundary_id IS NOT NULL
                   AND authorization_instance<>?2",
                params![attempt_id, witness.authorization_instance.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if prior_owner.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;
        lease.prepare_for_execution(witness, action, current_frame, attempt_id)?;

        let persisted_validity: Option<(String, Option<String>, String)> = tx
            .query_row(
                "SELECT validity_issued_at,validity_expires_at,validity_policy_digest
                 FROM authorization_leases
                 WHERE authorization_instance=?1",
                params![witness.authorization_instance.as_str()],
                |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)),
            )
            .optional()?;
        match persisted_validity {
            Some((issued, expires, policy))
                if !issued.is_empty() || expires.is_some() || !policy.is_empty() =>
            {
                if issued != validity_issued_at
                    || expires.as_deref() != validity_expires_at.as_deref()
                    || policy != validity_policy_digest
                {
                    return Err(
                        AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into()
                    );
                }
            }
            _ => {
                tx.execute(
                    "UPDATE authorization_leases
                     SET validity_issued_at=?2,validity_expires_at=?3,validity_policy_digest=?4
                     WHERE authorization_instance=?1",
                    params![
                        witness.authorization_instance.as_str(),
                        validity_issued_at.as_str(),
                        validity_expires_at.as_deref(),
                        validity_policy_digest.as_str(),
                    ],
                )?;
            }
        }

        let changed = tx.execute(
            "UPDATE authorization_leases
             SET state='prepared', attempt_id=?2, boundary_id=?3
             WHERE authorization_instance=?1 AND state='ready'
               AND remaining_executions>0 AND boundary_id IS NULL",
            params![witness.authorization_instance.as_str(), attempt_id, boundary_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::NotReady.into());
        }
        tx.commit()?;
        Ok(())
    }

    /// Durably cross the pre-dispatch fence. This must commit before the
    /// executor enters the external effect sink.
    pub fn mark_dispatch_pending(
        &self, authorization_instance: &str, attempt_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if load_lease_boundary(&tx, authorization_instance)?.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        lease.mark_dispatch_pending(attempt_id)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='dispatch_pending', attempt_id=?2
             WHERE authorization_instance=?1 AND state='prepared' AND attempt_id=?2",
            params![authorization_instance, attempt_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        tx.commit()?;
        Ok(())
    }

    /// Create the immutable provider-entry record and cross the pre-dispatch
    /// fence in one durable transaction. Effectful callers should use this path
    /// rather than the legacy attempt-only transition because it freezes every
    /// material field that the later provider boundary must consume.
    /// Derive the native replay identity at the protected effect boundary
    /// from the pinned authority namespace and native authorization ID.
    ///
    /// This is the preferred entry point for production native authorization
    /// handoffs. The former free-form replay-identity APIs are retained only as
    /// deprecated hard-fail shims; they cannot prove their derivation inputs.
    /// Canonical native-authorization entry point. The issuer is only a
    /// lookup key; the relying-party-pinned authority namespace is resolved
    /// inside the boundary before replay identity derivation.
    pub fn mark_dispatch_pending_bound_from_pinned_native_authority(
        &self,
        authorization_instance: &str,
        attempt_id: &str,
        action: &EpistemicAction,
        expected_effect: &super::ActionEffectBinding,
        boundary_id: &str,
        operation_id: &str,
        issuer: &str,
        native_authorization_id: &str,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        let authority_namespace = self.pinned_native_authority_namespace(issuer)?;
        let replay = super::NativeReplayDerivation::derive(
            authority_namespace,
            native_authorization_id,
        )
        .map_err(|_| AuthorizationConsumptionError::InvalidNativeReplayProvenance)?;
        self.mark_dispatch_pending_bound_with_provenance(
            authorization_instance,
            attempt_id,
            action,
            expected_effect,
            boundary_id,
            operation_id,
            &replay.native_replay_identity,
            &replay,
        )
    }

    #[deprecated(note = "use mark_dispatch_pending_bound_from_pinned_native_authority")]
    pub fn mark_dispatch_pending_bound_from_native_authority(
        &self,
        _authorization_instance: &str,
        _attempt_id: &str,
        _action: &EpistemicAction,
        _expected_effect: &super::ActionEffectBinding,
        _boundary_id: &str,
        _operation_id: &str,
        _authority_namespace: &str,
        _native_authorization_id: &str,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into())
    }

    #[deprecated(note = "use mark_dispatch_pending_bound_from_pinned_native_authority")]
    pub fn mark_dispatch_pending_bound(
        &self,
        _authorization_instance: &str,
        _attempt_id: &str,
        _action: &EpistemicAction,
        _expected_effect: &super::ActionEffectBinding,
        _boundary_id: &str,
        _operation_id: &str,
        _native_replay_identity: &str,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into())
    }

    fn mark_dispatch_pending_bound_with_provenance(
        &self,
        authorization_instance: &str,
        attempt_id: &str,
        action: &EpistemicAction,
        expected_effect: &super::ActionEffectBinding,
        boundary_id: &str,
        operation_id: &str,
        native_replay_identity: &str,
        native_replay_provenance: &super::NativeReplayDerivation,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        if action.effect_binding.as_ref() != Some(expected_effect)
            || boundary_id.is_empty()
            || operation_id.is_empty()
            || native_replay_identity.is_empty()
            || native_replay_provenance.authority_namespace.is_empty()
            || native_replay_provenance.native_authorization_id.is_empty()
            || native_replay_provenance.derivation_digest.is_empty()
            || native_replay_provenance.native_replay_identity != native_replay_identity
        {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }
        let derived = super::NativeReplayDerivation::derive(
            &native_replay_provenance.authority_namespace,
            &native_replay_provenance.native_authorization_id,
        )
        .map_err(|_| AuthorizationConsumptionError::InvalidNativeReplayProvenance)?;
        if derived.native_replay_identity != native_replay_identity
            || derived.derivation_digest != native_replay_provenance.derivation_digest
        {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let prior_owner: Option<(String, String)> = tx
            .query_row(
                "SELECT authorization_instance,boundary_id
                 FROM authorization_leases
                 WHERE attempt_id=?1 AND boundary_id IS NOT NULL
                   AND authorization_instance<>?2",
                params![attempt_id, authorization_instance],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if prior_owner.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let current_boundary = load_lease_boundary(&tx, authorization_instance)?;
        if current_boundary.as_deref().is_some_and(|id| id != boundary_id) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        let expected_digest = action.canonical_action_digest();
        if lease.action_id != action.id || lease.action_digest != expected_digest {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let occupied: Option<(String, String, String, String)> = tx
            .query_row(
                "SELECT authorization_instance,attempt_id,boundary_id,state
                 FROM authorization_dispatches
                 WHERE relying_party_id=?1 AND target_identity=?2 AND action_digest=?3
                   AND state IN ('dispatch_pending','invoked','indeterminate','succeeded')
                 LIMIT 1",
                params![
                    self.relying_party_id.as_str(),
                    expected_effect.target_identity.as_str(),
                    expected_digest.as_str()
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        if let Some((_, _, _, state)) = occupied {
            if state == "succeeded" {
                return Err(AuthorizationConsumptionError::ActionAlreadyClosed.into());
            }
            return Err(AuthorizationConsumptionError::ActionAlreadyInFlight.into());
        }

        lease.mark_dispatch_pending(attempt_id)?;
        let (native_authority_pin_set_id, native_authority_pin_set_digest) =
            self.persist_native_authority_pin_set_snapshot(&tx)?;

        let lease_validity: (String, Option<String>, String) = tx.query_row(
            "SELECT validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_leases
             WHERE authorization_instance=?1",
            params![authorization_instance],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)),
        )?;
        if self.clock_policy.validate(
            &lease_validity.0,
            lease_validity.1.as_deref(),
            trusted_utc_now()?,
        ).is_err()
            || lease_validity.2 != self.clock_policy.digest()
        {
            let authority_epoch = lease.authority_epoch;
            let changed = tx.execute(
                "UPDATE authorization_leases
                 SET state='expired',attempt_id=NULL,boundary_id=NULL
                 WHERE authorization_instance=?1 AND state='prepared' AND attempt_id=?2",
                params![authorization_instance, attempt_id],
            )?;
            if changed != 1 {
                return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
            }
            tx.execute(
                "INSERT OR IGNORE INTO authorization_recovery_markers
                 (authorization_instance,attempt_id,boundary_id,action_digest,authority_epoch,marker)
                 VALUES (?1,?2,?3,?4,?5,'not_entered_validity')",
                params![
                    authorization_instance,
                    attempt_id,
                    boundary_id,
                    action.canonical_action_digest(),
                    authority_epoch as i64,
                ],
            )?;
            tx.commit()?;
            return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
        }

        let record = DurableDispatchRecord::new(
            authorization_instance,
            attempt_id,
            operation_id,
            native_replay_identity,
            &action.id,
            &expected_digest,
            &lease
                .provider_idempotency_key_for_native_replay(
                    native_replay_identity,
                    &expected_effect.target_identity,
                )
                .map_err(|_| AuthorizationConsumptionError::InvalidBinding)?,
            expected_effect,
            boundary_id,
        );
        if current_boundary.is_none() {
            tx.execute(
                "UPDATE authorization_leases SET boundary_id=?2
                 WHERE authorization_instance=?1 AND state='prepared'
                   AND attempt_id=?3 AND boundary_id IS NULL",
                params![authorization_instance, boundary_id, attempt_id],
            )?;
        }
        tx.execute(
            "INSERT INTO authorization_dispatches
             (authorization_instance,attempt_id,operation_id,native_replay_identity,
              native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
              native_authority_pin_set_id,native_authority_pin_set_digest,
              validity_issued_at,validity_expires_at,validity_policy_digest,relying_party_id,
              action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,boundary_id,state)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,'dispatch_pending')",
            params![
                record.authorization_instance, record.attempt_id, record.operation_id,
                record.native_replay_identity,
                native_replay_provenance.authority_namespace.as_str(),
                native_replay_provenance.native_authorization_id.as_str(),
                native_replay_provenance.derivation_digest.as_str(),
                native_authority_pin_set_id.as_str(),
                native_authority_pin_set_digest.as_str(),
                lease_validity.0.as_str(),
                lease_validity.1.as_deref(),
                lease_validity.2.as_str(),
                self.relying_party_id.as_str(),
                record.action_id, record.action_digest, record.provider_idempotency_key,
                record.target_identity, record.audience, record.adapter, record.boundary_id],
        )?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='dispatch_pending', attempt_id=?2
             WHERE authorization_instance=?1 AND state='prepared' AND attempt_id=?2",
            params![authorization_instance, attempt_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        tx.commit()?;
        Ok(record)
    }

    /// Durably record provider entry after DispatchPending has committed.
    /// Cross the provider-entry boundary only for the exact immutable record
    /// created before dispatch. All identity/effect fields are checked again,
    /// preventing a stale executor from substituting a different sink or key.
    pub fn mark_invoked_bound(
        &self, record: &DurableDispatchRecord,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx.query_row(
            "SELECT operation_id,native_replay_identity,action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,boundary_id,state
             FROM authorization_dispatches WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((
                r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?,
                r.get::<_,String>(3)?, r.get::<_,String>(4)?, r.get::<_,String>(5)?,
                r.get::<_,String>(6)?, r.get::<_,String>(7)?, r.get::<_,String>(8)?,
                r.get::<_,String>(9)?,
            )),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;
        if row.0 != record.operation_id || row.1 != record.native_replay_identity
            || row.2 != record.action_id || row.3 != record.action_digest
            || row.4 != record.provider_idempotency_key || row.5 != record.target_identity
            || row.6 != record.audience || row.7 != record.adapter
            || row.8 != record.boundary_id || row.9 != "dispatch_pending"
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let persisted_relying_party: Option<String> = tx.query_row(
            "SELECT relying_party_id FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| r.get(0),
        )?;
        if persisted_relying_party.as_deref() != Some(self.relying_party_id.as_str()) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let native_provenance: (
            Option<String>, Option<String>, Option<String>, Option<String>, Option<String>,
            Option<String>, Option<String>, Option<String>
        ) = tx.query_row(
            "SELECT native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
                    native_authority_pin_set_id,native_authority_pin_set_digest,
                    validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?)),
        )?;
        Self::validate_persisted_native_replay_provenance(
            record,
            native_provenance.0.as_deref(),
            native_provenance.1.as_deref(),
            native_provenance.2.as_deref(),
        )?;
        self.validate_native_authority_pin_set_snapshot(
            &tx,
            native_provenance.3.as_deref(),
            native_provenance.4.as_deref(),
        )?;
        self.validate_persisted_authorization_validity(
            native_provenance.5.as_deref(),
            native_provenance.6.as_deref(),
            native_provenance.7.as_deref(),
            false,
        )?;
        if self.validate_persisted_authorization_validity(
            native_provenance.5.as_deref(),
            native_provenance.6.as_deref(),
            native_provenance.7.as_deref(),
            true,
        ).is_err() {
            let changed = tx.execute(
                "UPDATE authorization_dispatches
                 SET state='not_entered'
                 WHERE authorization_instance=?1 AND attempt_id=?2
                   AND boundary_id=?3 AND state='dispatch_pending'",
                params![
                    record.authorization_instance.as_str(),
                    record.attempt_id.as_str(),
                    record.boundary_id.as_str(),
                ],
            )?;
            if changed != 1 {
                return Err(AuthorizationConsumptionError::AttemptMismatch.into());
            }
            let changed = tx.execute(
                "UPDATE authorization_leases
                 SET state='expired',attempt_id=NULL,boundary_id=NULL
                 WHERE authorization_instance=?1 AND state='dispatch_pending'
                   AND attempt_id=?2 AND boundary_id=?3",
                params![
                    record.authorization_instance.as_str(),
                    record.attempt_id.as_str(),
                    record.boundary_id.as_str(),
                ],
            )?;
            if changed != 1 {
                return Err(AuthorizationConsumptionError::AttemptMismatch.into());
            }
            let authority_epoch: i64 = tx.query_row(
                "SELECT authority_epoch FROM authorization_leases
                 WHERE authorization_instance=?1",
                params![record.authorization_instance.as_str()],
                |row| row.get(0),
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO authorization_recovery_markers
                 (authorization_instance,attempt_id,boundary_id,action_digest,authority_epoch,marker)
                 VALUES (?1,?2,?3,?4,?5,'not_entered_validity')",
                params![
                    record.authorization_instance.as_str(),
                    record.attempt_id.as_str(),
                    record.boundary_id.as_str(),
                    record.action_digest.as_str(),
                    authority_epoch,
                ],
            )?;
            tx.commit()?;
            return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
        }
        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        lease.mark_invoked(&record.attempt_id)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='invoked', attempt_id=?2
             WHERE authorization_instance=?1 AND state='dispatch_pending' AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
        )?;
        if changed != 1 { return Err(AuthorizationConsumptionError::AttemptMismatch.into()); }
        let changed = tx.execute(
            "UPDATE authorization_dispatches SET state='invoked'
             WHERE authorization_instance=?1 AND attempt_id=?2 AND state='dispatch_pending'",
            params![record.authorization_instance, record.attempt_id],
        )?;
        if changed != 1 { return Err(AuthorizationConsumptionError::AttemptMismatch.into()); }
        tx.commit()?;
        Ok(())
    }

    pub fn mark_invoked(
        &self, authorization_instance: &str, attempt_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if load_lease_boundary(&tx, authorization_instance)?.is_some()
            || dispatch_boundary(&tx, authorization_instance, attempt_id)?.is_some()
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        lease.mark_invoked(attempt_id)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state='invoked', attempt_id=?2
             WHERE authorization_instance=?1 AND state='dispatch_pending' AND attempt_id=?2",
            params![authorization_instance, attempt_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        tx.commit()?;
        Ok(())
    }

    fn validate_persisted_authorization_validity(
        &self,
        issued_at: Option<&str>,
        expires_at: Option<&str>,
        policy_digest: Option<&str>,
        enforce_now: bool,
    ) -> Result<(), AuthorizationStoreError> {
        if policy_digest != Some(self.clock_policy.digest().as_str()) {
            return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
        }
        let issued_at = issued_at
            .ok_or_else(|| AuthorizationConsumptionError::AuthorizationValidityWindowFailed)?;
        let issued = DateTime::parse_from_rfc3339(issued_at)
            .map_err(|_| AuthorizationConsumptionError::AuthorizationValidityWindowFailed)?
            .with_timezone(&Utc);
        let expiry = match expires_at {
            Some(value) => Some(
                DateTime::parse_from_rfc3339(value)
                    .map_err(|_| AuthorizationConsumptionError::AuthorizationValidityWindowFailed)?
                    .with_timezone(&Utc),
            ),
            None if self.clock_policy.require_expiry => {
                return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
            }
            None => None,
        };
        if expiry.is_some_and(|value| value <= issued)
            || expiry.is_some_and(|value| {
                value > issued + Duration::seconds(self.clock_policy.max_age_seconds as i64)
            })
        {
            return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
        }
        if enforce_now {
            self.clock_policy.validate(issued_at, expires_at, trusted_utc_now()?)?;
        }
        Ok(())
    }

    fn validate_persisted_native_replay_provenance(
        record: &DurableDispatchRecord,
        authority_namespace: Option<&str>,
        native_authorization_id: Option<&str>,
        derivation_digest: Option<&str>,
    ) -> Result<(), AuthorizationStoreError> {
        match (authority_namespace, native_authorization_id, derivation_digest) {
            (None, None, None) => Ok(()),
            (Some(namespace), Some(native_id), Some(digest)) => {
                let derived = super::NativeReplayDerivation::derive(namespace, native_id)
                    .map_err(|_| AuthorizationConsumptionError::InvalidNativeReplayProvenance)?;
                if derived.native_replay_identity != record.native_replay_identity
                    || derived.derivation_digest != digest
                {
                    return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
                }
                Ok(())
            }
            _ => Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into()),
        }
    }

    /// Commit an outcome only through the frozen dispatch record. The record
    /// is revalidated immediately before the lease transition so the provider
    /// outcome cannot be attached to a different sink contract.
    fn validate_verified_terminal_outcome(
        &self,
        record: &DurableDispatchRecord,
        verified: &VerifiedProviderOutcome,
    ) -> Result<(), AuthorizationStoreError> {
        let evidence = &verified.evidence;
        if !matches!(evidence.kind, ProviderEvidenceKind::TerminalOutcome)
            || matches!(evidence.outcome, ExecutionOutcome::Indeterminate)
        {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }
        if verified.configuration.relying_party_id != self.relying_party_id
            || verified.configuration.verifier_id.is_empty()
            || verified.configuration.verifier_config_digest.is_empty()
            || verified.configuration.trust_anchor_digest.is_empty()
            || verified.configuration.evidence_profile_digest.is_empty()
            || verified.verification_digest.is_empty()
            || evidence.evidence_id.is_empty() || evidence.evidence_digest.is_empty()
            || evidence.action_id != record.action_id
            || evidence.action_digest != record.action_digest
            || evidence.attempt_id != record.attempt_id
            || evidence.operation_id != record.operation_id
            || evidence.native_replay_identity != record.native_replay_identity
            || evidence.provider_idempotency_key != record.provider_idempotency_key
            || evidence.target_identity != record.target_identity
            || evidence.audience != record.audience
            || evidence.adapter != record.adapter
            || evidence.boundary_id != record.boundary_id
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Ok(())
    }

    /// Commit a terminal provider outcome only after a relying-party configured
    /// verifier has authenticated and semantically bound the provider evidence
    /// to this exact frozen dispatch record.
    pub fn commit_bound_verified<V: ProviderEvidenceVerifier>(
        &self,
        record: &DurableDispatchRecord,
        evidence: &ProviderTerminalEvidence,
        verifier: &V,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let verified = verifier
            .verify(ProviderVerificationPurpose::TerminalOutcome, record, evidence)
            .map_err(|_| AuthorizationConsumptionError::ProviderEvidenceVerificationRequired)?;
        self.validate_verified_terminal_outcome(record, &verified)?;

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row: (
            String, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>,
            Option<String>, Option<String>, Option<String>
        ) = tx.query_row(
            "SELECT state,native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
                    native_authority_pin_set_id,native_authority_pin_set_digest,
                    validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
            params![record.authorization_instance, record.attempt_id, record.boundary_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?)),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;
        let persisted_relying_party: Option<String> = tx.query_row(
            "SELECT relying_party_id FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
            params![record.authorization_instance, record.attempt_id, record.boundary_id],
            |r| r.get(0),
        )?;
        if persisted_relying_party.as_deref() != Some(self.relying_party_id.as_str()) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Self::validate_persisted_native_replay_provenance(
            record, row.1.as_deref(), row.2.as_deref(), row.3.as_deref()
        )?;
        self.validate_native_authority_pin_set_snapshot(
            &tx, row.4.as_deref(), row.5.as_deref()
        )?;
        if !matches!(row.0.as_str(), "dispatch_pending" | "invoked" | "indeterminate") {
            if let Some(receipt) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")? {
                return Ok(receipt);
            }
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }

        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if lease.action_id != record.action_id
            || lease.action_digest != record.action_digest
            || lease
                .provider_idempotency_key_for_native_replay(
                    &record.native_replay_identity,
                    &record.target_identity,
                )
                .map_err(|_| AuthorizationConsumptionError::InvalidBinding)?
                != record.provider_idempotency_key
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if let Some(receipt) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")? {
            return Ok(receipt);
        }
        if let Some(receipt) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "indeterminate")? {
            return Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into());
        }

        let mut receipt = lease.commit(&record.attempt_id, evidence.outcome)?;
        receipt.provider_idempotency_key = record.provider_idempotency_key.clone();
        update_lease_with_boundary(&tx, &lease, Some(&record.boundary_id))?;
        tx.execute(
            "INSERT INTO authorization_terminal_evidence
             (authorization_instance,attempt_id,operation_id,native_replay_identity,
              native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
              native_authority_pin_set_id,native_authority_pin_set_digest,
              validity_issued_at,validity_expires_at,validity_policy_digest,relying_party_id,boundary_id,
              action_digest,provider_idempotency_key,target_identity,audience,adapter,outcome,evidence_id,
              evidence_digest,verifier_id,verifier_config_digest,trust_anchor_digest,evidence_profile_digest,verification_digest)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27)",
            params![
                record.authorization_instance, record.attempt_id, record.operation_id,
                record.native_replay_identity, row.1, row.2, row.3, row.4, row.5,
                row.6, row.7, row.8,
                self.relying_party_id.as_str(), record.boundary_id, record.action_digest,
                record.provider_idempotency_key, record.target_identity, record.audience, record.adapter,
                if matches!(evidence.outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                evidence.evidence_id, evidence.evidence_digest, verified.configuration.verifier_id,
                verified.configuration.verifier_config_digest, verified.configuration.trust_anchor_digest,
                verified.configuration.evidence_profile_digest, verified.verification_digest,
            ],
        )?;
        insert_receipt_with_boundary(&tx, &receipt, "final", Some(&record.boundary_id))?;
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?4
               AND state IN ('dispatch_pending','invoked','indeterminate')",
            params![record.authorization_instance, record.attempt_id,
                if matches!(evidence.outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                record.boundary_id],
        )?;
        tx.commit()?;
        Ok(receipt)
    }

    pub fn commit_bound(
        &self, record: &DurableDispatchRecord, outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx.query_row(
            "SELECT operation_id,native_replay_identity,relying_party_id,action_id,action_digest,provider_idempotency_key,
                    target_identity,audience,adapter,boundary_id,state
             FROM authorization_dispatches WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((
                r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?,
                r.get::<_,String>(3)?, r.get::<_,String>(4)?, r.get::<_,String>(5)?,
                r.get::<_,String>(6)?, r.get::<_,String>(7)?, r.get::<_,String>(8)?,
                r.get::<_,String>(9)?, r.get::<_,String>(10)?,
            )),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;
        if row.0 != record.operation_id || row.1 != record.native_replay_identity
            || row.2 != self.relying_party_id
            || row.3 != record.action_id || row.4 != record.action_digest
            || row.5 != record.provider_idempotency_key || row.6 != record.target_identity
            || row.7 != record.audience || row.8 != record.adapter || row.9 != record.boundary_id
            || !matches!(row.10.as_str(), "dispatch_pending" | "invoked" | "indeterminate" | "succeeded" | "failed")
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if !matches!(outcome, ExecutionOutcome::Indeterminate) {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }
        if let Some(r) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")? {
            return Ok(r);
        }
        if let Some(r) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "indeterminate")? {
            return if matches!(outcome, ExecutionOutcome::Indeterminate) { Ok(r) }
            else { Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into()) };
        }
        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if lease.action_id != record.action_id || lease.action_digest != record.action_digest
            || lease
                .provider_idempotency_key_for_native_replay(
                    &record.native_replay_identity,
                    &record.target_identity,
                )
                .map_err(|_| AuthorizationConsumptionError::InvalidBinding)?
                != record.provider_idempotency_key
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let mut receipt = lease.commit(&record.attempt_id, outcome)?;
        receipt.provider_idempotency_key = record.provider_idempotency_key.clone();
        update_lease_with_boundary(&tx, &lease, Some(&record.boundary_id))?;
        let phase = if matches!(outcome, ExecutionOutcome::Indeterminate) { "indeterminate" } else { "final" };
        insert_receipt_with_boundary(&tx, &receipt, phase, Some(&record.boundary_id))?;
        let dispatch_state = match outcome {
            ExecutionOutcome::Succeeded => "succeeded",
            ExecutionOutcome::Failed => "failed",
            ExecutionOutcome::Indeterminate => "indeterminate",
        };
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id, dispatch_state],
        )?;
        tx.commit()?;
        Ok(receipt)
    }

    pub fn commit(
        &self, authorization_instance: &str, attempt_id: &str, outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        if dispatch_boundary(&tx, authorization_instance, attempt_id)?.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if let Some(r) = load_receipt(&tx, authorization_instance, attempt_id, "final")? { return Ok(r); }
        if let Some(r) = load_receipt(&tx, authorization_instance, attempt_id, "indeterminate")? {
            return if matches!(outcome, ExecutionOutcome::Indeterminate) {
                Ok(r)
            } else {
                Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into())
            };
        }

        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        let receipt = lease.commit(attempt_id, outcome)?;
        update_lease(&tx, &lease)?;
        let phase = if matches!(outcome, ExecutionOutcome::Indeterminate) { "indeterminate" } else { "final" };
        insert_receipt(&tx, &receipt, phase)?;
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![authorization_instance, attempt_id, if matches!(outcome, ExecutionOutcome::Indeterminate) { "indeterminate" } else if matches!(outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" }],
        )?;
        tx.commit()?;
        Ok(receipt)
    }

    pub fn reconcile_indeterminate(
        &self, authorization_instance: &str, attempt_id: &str, outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        if matches!(outcome, ExecutionOutcome::Indeterminate) {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if dispatch_boundary(&tx, authorization_instance, attempt_id)?.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if let Some(r) = load_receipt(&tx, authorization_instance, attempt_id, "reconciled")? { return Ok(r); }

        let mut lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        let receipt = lease.reconcile_indeterminate(attempt_id, outcome)?;
        let changed = tx.execute(
            "UPDATE authorization_leases SET state=?2, attempt_id=?3, remaining_executions=?4
             WHERE authorization_instance=?1 AND state='indeterminate' AND attempt_id=?5",
            params![
                authorization_instance, encode_state(&lease.state), state_attempt(&lease.state),
                lease.remaining_executions as i64, attempt_id
            ],
        )?;
        if changed != 1 { return Err(AuthorizationConsumptionError::AttemptMismatch.into()); }
        tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2 AND state='indeterminate'",
            params![authorization_instance, attempt_id,
                if matches!(outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" }],
        )?;
        insert_receipt(&tx, &receipt, "reconciled")?;
        tx.commit()?;
        Ok(receipt)
    }

    /// Recover one exact Prepared attempt that is proven to have stopped
    /// before DispatchPending.
    ///
    /// This is not outcome reconciliation. It atomically marks the attempt as
    /// not entered, releases its reservation back to Ready, and records the
    /// explicit not-entered marker. A later dispatch using the old attempt ID
    /// therefore fails rather than continuing after recovery.
    pub fn recover_pre_dispatch_attempt(
        &self,
        witness: &RecoveryAuthorizationWitness,
    ) -> Result<bool, AuthorizationStoreError> {
        if witness.boundary_id.is_empty() || witness.attempt_id.is_empty() {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;

        if !witness.is_bound_to(&lease) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if let Some((boundary_id, action_digest)) = tx
            .query_row(
                "SELECT boundary_id,action_digest
                 FROM authorization_recovery_markers
                 WHERE authorization_instance=?1 AND attempt_id=?2 AND marker='not_entered'",
                params![witness.authorization_instance.as_str(), witness.attempt_id.as_str()],
                |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?)),
            )
            .optional()?
        {
            if boundary_id == witness.boundary_id && action_digest == witness.action_digest {
                return Ok(false);
            }
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if !matches!(
            &lease.state,
            AuthorizationLeaseState::Prepared { attempt_id } if attempt_id == &witness.attempt_id
        ) {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        if dispatch_boundary(&tx, &witness.authorization_instance, &witness.attempt_id)?.is_some() {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        let changed = tx.execute(
            "UPDATE authorization_leases
             SET state='ready', attempt_id=NULL, boundary_id=NULL
             WHERE authorization_instance=?1 AND state='prepared'
               AND attempt_id=?2 AND boundary_id=?3",
            params![
                witness.authorization_instance.as_str(),
                witness.attempt_id.as_str(),
                witness.boundary_id.as_str()
            ],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        tx.execute(
            "INSERT INTO authorization_recovery_markers
             (authorization_instance,attempt_id,boundary_id,action_digest,authority_epoch,marker)
             VALUES (?1,?2,?3,?4,?5,'not_entered')",
            params![
                witness.authorization_instance.as_str(),
                witness.attempt_id.as_str(),
                witness.boundary_id.as_str(),
                witness.action_digest.as_str(),
                witness.authority_epoch as i64
            ],
        )?;

        tx.commit()?;
        Ok(true)
    }

    /// Reconcile an indeterminate effect using its exact frozen dispatch record.
    ///
    /// The durable dispatch record and current lease must name the same boundary
    /// before this operation can consume the authorization budget.
    fn reconcile_indeterminate_bound_verified_inner(
        &self,
        record: &DurableDispatchRecord,
        outcome: ExecutionOutcome,
        verified: &VerifiedProviderOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        if matches!(outcome, ExecutionOutcome::Indeterminate) {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let row = tx.query_row(
            "SELECT operation_id,native_replay_identity,relying_party_id,action_id,action_digest,
                    provider_idempotency_key,target_identity,audience,adapter,boundary_id,state
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance.as_str(), record.attempt_id.as_str()],
            |r| Ok((
                r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,Option<String>>(2)?,
                r.get::<_,String>(3)?, r.get::<_,String>(4)?, r.get::<_,String>(5)?,
                r.get::<_,String>(6)?, r.get::<_,String>(7)?, r.get::<_,String>(8)?,
                r.get::<_,String>(9)?, r.get::<_,String>(10)?,
            )),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;

        if row.0 != record.operation_id || row.1 != record.native_replay_identity
            || row.2.as_deref() != Some(self.relying_party_id.as_str())
            || row.3 != record.action_id || row.4 != record.action_digest
            || row.5 != record.provider_idempotency_key || row.6 != record.target_identity
            || row.7 != record.audience || row.8 != record.adapter
            || row.9 != record.boundary_id || row.10 != "indeterminate"
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let native_provenance: (
            Option<String>, Option<String>, Option<String>, Option<String>, Option<String>,
            Option<String>, Option<String>, Option<String>
        ) = tx.query_row(
            "SELECT native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
                    native_authority_pin_set_id,native_authority_pin_set_digest,
                    validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?)),
        )?;
        Self::validate_persisted_native_replay_provenance(
            record,
            native_provenance.0.as_deref(),
            native_provenance.1.as_deref(),
            native_provenance.2.as_deref(),
        )?;
        self.validate_native_authority_pin_set_snapshot(
            &tx,
            native_provenance.3.as_deref(),
            native_provenance.4.as_deref(),
        )?;
        self.validate_persisted_authorization_validity(
            native_provenance.5.as_deref(),
            native_provenance.6.as_deref(),
            native_provenance.7.as_deref(),
            false,
        )?;

        if let Some(r) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "reconciled")? {
            return Ok(r);
        }

        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if lease.action_id != record.action_id || lease.action_digest != record.action_digest
            || lease
                .provider_idempotency_key_for_native_replay(
                    &record.native_replay_identity,
                    &record.target_identity,
                )
                .map_err(|_| AuthorizationConsumptionError::InvalidBinding)?
                != record.provider_idempotency_key
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut receipt = lease.reconcile_indeterminate(&record.attempt_id, outcome)?;
        receipt.provider_idempotency_key = record.provider_idempotency_key.clone();
        update_lease_with_boundary(&tx, &lease, Some(&record.boundary_id))?;

        let dispatch_state = if matches!(outcome, ExecutionOutcome::Succeeded) {
            "succeeded"
        } else {
            "failed"
        };
        let changed = tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2
               AND boundary_id=?4 AND state='indeterminate'",
            params![record.authorization_instance, record.attempt_id, dispatch_state, record.boundary_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
        insert_receipt_with_boundary(&tx, &receipt, "reconciled", Some(&record.boundary_id))?;
        tx.execute(
            "INSERT INTO authorization_terminal_evidence
             (authorization_instance,attempt_id,operation_id,native_replay_identity,
              native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
              native_authority_pin_set_id,native_authority_pin_set_digest,relying_party_id,boundary_id,
              action_digest,provider_idempotency_key,target_identity,audience,adapter,outcome,evidence_id,
              evidence_digest,verifier_id,verifier_config_digest,trust_anchor_digest,
              evidence_profile_digest,verification_digest)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27)",
            params![
                record.authorization_instance, record.attempt_id, record.operation_id,
                record.native_replay_identity, native_provenance.0, native_provenance.1, native_provenance.2,
                native_provenance.3, native_provenance.4,
                native_provenance.5, native_provenance.6, native_provenance.7,
                self.relying_party_id.as_str(), record.boundary_id, record.action_digest,
                record.provider_idempotency_key, record.target_identity, record.audience, verified.evidence.adapter,
                if matches!(outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                verified.evidence.evidence_id, verified.evidence.evidence_digest,
                verified.configuration.verifier_id, verified.configuration.verifier_config_digest,
                verified.configuration.trust_anchor_digest,
                verified.configuration.evidence_profile_digest, verified.verification_digest,
            ],
        )?;
        tx.commit()?;
        Ok(receipt)
    }

    /// Terminal reconciliation requires authenticated provider evidence. The legacy
    /// outcome-only API is deliberately fenced so a local enum cannot masquerade
    /// as provider truth.
    pub fn reconcile_indeterminate_bound(
        &self,
        _record: &DurableDispatchRecord,
        _outcome: ExecutionOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into())
    }

    pub fn reconcile_indeterminate_bound_verified<V: ProviderEvidenceVerifier>(
        &self,
        record: &DurableDispatchRecord,
        evidence: &ProviderTerminalEvidence,
        verifier: &V,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        let verified = verifier
            .verify(ProviderVerificationPurpose::TerminalOutcome, record, evidence)
            .map_err(|_| AuthorizationConsumptionError::ProviderEvidenceVerificationRequired)?;
        Self::validate_verified_terminal_outcome(record, &verified)?;
        if !matches!(verified.evidence.outcome, ExecutionOutcome::Succeeded | ExecutionOutcome::Failed) {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }
        self.reconcile_indeterminate_bound_verified_inner(record, verified.evidence.outcome, &verified)
    }

    /// Crash recovery is deliberately conservative: a process may have reached
    /// the external sink after its last durable local write. Any non-terminal
    /// prepared/dispatch-pending reservation therefore becomes Indeterminate
    /// before another execution can be admitted.
    /// Recover one exact attempt owned by one execution boundary.
    ///
    /// This is the preferred recovery primitive for an external effect: the
    /// boundary and attempt are both supplied explicitly, so recovery cannot
    /// claim a sibling attempt merely because it shares the same boundary.
    pub fn recover_incomplete_attempt_for_boundary(
        &self,
        boundary_id: &str,
        attempt_id: &str,
    ) -> Result<usize, AuthorizationStoreError> {
        if boundary_id.is_empty() || attempt_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        self.recover_incomplete_attempts_scoped(Some(boundary_id), Some(attempt_id))
    }

    /// Recover all incomplete attempts owned by one boundary.
    ///
    /// This batch form is intended for executor startup maintenance. Any
    /// per-attempt recovery authorization should use the exact-attempt API.
    pub fn recover_incomplete_attempts_for_boundary(
        &self,
        boundary_id: &str,
    ) -> Result<usize, AuthorizationStoreError> {
        if boundary_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        self.recover_incomplete_attempts_scoped(Some(boundary_id), None)
    }

    /// Recover only legacy/unscoped attempts. Bound effectful attempts should use
    /// one of the boundary-scoped recovery paths.
    pub fn recover_incomplete_attempts(&self) -> Result<usize, AuthorizationStoreError> {
        self.recover_incomplete_attempts_scoped(None, None)
    }

    fn recover_incomplete_attempts_scoped(
        &self,
        boundary_filter: Option<&str>,
        attempt_filter: Option<&str>,
    ) -> Result<usize, AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut recovered = Vec::new();
        {
            let (mut stmt, query_params) = match (boundary_filter, attempt_filter) {
                (Some(_), Some(_)) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id=?1 AND attempt_id=?2",
                    )?,
                    vec![
                        boundary_filter.unwrap().to_owned(),
                        attempt_filter.unwrap().to_owned(),
                    ],
                ),
                (Some(_), None) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('prepared','dispatch_pending','invoked')
                           AND boundary_id=?1",
                    )?,
                    vec![boundary_filter.unwrap().to_owned()],
                ),
                (None, Some(_)) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id IS NULL AND attempt_id=?1",
                    )?,
                    vec![attempt_filter.unwrap().to_owned()],
                ),
                (None, None) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id IS NULL",
                    )?,
                    Vec::new(),
                ),
            };
            let rows = stmt.query_map(rusqlite::params_from_iter(query_params.iter()), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)? as u64,
                    row.get::<_, Option<String>>(5)?,
                ))
            })?;
            for row in rows {
                recovered.push(row?);
            }
        }

        for (instance, action_id, attempt_id, action_digest, authority_epoch, lease_boundary) in &recovered {
            if let Some(boundary_id) = lease_boundary.as_deref() {
                match dispatch_boundary(&tx, instance, attempt_id)? {
                    Some(dispatch_boundary_id) if dispatch_boundary_id == boundary_id => {}
                    Some(_) => return Err(AuthorizationConsumptionError::InvalidBinding.into()),
                    None => return Err(AuthorizationConsumptionError::InvalidBinding.into()),
                }
            }
            tx.execute(
                "UPDATE authorization_leases
                 SET state='indeterminate', attempt_id=?2
                 WHERE authorization_instance=?1
                   AND state IN ('prepared','dispatch_pending','invoked')
                   AND attempt_id=?2",
                params![instance, attempt_id],
            )?;
            let recovered_provider_key: String = if lease_boundary.is_some() {
                tx.query_row(
                    "SELECT provider_idempotency_key
                     FROM authorization_dispatches
                     WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
                    params![instance, attempt_id, lease_boundary.as_deref()],
                    |row| row.get(0),
                )?
            } else {
                let lease = AuthorizationLease::new_with_instance(
                    instance.clone(),
                    action_id.clone(),
                    action_digest.clone(),
                    String::new(),
                    String::new(),
                    *authority_epoch,
                    1,
                );
                lease.provider_idempotency_key()
            };
            let receipt = ExecutionReceipt {
                action_id: action_id.clone(),
                authorization_instance: instance.clone(),
                action_digest: action_digest.clone(),
                provider_idempotency_key: recovered_provider_key,
                attempt_id: attempt_id.clone(),
                authority_epoch: *authority_epoch,
                outcome: ExecutionOutcome::Indeterminate,
            };
            tx.execute(
                "UPDATE authorization_dispatches SET state='indeterminate'
                 WHERE authorization_instance=?1 AND attempt_id=?2
                   AND state IN ('dispatch_pending','invoked')
                   AND (?3 IS NULL OR boundary_id=?3)",
                params![instance, attempt_id, lease_boundary],
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO authorization_receipts
                 (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,boundary_id)
                 VALUES (?1,?2,?3,'indeterminate','indeterminate',?4,?5,?6)",
                params![
                    receipt.authorization_instance,
                    receipt.action_id,
                    receipt.attempt_id,
                    receipt.action_digest,
                    receipt.authority_epoch as i64,
                    lease_boundary
                ],
            )?;
        }
        tx.commit()?;
        Ok(recovered.len())
    }
}

fn migrate_legacy_schema(connection: &mut Connection) -> Result<(), AuthorizationStoreError> {
    let has_lease_table: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='authorization_leases'
            )",
            [],
            |row| row.get(0),
        )?;

    if !has_lease_table {
        return Ok(());
    }

    let has_instance: bool = connection
        .prepare("PRAGMA table_info(authorization_leases)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|name| name == "authorization_instance");

    if has_instance {
        return Ok(());
    }

    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute_batch(
        "ALTER TABLE authorization_leases RENAME TO authorization_leases_legacy;
         ALTER TABLE authorization_receipts RENAME TO authorization_receipts_legacy;

         CREATE TABLE authorization_leases (
           authorization_instance TEXT PRIMARY KEY,
           action_id TEXT NOT NULL,
           action_digest TEXT NOT NULL,
           support_digest TEXT NOT NULL,
           policy TEXT NOT NULL,
           authority_epoch INTEGER NOT NULL,
           remaining_executions INTEGER NOT NULL,
           state TEXT NOT NULL,
           attempt_id TEXT,
           boundary_id TEXT
         );

         CREATE TABLE authorization_receipts (
           authorization_instance TEXT NOT NULL,
           action_id TEXT NOT NULL,
           attempt_id TEXT NOT NULL,
           phase TEXT NOT NULL,
           outcome TEXT NOT NULL,
           action_digest TEXT NOT NULL,
           authority_epoch INTEGER NOT NULL,
           boundary_id TEXT,
           PRIMARY KEY(authorization_instance, attempt_id, phase)
         );",
    )?;
    tx.execute(
        "INSERT INTO authorization_leases
         (authorization_instance,action_id,action_digest,support_digest,policy,
          authority_epoch,remaining_executions,state,attempt_id,boundary_id)
         SELECT action_id,action_id,action_digest,support_digest,policy,
                authority_epoch,remaining_executions,state,attempt_id,NULL
         FROM authorization_leases_legacy",
        [],
    )?;
    tx.execute(
        "INSERT INTO authorization_receipts
         (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,boundary_id)
         SELECT action_id,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,NULL
         FROM authorization_receipts_legacy",
        [],
    )?;
    tx.execute_batch(
        "DROP TABLE authorization_leases_legacy;
         DROP TABLE authorization_receipts_legacy;",
    )?;
    tx.commit()?;
    Ok(())
}

fn encode_state(s: &AuthorizationLeaseState) -> &'static str {
    match s {
        AuthorizationLeaseState::Ready => "ready",
        AuthorizationLeaseState::Prepared { .. } => "prepared",
        AuthorizationLeaseState::DispatchPending { .. } => "dispatch_pending",
        AuthorizationLeaseState::Invoked { .. } => "invoked",
        AuthorizationLeaseState::Indeterminate { .. } => "indeterminate",
        AuthorizationLeaseState::Exhausted => "exhausted",
        AuthorizationLeaseState::Revoked => "revoked",
        AuthorizationLeaseState::Expired => "expired",
    }
}
fn state_attempt(s: &AuthorizationLeaseState) -> Option<&str> {
    match s {
        AuthorizationLeaseState::Prepared { attempt_id }
        | AuthorizationLeaseState::DispatchPending { attempt_id }
        | AuthorizationLeaseState::Invoked { attempt_id }
        | AuthorizationLeaseState::Indeterminate { attempt_id } => Some(attempt_id),
        _ => None,
    }
}
fn load_lease(tx: &Transaction<'_>, id: &str) -> Result<Option<AuthorizationLease>, rusqlite::Error> {
    tx.query_row(
        "SELECT authorization_instance,action_id,action_digest,support_digest,policy,authority_epoch,
                remaining_executions,state,attempt_id FROM authorization_leases
         WHERE authorization_instance=?1",
        params![id],
        |r| {
            let state: String = r.get(7)?;
            let attempt: Option<String> = r.get(8)?;
            let decoded = match state.as_str() {
                "ready" => AuthorizationLeaseState::Ready,
                "prepared" => AuthorizationLeaseState::Prepared {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "dispatch_pending" => AuthorizationLeaseState::DispatchPending {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "invoked" => AuthorizationLeaseState::Invoked {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "indeterminate" => AuthorizationLeaseState::Indeterminate {
                    attempt_id: attempt.ok_or_else(|| rusqlite::Error::InvalidQuery)?,
                },
                "exhausted" => AuthorizationLeaseState::Exhausted,
                "revoked" => AuthorizationLeaseState::Revoked,
                "expired" => AuthorizationLeaseState::Expired,
                _ => return Err(rusqlite::Error::InvalidQuery),
            };
            Ok(AuthorizationLease {
                authorization_instance: r.get(0)?,
                action_id: r.get(1)?,
                action_digest: r.get(2)?,
                support_digest: r.get(3)?,
                policy: r.get(4)?,
                authority_epoch: r.get::<_,i64>(5)? as u64,
                remaining_executions: r.get::<_,i64>(6)? as u32,
                state: decoded,
            })
        },
    ).optional()
}
fn ensure_column(
    connection: &mut Connection,
    table: &str,
    column: &str,
    declaration: &str,
) -> Result<(), AuthorizationStoreError> {
    let names = connection
        .prepare(&format!("PRAGMA table_info({table})"))?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if names.iter().any(|name| name == column) {
        return Ok(());
    }
    connection.execute(
        &format!("ALTER TABLE {table} ADD COLUMN {column} {declaration}"),
        [],
    )?;
    Ok(())
}

fn dispatch_boundary(
    tx: &Transaction<'_>,
    authorization_instance: &str,
    attempt_id: &str,
) -> Result<Option<String>, AuthorizationStoreError> {
    tx.query_row(
        "SELECT boundary_id FROM authorization_dispatches
         WHERE authorization_instance=?1 AND attempt_id=?2",
        params![authorization_instance, attempt_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

fn load_lease_boundary(
    tx: &Transaction<'_>,
    authorization_instance: &str,
) -> Result<Option<String>, AuthorizationStoreError> {
    tx.query_row(
        "SELECT boundary_id FROM authorization_leases
         WHERE authorization_instance=?1",
        params![authorization_instance],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

fn update_lease(tx: &Transaction<'_>, lease: &AuthorizationLease) -> Result<(), AuthorizationStoreError> {
    update_lease_with_boundary(tx, lease, None)
}

fn update_lease_with_boundary(
    tx: &Transaction<'_>,
    lease: &AuthorizationLease,
    boundary_id: Option<&str>,
) -> Result<(), AuthorizationStoreError> {
    let current_boundary = match lease.state {
        AuthorizationLeaseState::Ready
        | AuthorizationLeaseState::Exhausted
        | AuthorizationLeaseState::Revoked
        | AuthorizationLeaseState::Expired => None,
        _ => boundary_id,
    };
    let changed = tx.execute(
        "UPDATE authorization_leases
         SET state=?2,attempt_id=?3,remaining_executions=?4,boundary_id=?5
         WHERE authorization_instance=?1",
        params![
            lease.authorization_instance,
            encode_state(&lease.state),
            state_attempt(&lease.state),
            lease.remaining_executions as i64,
            current_boundary
        ],
    )?;
    if changed != 1 {
        return Err(AuthorizationStoreError::NotFound(lease.authorization_instance.clone()));
    }
    Ok(())
}

fn insert_receipt(tx: &Transaction<'_>, r: &ExecutionReceipt, phase: &str) -> Result<(), AuthorizationStoreError> {
    insert_receipt_with_boundary(tx, r, phase, None)
}

fn insert_receipt_with_boundary(
    tx: &Transaction<'_>,
    r: &ExecutionReceipt,
    phase: &str,
    boundary_id: Option<&str>,
) -> Result<(), AuthorizationStoreError> {
    let outcome = match r.outcome {
        ExecutionOutcome::Succeeded => "succeeded",
        ExecutionOutcome::Failed => "failed",
        ExecutionOutcome::Indeterminate => "indeterminate",
    };
    tx.execute(
        "INSERT INTO authorization_receipts
         (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,boundary_id)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
        params![
            r.authorization_instance, r.action_id, r.attempt_id, phase, outcome,
            r.action_digest, r.authority_epoch as i64, boundary_id
        ],
    )?;
    Ok(())
}
fn load_receipt(
    tx: &Transaction<'_>, authorization_instance: &str, attempt_id: &str, phase: &str,
) -> Result<Option<ExecutionReceipt>, AuthorizationStoreError> {
    tx.query_row(
        "SELECT authorization_instance,action_id,attempt_id,outcome,action_digest,authority_epoch
         FROM authorization_receipts
         WHERE authorization_instance=?1 AND attempt_id=?2 AND phase=?3",
        params![authorization_instance,attempt_id,phase],
        |r| {
            let outcome = match r.get::<_,String>(3)?.as_str() {
                "succeeded" => ExecutionOutcome::Succeeded,
                "failed" => ExecutionOutcome::Failed,
                "indeterminate" => ExecutionOutcome::Indeterminate,
                _ => return Err(rusqlite::Error::InvalidQuery),
            };
            let authorization_instance: String = r.get(0)?;
            let action_id: String = r.get(1)?;
            let attempt_id: String = r.get(2)?;
            let action_digest: String = r.get(4)?;
            let authority_epoch = r.get::<_,i64>(5)? as u64;
            let lease = AuthorizationLease::new_with_instance(
                authorization_instance.clone(),
                action_id.clone(),
                action_digest.clone(),
                String::new(),
                String::new(),
                authority_epoch,
                1,
            );
            Ok(ExecutionReceipt {
                authorization_instance,
                action_id,
                attempt_id,
                outcome,
                action_digest,
                provider_idempotency_key: lease.provider_idempotency_key(),
                authority_epoch,
            })
        },
    ).optional().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    fn fixture(path: &Path) -> (SqliteAuthorizationStore, EpistemicAction, ActionAuthorizationWitness) {
        let store = SqliteAuthorizationStore::open(path).unwrap();
        let action = EpistemicAction::new("durable-action","intervention",super::super::ActionRisk::Critical);
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:action.id.clone(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:00:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new(action.id.clone(),digest,"sha256:support","policy-v1",1,1)).unwrap();
        (store,action,witness)
    }

    struct TestProviderVerifier;

    impl ProviderEvidenceVerifier for TestProviderVerifier {
        fn verify(
            &self,
            purpose: ProviderVerificationPurpose,
            record: &DurableDispatchRecord,
            evidence: &ProviderTerminalEvidence,
        ) -> Result<VerifiedProviderOutcome, ProviderVerificationError> {
            if !matches!(purpose, ProviderVerificationPurpose::TerminalOutcome)
                || !matches!(evidence.kind, ProviderEvidenceKind::TerminalOutcome)
                || evidence.action_id != record.action_id
                || evidence.action_digest != record.action_digest
                || evidence.attempt_id != record.attempt_id
                || evidence.operation_id != record.operation_id
                || evidence.native_replay_identity != record.native_replay_identity
                || evidence.provider_idempotency_key != record.provider_idempotency_key
                || evidence.target_identity != record.target_identity
                || evidence.audience != record.audience
                || evidence.adapter != record.adapter
                || evidence.boundary_id != record.boundary_id
                || matches!(evidence.outcome, ExecutionOutcome::Indeterminate)
            {
                return Err(ProviderVerificationError::VerificationFailed);
            }
            Ok(VerifiedProviderOutcome {
                evidence: evidence.clone(),
                configuration: ProviderVerifierConfiguration {
                    relying_party_id: "legacy-local".into(),
                    verifier_id: "test-verifier/v1".into(),
                    verifier_config_digest: "sha256:test-verifier-config".into(),
                    trust_anchor_digest: "sha256:test-trust-anchors".into(),
                    evidence_profile_digest: "sha256:test-evidence-profile".into(),
                },
                verification_digest: "sha256:test-verification".into(),
            })
        }
    }

    fn verified_evidence(record: &DurableDispatchRecord, outcome: ExecutionOutcome) -> ProviderTerminalEvidence {
        ProviderTerminalEvidence {
            kind: ProviderEvidenceKind::TerminalOutcome,
            outcome,
            evidence_id: format!("provider-evidence:{}", record.attempt_id),
            evidence_digest: "sha256:provider-evidence".into(),
            action_id: record.action_id.clone(),
            action_digest: record.action_digest.clone(),
            attempt_id: record.attempt_id.clone(),
            operation_id: record.operation_id.clone(),
            native_replay_identity: record.native_replay_identity.clone(),
            provider_idempotency_key: record.provider_idempotency_key.clone(),
            target_identity: record.target_identity.clone(),
            audience: record.audience.clone(),
            adapter: record.adapter.clone(),
            boundary_id: record.boundary_id.clone(),
        }
    }

    fn mark_dispatch_pending_bound_for_test(
        store: &SqliteAuthorizationStore,
        authorization_instance: &str,
        attempt_id: &str,
        action: &EpistemicAction,
        expected_effect: &super::super::ActionEffectBinding,
        boundary_id: &str,
        operation_id: impl AsRef<str>,
        native_authorization_id: impl AsRef<str>,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        const ISSUER: &str = "test-issuer";
        const NAMESPACE: &str = "test-authority/v1";
        store.pin_native_authority_namespace(ISSUER, NAMESPACE)?;
        store.mark_dispatch_pending_bound_from_pinned_native_authority(
            authorization_instance,
            attempt_id,
            action,
            expected_effect,
            boundary_id,
            operation_id.as_ref(),
            ISSUER,
            native_authorization_id.as_ref(),
        )
    }

    fn mark_dispatch_pending_bound_from_native_authority_for_test(
        store: &SqliteAuthorizationStore,
        authorization_instance: &str,
        attempt_id: &str,
        action: &EpistemicAction,
        expected_effect: &super::super::ActionEffectBinding,
        boundary_id: &str,
        operation_id: &str,
        authority_namespace: &str,
        native_authorization_id: &str,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        const ISSUER: &str = "test-explicit-issuer";
        store.pin_native_authority_namespace(ISSUER, authority_namespace)?;
        store.mark_dispatch_pending_bound_from_pinned_native_authority(
            authorization_instance,
            attempt_id,
            action,
            expected_effect,
            boundary_id,
            operation_id,
            ISSUER,
            native_authorization_id,
        )
    }

    #[test]
    fn unbound_prepare_rejects_effect_bound_actions() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-effect-prepare-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new(
            "effect-bound-prepare","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"effect-bound-prepare".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T06:50:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        assert!(matches!(
            store.prepare_for_execution(&witness,&action,"frame@1","attempt-effect-bound"),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_evidence_storage_is_insert_only_for_attempt() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-insert-only-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new(
            "target-terminal-insert-only","prod","adapter-terminal"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"terminal-insert-only".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T06:35:00Z".into(),
            expires_at:Some("2026-10-04T06:35:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-terminal-insert-only","boundary-terminal"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-terminal-insert-only",
            &action,&effect,"boundary-terminal",
            "operation:terminal-insert-only","native-terminal-insert-only"
        ).unwrap();
        store.commit_bound_verified(
            &record,&verified_evidence(&record,ExecutionOutcome::Succeeded),&TestProviderVerifier
        ).unwrap();

        let err=store.connection().unwrap().execute(
            "INSERT INTO authorization_terminal_evidence(
                authorization_instance,attempt_id,operation_id,native_replay_identity,
                action_digest,provider_idempotency_key,target_identity,audience,adapter,
                outcome,evidence_id,evidence_digest,verifier_id,verifier_config_digest,
                trust_anchor_digest,evidence_profile_digest,verification_digest
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
            params![
                record.authorization_instance,record.attempt_id,"tampered-operation",
                record.native_replay_identity,record.action_digest,record.provider_idempotency_key,
                record.target_identity,record.audience,"tampered-adapter","succeeded",
                "tampered-evidence","sha256:tampered","tampered-verifier",
                "sha256:tampered-config","sha256:tampered-anchor","sha256:tampered-profile",
                "sha256:tampered-verification"
            ],
        ).unwrap_err();
        assert!(matches!(err, rusqlite::Error::SqliteFailure(_, _)));
        let adapter:String=store.connection().unwrap().query_row(
            "SELECT adapter FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(adapter,record.adapter);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_evidence_persists_exact_adapter_identity() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-adapter-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new(
            "target-terminal-adapter","prod","adapter-terminal"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"terminal-adapter".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T06:30:00Z".into(),
            expires_at:Some("2026-10-04T06:30:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-terminal-adapter","boundary-terminal"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-terminal-adapter",
            &action,&effect,"boundary-terminal",
            "operation:terminal-adapter","native-terminal-adapter"
        ).unwrap();
        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        store.commit_bound_verified(&record,&evidence,&TestProviderVerifier).unwrap();
        let adapter:String=store.connection().unwrap().query_row(
            "SELECT adapter FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(adapter,record.adapter);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn reopening_rejects_preexisting_normalized_issuer_conflict() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-issuer-reopen-conflict-{}.db",std::process::id()
        ));
        {
            let store=SqliteAuthorizationStore::open_with_relying_party(
                &path,"rp-issuer-reopen-conflict"
            ).unwrap();
            store.connection().unwrap().execute(
                "INSERT INTO authorization_native_authority_pins(issuer,authority_namespace)
                 VALUES ('https://issuer.example/','authority/v1'),
                        ('HTTPS://ISSUER.EXAMPLE:443','authority/v2')",
                [],
            ).unwrap();
        }
        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party(
                &path,"rp-issuer-reopen-conflict"
            ),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn normalized_issuer_alias_resolves_to_one_pinned_namespace() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-issuer-alias-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-issuer-alias"
        ).unwrap();
        store.pin_native_authority_namespace(
            "HTTPS://Issuer.Example.:443/","authority/v1"
        ).unwrap();
        assert_eq!(
            store.pinned_native_authority_namespace("https:issuer.example").unwrap(),
            "authority/v1"
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn issuer_normalization_collisions_cannot_split_authority_namespace() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-issuer-normalization-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-issuer-normalization"
        ).unwrap();

        store.pin_native_authority_namespace(
            "HTTPS://Issuer.Example.:443/","authority/v1"
        ).unwrap();
        store.pin_native_authority_namespace(
            "https://issuer.example","authority/v1"
        ).unwrap();

        assert!(matches!(
            store.pin_native_authority_namespace(
                "https://issuer.example/","authority/v2"
            ),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let normalized_a=normalize_native_issuer("HTTPS://Issuer.Example.:443/");
        let normalized_b=normalize_native_issuer("https://issuer.example");
        let normalized_short_url=normalize_native_issuer("https:issuer.example");
        let normalized_explicit_url=normalize_native_issuer("https://issuer.example");
        assert_eq!(normalized_a,normalized_b);
        assert_eq!(normalized_short_url,normalized_explicit_url);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pin_set_snapshot_is_frozen_per_attempt_and_reused_for_terminal_evidence() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-pin-snapshot-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-pin-snapshot").unwrap();
        store.pin_native_authority_namespace("issuer-a","authority/v1").unwrap();

        let effect=super::super::ActionEffectBinding::new("target-pin-snapshot","prod","adapter");
        let action=EpistemicAction::new(
            "pin-snapshot-action","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"pin-snapshot-1".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T09:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            "support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-pin-snapshot","boundary-pin"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,"pin-snapshot-1","attempt-pin-snapshot",&action,&effect,
            "boundary-pin","operation:pin-snapshot","native-pin-snapshot"
        ).unwrap();

        let validity:(String,String,String)=store.connection().unwrap().query_row(
            "SELECT validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))
        ).unwrap();

        let snapshot:(String,String,String)=store.connection().unwrap().query_row(
            "SELECT d.native_authority_pin_set_id,d.native_authority_pin_set_digest,s.snapshot
             FROM authorization_dispatches d
             JOIN authorization_native_authority_pin_sets s
               ON s.pin_set_id=d.native_authority_pin_set_id
              AND s.pin_set_digest=d.native_authority_pin_set_digest
             WHERE d.authorization_instance=?1 AND d.attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))
        ).unwrap();

        store.pin_native_authority_namespace("issuer-b","authority/v1").unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_dispatches
             SET native_authority_pin_set_digest='sha256:forged'
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();
        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        assert!(matches!(
            store.commit_bound_verified(&record,&evidence,&TestProviderVerifier),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            ))
        ));
        store.connection().unwrap().execute(
            "UPDATE authorization_dispatches
             SET native_authority_pin_set_digest=?3
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id,snapshot.1.as_str()],
        ).unwrap();

        let later_digest:String=store.connection().unwrap().query_row(
            "SELECT native_authority_pin_set_digest FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(later_digest,snapshot.1);

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        store.commit_bound_verified(&record,&evidence,&TestProviderVerifier).unwrap();

        let terminal:(String,String,String,String,String)=store.connection().unwrap().query_row(
            "SELECT native_authority_pin_set_id,native_authority_pin_set_digest,
                    validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))
        ).unwrap();
        assert_eq!(terminal.0,snapshot.0);
        assert_eq!(terminal.1,snapshot.1);
        assert_eq!(terminal.2,validity.0);
        assert_eq!(terminal.3,validity.1);
        assert_eq!(terminal.4,validity.2);

        let snapshot_bytes=hex::decode(&snapshot.2).unwrap();
        assert_eq!(
            format!("sha256:{}",hex::encode(Sha256::digest(snapshot_bytes))),
            snapshot.1
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn clock_policy_is_durable_and_cannot_be_changed_on_reopen() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-clock-policy-{}.db",std::process::id()
        ));
        let policy=AuthorizationClockPolicy {
            max_age_seconds: 3600,
            allowed_skew_seconds: 30,
            require_expiry: true,
        };
        let store=SqliteAuthorizationStore::open_with_relying_party_and_clock_policy(
            &path,"rp-clock-policy",policy
        ).unwrap();
        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party_and_clock_policy(
                &path,
                "rp-clock-policy",
                AuthorizationClockPolicy {
                    max_age_seconds: 7200,
                    allowed_skew_seconds: 30,
                    require_expiry: true,
                }
            ),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party_and_clock_policy(
                &std::env::temp_dir().join(format!(
                    "symthaea-gis-auth-clock-policy-invalid-{}.db",std::process::id()
                )),
                "rp-clock-policy-invalid",
                AuthorizationClockPolicy {
                    max_age_seconds: 0,
                    allowed_skew_seconds: 0,
                    require_expiry: true,
                }
            ),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        drop(store);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn authorization_validity_window_cannot_be_extended_by_reusing_instance() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-validity-immutable-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-validity-immutable"
        ).unwrap();
        let effect=super::super::ActionEffectBinding::new(
            "target-validity-immutable","prod","adapter"
        );
        let action=EpistemicAction::new(
            "validity-immutable","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"validity-immutable".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T06:00:00Z".into(),
            expires_at:Some("2026-10-04T06:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            "support","policy@1",1,2
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-validity-immutable-1","boundary-validity"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,
            "attempt-validity-immutable-1",&action,&effect,"boundary-validity",
            "operation:validity-immutable-1","native-validity-immutable-1"
        ).unwrap();
        store.commit_bound_verified(
            &record,&verified_evidence(&record,ExecutionOutcome::Failed),&TestProviderVerifier
        ).unwrap();

        let extended=ActionAuthorizationWitness {
            issued_at:"2026-10-03T06:00:00Z".into(),
            expires_at:Some("2026-10-05T06:00:00Z".into()),
            ..witness.clone()
        };
        assert!(matches!(
            store.prepare_for_execution_bound(
                &extended,&action,"frame@1",
                "attempt-validity-immutable-2","boundary-validity"
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::AuthorizationValidityWindowFailed
            ))
        ));
        let persisted:(String,Option<String>,String)=store.connection().unwrap().query_row(
            "SELECT validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_leases WHERE authorization_instance=?1",
            params![witness.authorization_instance.as_str()],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))
        ).unwrap();
        assert_eq!(persisted.0,witness.issued_at);
        assert_eq!(persisted.1,witness.expires_at);
        assert_eq!(persisted.2,store.clock_policy.digest());
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn expired_before_dispatch_closes_authorization_instance() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-validity-dispatch-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-validity-dispatch"
        ).unwrap();
        let effect=super::super::ActionEffectBinding::new(
            "target-validity-dispatch","prod","adapter"
        );
        let action=EpistemicAction::new(
            "validity-dispatch","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"validity-dispatch".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T06:00:00Z".into(),
            expires_at:Some("2026-10-03T12:00:00Z".into()),
            authority_epoch:3,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            "support","policy@1",3,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1",
            "attempt-validity-dispatch","boundary-validity"
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_leases
             SET validity_expires_at='2026-10-03T05:59:59Z'
             WHERE authorization_instance=?1",
            params![witness.authorization_instance.as_str()],
        ).unwrap();

        let result=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,
            "attempt-validity-dispatch",&action,&effect,"boundary-validity",
            "operation:validity-dispatch","native-validity-dispatch"
        );
        assert!(matches!(
            result,
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::AuthorizationValidityWindowFailed
            ))
        ));
        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_leases WHERE authorization_instance=?1",
            params![witness.authorization_instance.as_str()],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"expired");

        let renewed=ActionAuthorizationWitness {
            issued_at:"2026-10-03T08:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            ..witness.clone()
        };
        assert!(matches!(
            store.prepare_for_execution_bound(
                &renewed,&action,"frame@1",
                "attempt-validity-dispatch-renewed","boundary-validity"
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::NotReady
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn effectful_validity_window_is_rejected_at_admission() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-validity-admission-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-validity").unwrap();
        let effect=super::super::ActionEffectBinding::new("target-validity","prod","adapter");
        let action=EpistemicAction::new(
            "validity-admission","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"validity-admission".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-02T19:00:00Z".into(),
            expires_at:Some("2026-10-02T19:30:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            "support","policy@1",1,1
        )).unwrap();

        assert!(matches!(
            store.prepare_for_execution_bound(
                &witness,&action,"frame@1","attempt-validity-admission","boundary-validity"
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::AuthorizationValidityWindowFailed
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn expired_after_dispatch_is_closed_as_not_entered() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-validity-preentry-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-validity-preentry").unwrap();
        let effect=super::super::ActionEffectBinding::new("target-validity-preentry","prod","adapter");
        let action=EpistemicAction::new(
            "validity-preentry","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(),
            authorization_instance:"validity-preentry".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T06:00:00Z".into(),
            expires_at:Some("2026-10-03T12:00:00Z".into()),
            authority_epoch:7,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            "support","policy@1",7,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-validity-preentry","boundary-validity"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,"validity-preentry","attempt-validity-preentry",&action,&effect,
            "boundary-validity","operation:validity-preentry","native-validity-preentry"
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_dispatches
             SET validity_expires_at='2026-10-03T05:59:59Z'
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();

        assert!(matches!(
            store.mark_invoked_bound(&record),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::AuthorizationValidityWindowFailed
            ))
        ));

        let states:(String,String)=store.connection().unwrap().query_row(
            "SELECT d.state,l.state FROM authorization_dispatches d
             JOIN authorization_leases l ON l.authorization_instance=d.authorization_instance
             WHERE d.authorization_instance=?1 AND d.attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?))
        ).unwrap();
        assert_eq!(states.0,"not_entered");
        assert_eq!(states.1,"ready");

        let _=std::fs::remove_file(path);
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_provenance_less_bound_apis_fail_closed() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-legacy-fence-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());

        assert!(matches!(
            SqliteAuthorizationStore::mark_dispatch_pending_bound(
                &store,
                &witness.authorization_instance,
                "attempt-legacy-api",
                &action,
                &effect,
                "boundary-A",
                "operation:legacy-api",
                "native-replay:legacy-api",
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            ))
        ));
        assert!(matches!(
            SqliteAuthorizationStore::mark_dispatch_pending_bound_from_native_authority(
                &store,
                &witness.authorization_instance,
                "attempt-legacy-authority-api",
                &action,
                &effect,
                "boundary-A",
                "operation:legacy-authority-api",
                "authority",
                "native-auth",
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_entry_lookup_cannot_be_used_as_terminal_outcome() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-verifier-kind-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            authorization_instance:"verifier-kind".into(),
            action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-02T20:30:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(), action.id.clone(), witness.action_digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-kind","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-kind",&action,&effect,"boundary-A"
        ,
            format!("operation:{}", "attempt-kind"),
            format!("native-replay:{}", &witness.authorization_instance)).unwrap();
        let mut evidence=verified_evidence(&record,ExecutionOutcome::Failed);
        evidence.kind=ProviderEvidenceKind::PreEntryLookup;
        assert!(matches!(
            store.commit_bound_verified(&record,&evidence,&TestProviderVerifier),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn verified_terminal_evidence_is_exact_attempt_and_sink_bound() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-verifier-binding-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            authorization_instance:"verifier-binding".into(),
            action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-02T20:31:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(), action.id.clone(), witness.action_digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-exact","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-exact",&action,&effect,"boundary-A"
        ,
            format!("operation:{}", "attempt-exact"),
            format!("native-replay:{}", &witness.authorization_instance)).unwrap();
        let mut evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        evidence.provider_idempotency_key.push_str("-forged");
        assert!(matches!(
            store.commit_bound_verified(&record,&evidence,&TestProviderVerifier),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn reconciled_receipt_preserves_native_derived_provider_identity() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-reconcile-key-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            authorization_instance:"reconcile-key".into(), action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(), support_digest:witness.support_digest,
            frame:witness.frame, policy:witness.policy, authority_epoch:witness.authority_epoch,
        };
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-reconcile-key","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-reconcile-key",&action,&effect,"boundary-A",
            "operation:reconcile-key","native-grant:reconcile"
        ).unwrap();
        store.mark_invoked_bound(&record).unwrap();
        store.recover_incomplete_attempt_for_boundary("boundary-A","attempt-reconcile-key").unwrap();
        let receipt=store.reconcile_indeterminate_bound_verified(
            &record,&verified_evidence(&record,ExecutionOutcome::Succeeded),&TestProviderVerifier
        ).unwrap();
        assert_eq!(receipt.provider_idempotency_key,record.provider_idempotency_key);
        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        let connection=reopened.connection().unwrap();
        let persisted:String=connection.query_row(
            "SELECT provider_idempotency_key
             FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(persisted,record.provider_idempotency_key);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn provider_idempotency_key_is_derived_from_native_replay_identity() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-provider-key-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            authorization_instance:"provider-key-fence".into(),
            action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(),
            support_digest:witness.support_digest,
            frame:witness.frame,
            policy:witness.policy,
            authority_epoch:witness.authority_epoch,
        };
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-provider-key","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-provider-key",&action,&effect,"boundary-A",
            "operation:provider-key","native-grant:one"
        ).unwrap();
        let native_replay=super::super::NativeReplayDerivation::derive(
            "test-authority/v1","native-grant:one"
        ).unwrap();
        assert_eq!(record.native_replay_identity,native_replay.native_replay_identity);
        let expected=AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),action.canonical_action_digest(),
            witness.support_digest.clone(),witness.policy.clone(),witness.authority_epoch,1
        ).provider_idempotency_key_for_native_replay(
            &native_replay.native_replay_identity,"target-A"
        ).unwrap();
        assert_eq!(record.provider_idempotency_key,expected);
        let mut operation_changed=record.clone();
        operation_changed.operation_id="operation:provider-key-renamed".into();
        assert_eq!(record.provider_idempotency_key,operation_changed.provider_idempotency_key);
        let mut attempt_changed=record.clone();
        attempt_changed.attempt_id="attempt-provider-key-retry".into();
        assert_eq!(record.provider_idempotency_key,attempt_changed.provider_idempotency_key);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn durable_state_survives_reopen_and_blocks_replay() {

        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-provider-key-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            authorization_instance:"provider-key-fence".into(),
            action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(),
            support_digest:witness.support_digest,
            frame:witness.frame,
            policy:witness.policy,
            authority_epoch:witness.authority_epoch,
        };
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-provider-key","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-provider-key",&action,&effect,"boundary-A",
            "operation:provider-key","native-grant:one"
        ).unwrap();
        assert!(record.provider_idempotency_key != AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(), action.id.clone(), action.canonical_action_digest(),
            witness.support_digest.clone(), witness.policy.clone(), witness.authority_epoch, 1
        ).provider_idempotency_key());

        let native_replay=super::super::NativeReplayDerivation::derive(
            "test-authority/v1","native-grant:one"
        ).unwrap();
        assert_eq!(record.native_replay_identity,native_replay.native_replay_identity);
        let expected=AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),action.canonical_action_digest(),
            witness.support_digest.clone(),witness.policy.clone(),witness.authority_epoch,1
        ).provider_idempotency_key_for_native_replay(
            &native_replay.native_replay_identity,"target-A"
        ).unwrap();
        assert_eq!(record.provider_idempotency_key,expected);
        let mut operation_changed=record.clone();
        operation_changed.operation_id="operation:provider-key-renamed".into();
        assert_eq!(record.provider_idempotency_key,operation_changed.provider_idempotency_key);
        let mut attempt_changed=record.clone();
        attempt_changed.attempt_id="attempt-provider-key-retry".into();
        assert_eq!(record.provider_idempotency_key,attempt_changed.provider_idempotency_key);
        let _=std::fs::remove_file(path);
    }
    #[test]
    fn native_authority_entry_derives_replay_identity_at_boundary() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-native-derived-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            authorization_instance:"native-derived".into(),action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(),frame:witness.frame, support_digest:witness.support_digest,
            policy:witness.policy,authority_epoch:witness.authority_epoch,
        };
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-native-derived","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_from_native_authority_for_test(&store,
            &witness.authorization_instance,"attempt-native-derived",&action,&effect,"boundary-A",
            "operation:native-derived","issuer.example","native-auth-1"
        ).unwrap();
        let expected=super::super::NativeReplayDerivation::derive("issuer.example","native-auth-1").unwrap();
        assert_eq!(record.native_replay_identity,expected.native_replay_identity);
        assert!(record.provider_idempotency_key.starts_with("sha256:"));

        let connection=store.connection().unwrap();
        let persisted:(Option<String>,Option<String>,Option<String>)=connection.query_row(
            "SELECT native_authority_namespace,native_authorization_id,native_replay_derivation_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))
        ).unwrap();
        assert_eq!(persisted.0,Some(expected.authority_namespace.clone()));
        assert_eq!(persisted.1,Some(expected.native_authorization_id.clone()));
        assert_eq!(persisted.2,Some(expected.derivation_digest.clone()));

        connection.execute(
            "UPDATE authorization_dispatches
             SET native_replay_derivation_digest='sha256:forged'
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();
        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        assert!(matches!(
            store.commit_bound_verified(&record,&evidence,&TestProviderVerifier),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            ))
        ));
        connection.execute(
            "UPDATE authorization_dispatches
             SET native_replay_derivation_digest=?3
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id,expected.derivation_digest.as_str()],
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        store.commit_bound_verified(&record,&evidence,&TestProviderVerifier).unwrap();
        let terminal:(Option<String>,Option<String>,Option<String>)=connection.query_row(
            "SELECT native_authority_namespace,native_authorization_id,native_replay_derivation_digest
             FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))
        ).unwrap();
        assert_eq!(terminal.0,persisted.0);
        assert_eq!(terminal.1,persisted.1);
        assert_eq!(terminal.2,persisted.2);
        let _=std::fs::remove_file(path);
    }
    #[test]
    fn verifier_configuration_cannot_cross_relying_party_domain() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-verifier-rp-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-A").unwrap();
        let effect=super::super::ActionEffectBinding::new("target-verifier-rp","prod","adapter-A");
        let action=EpistemicAction::new("verifier-rp","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            authorization_instance:"verifier-rp".into(), action_id:action.id.clone(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"support".into(), policy:"policy@1".into(),
            decision:"execute".into(), issued_at:"2026-10-03T06:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "verifier-rp",action.id.clone(),digest,"support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-verifier-rp","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            "verifier-rp","attempt-verifier-rp",&action,&effect,"boundary-A",
            "operation:verifier-rp","native:verifier-rp"
        ).unwrap();
        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        struct WrongRpVerifier;
        impl ProviderEvidenceVerifier for WrongRpVerifier {
            fn verify(
                &self,
                purpose: ProviderVerificationPurpose,
                _record: &DurableDispatchRecord,
                evidence: &ProviderTerminalEvidence,
            ) -> Result<VerifiedProviderOutcome, ProviderVerificationError> {
                if !matches!(purpose,ProviderVerificationPurpose::TerminalOutcome) {
                    return Err(ProviderVerificationError::VerificationFailed);
                }
                Ok(VerifiedProviderOutcome {
                    evidence:evidence.clone(),
                    configuration:ProviderVerifierConfiguration {
                        relying_party_id:"rp-B".into(),
                        verifier_id:"test-verifier/v1".into(),
                        verifier_config_digest:"sha256:test-verifier-config".into(),
                        trust_anchor_digest:"sha256:test-trust-anchors".into(),
                        evidence_profile_digest:"sha256:test-evidence-profile".into(),
                    },
                    verification_digest:"sha256:test-verification".into(),
                })
            }
        }
        assert!(matches!(
            store.commit_bound_verified(&record,&evidence,&WrongRpVerifier),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pinned_native_authority_namespace_drives_canonical_replay_derivation() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-native-pin-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-native-pin").unwrap();

        assert!(matches!(
            store.mark_dispatch_pending_bound_from_pinned_native_authority(
                "missing","attempt-missing",
                &EpistemicAction::new("missing-action","intervention",super::super::ActionRisk::Critical),
                &super::super::ActionEffectBinding::new("target-missing","prod","adapter"),
                "boundary","operation","issuer.unpinned","native-auth"
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            ))
        ));

        store.pin_native_authority_namespace(
            "issuer.example","issuer.example/authority/v1"
        ).unwrap();
        store.pin_native_authority_namespace(
            "issuer.example","issuer.example/authority/v1"
        ).unwrap();
        assert!(matches!(
            store.pin_native_authority_namespace(
                "issuer.example","issuer.example/authority/v2"
            ),
            Err(AuthorizationStoreError::InvalidState(_))
        ));

        let effect=super::super::ActionEffectBinding::new("target-pin","prod","adapter-pin");
        let action=EpistemicAction::new("native-pin-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            authorization_instance:"native-pin".into(), action_id:action.id.clone(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"support".into(), policy:"policy@1".into(),
            decision:"execute".into(), issued_at:"2026-10-03T06:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "native-pin",action.id.clone(),digest,"support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-native-pin","boundary-pin"
        ).unwrap();

        let record=store.mark_dispatch_pending_bound_from_pinned_native_authority(
            "native-pin","attempt-native-pin",&action,&effect,"boundary-pin",
            "operation:native-pin","issuer.example","native-auth-pin"
        ).unwrap();
        let expected=super::super::NativeReplayDerivation::derive(
            "issuer.example/authority/v1","native-auth-pin"
        ).unwrap();
        assert_eq!(record.native_replay_identity,expected.native_replay_identity);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn operation_and_native_replay_identity_tampering_is_rejected() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-identity-fence-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            authorization_instance:"identity-fence".into(),
            action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(),
            support_digest:witness.support_digest,
            frame:witness.frame,
            policy:witness.policy,
            authority_epoch:witness.authority_epoch,
        };
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-identity","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-identity",&action,&effect,"boundary-A",
            "operation:identity-fence","native-replay:identity-fence"
        ).unwrap();

        let mut wrong_operation=record.clone();
        wrong_operation.operation_id.push_str("-forged");
        assert!(matches!(
            store.mark_invoked_bound(&wrong_operation),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let mut wrong_native=record.clone();
        wrong_native.native_replay_identity.push_str("-forged");
        assert!(matches!(
            store.mark_invoked_bound(&wrong_native),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let mut evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        evidence.operation_id.push_str("-forged");
        assert!(matches!(
            store.commit_bound_verified(&record,&evidence,&TestProviderVerifier),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired))
        ));

        let _=std::fs::remove_file(path);
    }


        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-1").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-1").unwrap();
        let first=store.commit(&action.id,"attempt-1",ExecutionOutcome::Succeeded).unwrap();
        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        reopened.recover_incomplete_attempts().unwrap();
        assert!(matches!(
            reopened.prepare_for_execution(&witness,&action,"frame@1","attempt-2"),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::BudgetExhausted))
        ));
        assert_eq!(reopened.commit(&action.id,"attempt-1",ExecutionOutcome::Succeeded).unwrap(),first);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn explicit_authorization_instances_allow_fresh_issuance_without_replay() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-instance-{}.db",std::process::id()));
        let (store,action,old_witness)=fixture(&path);
        store.prepare_for_execution(&old_witness,&action,"frame@1","old-attempt").unwrap();
        store.mark_dispatch_pending(&old_witness.authorization_instance,"old-attempt").unwrap();
        store.commit(&action.id,"old-attempt",ExecutionOutcome::Succeeded).unwrap();

        let digest=action.canonical_action_digest();
        let fresh_witness=ActionAuthorizationWitness {
            authorization_instance:"approval-new".into(),
            issued_at:"2026-10-02T20:02:00Z".into(),
            ..old_witness
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-new", action.id.clone(), digest, "sha256:support", "policy-v1", 1, 1,
        )).unwrap();

        store.prepare_for_execution(&fresh_witness,&action,"frame@1","new-attempt").unwrap();
        store.mark_dispatch_pending("approval-new","new-attempt").unwrap();
        let receipt=store.commit("approval-new","new-attempt",ExecutionOutcome::Succeeded).unwrap();
        assert_eq!(receipt.authorization_instance,"approval-new");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn reopen_converts_nonterminal_dispatch_to_indeterminate() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-recovery-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-crash").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-crash").unwrap();
        drop(store);

        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        assert!(matches!(
            reopened.prepare_for_execution(&witness,&action,"frame@1","attempt-retry"),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::IndeterminateRequiresReconciliation
            ))
        ));
        assert_eq!(
            reopened.commit(&witness.authorization_instance,"attempt-crash",ExecutionOutcome::Succeeded)
                .unwrap_err().to_string(),
            "authorization consumption error: IndeterminateRequiresReconciliation"
        );
        let reconciled=reopened.reconcile_indeterminate(
            &witness.authorization_instance,"attempt-crash",ExecutionOutcome::Succeeded
        ).unwrap();
        assert_eq!(reconciled.outcome,ExecutionOutcome::Succeeded);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn invoked_state_is_durable_and_committable() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-invoked-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-invoked").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-invoked").unwrap();
        store.mark_invoked(&witness.authorization_instance,"attempt-invoked").unwrap();
        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        let receipt=reopened.commit(
            &witness.authorization_instance,"attempt-invoked",ExecutionOutcome::Succeeded
        ).unwrap();
        assert_eq!(receipt.outcome,ExecutionOutcome::Succeeded);
        assert_eq!(receipt.provider_idempotency_key, AuthorizationLease::new(
            action.id.clone(), action.canonical_action_digest(), "sha256:support", "policy-v1", 1, 1
        ).provider_idempotency_key());
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn bound_dispatch_record_freezes_effect_and_boundary_identity() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-bound-dispatch-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("bound-dispatch-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:action.id.clone(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:00:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new(action.id.clone(),digest,"sha256:support","policy-v1",1,1)).unwrap();
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-bound").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,&witness.authorization_instance,"attempt-bound",&action,&effect,"boundary-A",
            format!("operation:{}", "attempt-bound"),
            format!("native-replay:{}", &witness.authorization_instance)).unwrap();
        assert_eq!(record.target_identity,"target-A");
        let mut tampered=record.clone(); tampered.adapter="adapter-B".into();
        assert!(matches!(store.mark_invoked_bound(&tampered),Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))));
        store.mark_invoked_bound(&record).unwrap();
        assert!(store.mark_invoked_bound(&record).is_err());
        let mut wrong_key=record.clone();
        wrong_key.provider_idempotency_key.push_str("-tampered");
        assert!(matches!(
            store.commit_bound(&wrong_key,ExecutionOutcome::Succeeded),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));
        assert!(matches!(
            store.commit_bound(&record,ExecutionOutcome::Succeeded),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            ))
        ));
        let receipt=store.commit_bound_verified(
            &record,
            &verified_evidence(&record,ExecutionOutcome::Succeeded),
            &TestProviderVerifier,
        ).unwrap();
        assert_eq!(receipt.provider_idempotency_key,record.provider_idempotency_key);
        let _=std::fs::remove_file(path);
    }


    #[test]
    fn fresh_native_authority_cannot_bypass_same_action_in_flight() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-action-fence-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("same-action-fence","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();

        let witness_a=ActionAuthorizationWitness {
            authorization_instance:"fence-a".into(), action_id:action.id.clone(),
            action_digest:digest.clone(), support_digest:"support-a".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            authorization_instance:"fence-b".into(), action_id:action.id.clone(),
            action_digest:digest, support_digest:"support-b".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness_a.authorization_instance.clone(), witness_a.action_id.clone(),
            witness_a.action_digest.clone(), witness_a.support_digest.clone(),
            witness_a.policy.clone(), witness_a.authority_epoch, 1
        )).unwrap();
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness_b.authorization_instance.clone(), witness_b.action_id.clone(),
            witness_b.action_digest.clone(), witness_b.support_digest.clone(),
            witness_b.policy.clone(), witness_b.authority_epoch, 1
        )).unwrap();

        store.prepare_for_execution_bound(&witness_a,&action,"frame@1","attempt-fence-a","boundary-A").unwrap();
        mark_dispatch_pending_bound_for_test(&store,
            &witness_a.authorization_instance,"attempt-fence-a",&action,&effect,"boundary-A",
            "operation:fence-a","native-grant:fence-a"
        ).unwrap();

        store.prepare_for_execution_bound(&witness_b,&action,"frame@1","attempt-fence-b","boundary-A").unwrap();
        assert!(matches!(
            mark_dispatch_pending_bound_for_test(&store,
                &witness_b.authorization_instance,"attempt-fence-b",&action,&effect,"boundary-A",
                "operation:fence-b","native-grant:fence-b"
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ActionAlreadyInFlight
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn executed_action_instance_remains_closed_to_fresh_authority() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-action-closed-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-closed").unwrap();
        store.pin_native_authority_namespace("issuer.closed","issuer.closed/authority/v1").unwrap();

        let effect=super::super::ActionEffectBinding::new("target-closed","prod","adapter-closed");
        let action=EpistemicAction::new("closed-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();

        let witness_a=ActionAuthorizationWitness {
            authorization_instance:"closed-a".into(), action_id:action.id.clone(),
            action_digest:digest.clone(), support_digest:"support-a".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), decision:"execute".into(),
            issued_at:"2026-10-03T06:30:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "closed-a",action.id.clone(),digest.clone(),"support-a","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness_a,&action,"frame@1","attempt-closed-a","boundary-A"
        ).unwrap();
        let record_a=store.mark_dispatch_pending_bound_from_pinned_native_authority(
            "closed-a","attempt-closed-a",&action,&effect,"boundary-A",
            "operation:closed-a","issuer.closed","native-closed-a"
        ).unwrap();
        store.mark_invoked_bound(&record_a).unwrap();
        store.commit_bound_verified(
            &record_a,
            &verified_evidence(&record_a,ExecutionOutcome::Succeeded),
            &TestProviderVerifier,
        ).unwrap();

        let witness_b=ActionAuthorizationWitness {
            authorization_instance:"closed-b".into(), action_id:action.id.clone(),
            action_digest:record_a.action_digest.clone(), support_digest:"support-b".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), decision:"execute".into(),
            issued_at:"2026-10-03T06:30:01Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "closed-b",action.id.clone(),record_a.action_digest.clone(),"support-b","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness_b,&action,"frame@1","attempt-closed-b","boundary-B"
        ).unwrap();
        assert!(matches!(
            store.mark_dispatch_pending_bound_from_pinned_native_authority(
                "closed-b","attempt-closed-b",&action,&effect,"boundary-B",
                "operation:closed-b","issuer.closed","native-closed-b"
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ActionAlreadyClosed
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn native_replay_identity_cannot_be_reserved_twice() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-native-replay-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");

        let action_a=EpistemicAction::new("native-a","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let action_b=EpistemicAction::new("native-b","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let witness_a=ActionAuthorizationWitness {
            authorization_instance:"native-a".into(), action_id:action_a.id.clone(),
            action_digest:action_a.canonical_action_digest(), support_digest:"support-a".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            authorization_instance:"native-b".into(), action_id:action_b.id.clone(),
            action_digest:action_b.canonical_action_digest(), support_digest:"support-b".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness_a.authorization_instance.clone(), witness_a.action_id.clone(),
            witness_a.action_digest.clone(), witness_a.support_digest.clone(),
            witness_a.policy.clone(), witness_a.authority_epoch, 1
        )).unwrap();
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness_b.authorization_instance.clone(), witness_b.action_id.clone(),
            witness_b.action_digest.clone(), witness_b.support_digest.clone(),
            witness_b.policy.clone(), witness_b.authority_epoch, 1
        )).unwrap();
        store.prepare_for_execution_bound(&witness_a,&action_a,"frame@1","attempt-native-a","boundary-A").unwrap();
        store.prepare_for_execution_bound(&witness_b,&action_b,"frame@1","attempt-native-b","boundary-A").unwrap();
        mark_dispatch_pending_bound_for_test(&store,
            &witness_a.authorization_instance,"attempt-native-a",&action_a,&effect,"boundary-A",
            "operation:native-a","native-replay:shared"
        ).unwrap();
        let duplicate=mark_dispatch_pending_bound_for_test(&store,
            &witness_b.authorization_instance,"attempt-native-b",&action_b,&effect,"boundary-A",
            "operation:native-b","native-replay:shared"
        );
        assert!(matches!(duplicate,Err(AuthorizationStoreError::Sqlite(_))));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn legacy_transitions_cannot_bypass_boundary_owned_attempts() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-legacy-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("boundary-legacy-fence","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-legacy-fence".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:09:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-legacy-fence",action.id.clone(),digest,"sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-legacy-fence","boundary-A"
        ).unwrap();
        assert!(matches!(
            store.mark_dispatch_pending("approval-legacy-fence","attempt-legacy-fence"),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let record=mark_dispatch_pending_bound_for_test(&store,
            "approval-legacy-fence","attempt-legacy-fence",&action,&effect,"boundary-A"
        ,
            format!("operation:{}", "attempt-legacy-fence"),
            format!("native-replay:{}", "approval-legacy-fence")).unwrap();
        assert!(matches!(
            store.mark_invoked("approval-legacy-fence","attempt-legacy-fence"),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));
        store.mark_invoked_bound(&record).unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_dispatch_recovery_releases_prepared_attempt_and_records_marker() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-pre-recovery-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("pre-recovery","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-pre-recovery".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:13:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-pre-recovery",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-pre","boundary-A"
        ).unwrap();

        let recovery=RecoveryAuthorizationWitness {
            authorization_instance:witness.authorization_instance.clone(),
            attempt_id:"attempt-pre".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:14:00Z".into(),
        };
        assert!(store.recover_pre_dispatch_attempt(&recovery).unwrap());
        assert!(!store.recover_pre_dispatch_attempt(&recovery).unwrap());
        assert_eq!(store.recover_incomplete_attempts().unwrap(),0);
        assert_eq!(store.recover_incomplete_attempts_for_boundary("boundary-A").unwrap(),0);

        let old_attempt=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-pre",&action,
            &super::super::ActionEffectBinding::new("target-A","prod","adapter-A"),
            "boundary-A"
        ,
            format!("operation:{}", "attempt-pre"),
            format!("native-replay:{}", &witness.authorization_instance));
        assert!(old_attempt.is_err());

        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-pre-retry","boundary-A"
        ).unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_dispatch_recovery_rejects_tampered_scope_and_digest() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-pre-recovery-fence-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("pre-recovery-fence","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-pre-fence".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:15:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-pre-fence",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-pre-fence","boundary-A"
        ).unwrap();

        let wrong_boundary=RecoveryAuthorizationWitness {
            authorization_instance:"approval-pre-fence".into(),
            attempt_id:"attempt-pre-fence".into(),
            boundary_id:"boundary-B".into(),
            action_digest:digest.clone(),
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:16:00Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&wrong_boundary),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let wrong_policy=RecoveryAuthorizationWitness {
            authorization_instance:"approval-pre-fence".into(),
            attempt_id:"attempt-pre-fence".into(),
            boundary_id:"boundary-A".into(),
            action_digest:"sha256:action".into(),
            policy:"forged-recovery-policy".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:16:00Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&wrong_policy),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let wrong_digest=RecoveryAuthorizationWitness {
            authorization_instance:"approval-pre-fence".into(),
            attempt_id:"attempt-pre-fence".into(),
            boundary_id:"boundary-A".into(),
            action_digest:"sha256:forged".into(),
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:16:01Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&wrong_digest),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let correct=RecoveryAuthorizationWitness {
            authorization_instance:"approval-pre-fence".into(),
            attempt_id:"attempt-pre-fence".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:16:02Z".into(),
        };
        assert!(store.recover_pre_dispatch_attempt(&correct).unwrap());
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_dispatch_recovery_cannot_release_after_dispatch_pending() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-pre-recovery-after-dispatch-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("pre-recovery-after-dispatch","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-after-dispatch".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:17:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-after-dispatch",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-after-dispatch","boundary-A"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            "approval-after-dispatch","attempt-after-dispatch",&action,&effect,"boundary-A"
        ,
            format!("operation:{}", "attempt-after-dispatch"),
            format!("native-replay:{}", "approval-after-dispatch")).unwrap();

        let recovery=RecoveryAuthorizationWitness {
            authorization_instance:"approval-after-dispatch".into(),
            attempt_id:"attempt-after-dispatch".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:18:00Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&recovery),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed
            ))
        ));
        assert_eq!(record.boundary_id,"boundary-A");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn boundary_scoped_recovery_cannot_claim_another_boundary() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-recovery-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=super::super::ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("boundary-recovery-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-boundary".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:10:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-boundary","boundary-A"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-boundary",&action,&effect,"boundary-A"
        ,
            format!("operation:{}", "attempt-boundary"),
            format!("native-replay:{}", &witness.authorization_instance)).unwrap();
        drop(store);

        let boundary_b=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(boundary_b.recover_incomplete_attempts().unwrap(),0);
        assert_eq!(
            boundary_b
                .recover_incomplete_attempt_for_boundary("boundary-B","attempt-boundary")
                .unwrap(),
            0
        );

        let mut wrong=record.clone();
        wrong.boundary_id="boundary-B".into();
        assert!(matches!(
            boundary_b.reconcile_indeterminate_bound_verified(&wrong,&verified_evidence(&wrong,ExecutionOutcome::Succeeded),&TestProviderVerifier),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let boundary_a=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(
            boundary_a
                .recover_incomplete_attempt_for_boundary("boundary-A","attempt-boundary")
                .unwrap(),
            1
        );
        let receipt=boundary_a.reconcile_indeterminate_bound_verified(&record,&verified_evidence(&record,ExecutionOutcome::Succeeded),&TestProviderVerifier).unwrap();
        assert_eq!(receipt.outcome,ExecutionOutcome::Succeeded);
        assert_eq!(receipt.authorization_instance,"approval-boundary");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn boundary_ownership_survives_crash_before_dispatch_record() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-prepared-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("boundary-prepared","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            action_id:action.id.clone(), authorization_instance:"approval-prepared".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:12:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-prepared",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-prepared","boundary-A"
        ).unwrap();
        drop(store);

        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(reopened.recover_incomplete_attempts().unwrap(),0);
        assert_eq!(
            reopened
                .recover_incomplete_attempt_for_boundary("boundary-A","attempt-prepared")
                .unwrap(),
            0
        );

        let recovery=RecoveryAuthorizationWitness {
            authorization_instance:witness.authorization_instance.clone(),
            attempt_id:"attempt-prepared".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest,
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-02T20:12:30Z".into(),
        };
        assert!(reopened.recover_pre_dispatch_attempt(&recovery).unwrap());
        assert!(matches!(
            mark_dispatch_pending_bound_for_test(&reopened,
                &witness.authorization_instance,"attempt-prepared",&action,
                &super::super::ActionEffectBinding::new("target-A","prod","adapter-A"),
                "boundary-A"
            ,
            format!("operation:{}", "attempt-prepared"),
            format!("native-replay:{}", &witness.authorization_instance)),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::AttemptMismatch))
        ));
        let _=std::fs::remove_file(path);
    }


    #[test]
    fn bound_attempt_id_cannot_be_reused_across_boundaries() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-attempt-scope-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action_a=EpistemicAction::new("scope-a","a",super::super::ActionRisk::Critical);
        let action_b=EpistemicAction::new("scope-b","b",super::super::ActionRisk::Critical);
        let digest_a=action_a.canonical_action_digest();
        let digest_b=action_b.canonical_action_digest();
        let witness_a=ActionAuthorizationWitness {
            action_id:action_a.id.clone(), authorization_instance:"scope-approval-a".into(),
            action_digest:digest_a.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:11:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            action_id:action_b.id.clone(), authorization_instance:"scope-approval-b".into(),
            action_digest:digest_b.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:11:01Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "scope-approval-a",action_a.id.clone(),digest_a,"sha256:support","policy-v1",1,1
        )).unwrap();
        store.register_lease(&AuthorizationLease::new_with_instance(
            "scope-approval-b",action_b.id.clone(),digest_b,"sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness_a,&action_a,"frame@1","attempt-reused","boundary-A"
        ).unwrap();
        assert!(matches!(
            store.prepare_for_execution_bound(
                &witness_b,&action_b,"frame@1","attempt-reused","boundary-B"
            ),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn recovery_treats_invoked_as_indeterminate() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-invoked-recovery-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-crash").unwrap();
        store.mark_dispatch_pending(&witness.authorization_instance,"attempt-crash").unwrap();
        store.mark_invoked(&witness.authorization_instance,"attempt-crash").unwrap();
        drop(store);

        let reopened=SqliteAuthorizationStore::open(&path).unwrap();
        assert_eq!(reopened.recover_incomplete_attempts().unwrap(),1);
        assert!(matches!(
            reopened.prepare_for_execution(&witness,&action,"frame@1","attempt-retry"),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::IndeterminateRequiresReconciliation
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn dispatch_pending_wrong_attempt_is_fenced() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-dispatch-fence-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        store.prepare_for_execution(&witness,&action,"frame@1","attempt-1").unwrap();
        assert_eq!(
            store.mark_dispatch_pending(&witness.authorization_instance,"attempt-2").unwrap_err().to_string(),
            "authorization consumption error: AttemptMismatch"
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn relying_party_scope_is_pinned_per_store() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-rp-scope-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-A").unwrap();
        assert_eq!(store.relying_party_id(),"rp-A");
        drop(store);

        let same=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-A").unwrap();
        assert_eq!(same.relying_party_id(),"rp-A");
        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party(&path,"rp-B"),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party(&path,""),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn same_action_fence_is_scoped_to_pinned_relying_party_domain() {
        let path_a=std::env::temp_dir().join(format!("symthaea-gis-auth-rp-a-{}.db",std::process::id()));
        let path_b=std::env::temp_dir().join(format!("symthaea-gis-auth-rp-b-{}.db",std::process::id()));
        let store_a=SqliteAuthorizationStore::open_with_relying_party(&path_a,"rp-A").unwrap();
        let store_b=SqliteAuthorizationStore::open_with_relying_party(&path_b,"rp-B").unwrap();
        let effect=super::super::ActionEffectBinding::new("target-cross-rp","prod","adapter-A");
        let action=EpistemicAction::new("rp-scoped-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();

        for (store,instance,attempt,boundary,operation,native) in [
            (&store_a,"rp-approval-a","attempt-rp-a","boundary-A","operation-rp-a","native-rp-a"),
            (&store_b,"rp-approval-b","attempt-rp-b","boundary-B","operation-rp-b","native-rp-b"),
        ] {
            let witness=ActionAuthorizationWitness {
                authorization_instance:instance.into(),
                action_id:action.id.clone(),
                action_digest:digest.clone(),
                frame:"frame@1".into(),
                support_digest:"support".into(),
                policy:"policy@1".into(),
                decision:"execute".into(),
                issued_at:"2026-10-03T06:00:00Z".into(),
                expires_at:Some("2026-10-04T12:00:00Z".into()),
                authority_epoch:1,
            };
            store.register_lease(&AuthorizationLease::new_with_instance(
                instance,action.id.clone(),digest.clone(),"support","policy@1",1,1
            )).unwrap();
            store.prepare_for_execution_bound(&witness,&action,"frame@1",attempt,boundary).unwrap();
            let record=mark_dispatch_pending_bound_for_test(&store,
                instance,attempt,&action,&effect,boundary,operation,native
            ).unwrap();
            assert_eq!(record.target_identity,"target-cross-rp");
            let persisted_rp:String=store.connection().unwrap().query_row(
                "SELECT relying_party_id FROM authorization_dispatches
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![instance,attempt],
                |row| row.get(0)
            ).unwrap();
            assert_eq!(persisted_rp,store.relying_party_id());
        }

        let _=std::fs::remove_file(path_a);
        let _=std::fs::remove_file(path_b);
    }

    #[test]
    fn current_instance_store_is_upgraded_with_boundary_columns_and_indexes() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-migrate-{}.db",std::process::id()));
        {
            let connection=Connection::open(&path).unwrap();
            connection.execute_batch(
                "CREATE TABLE authorization_leases (
                   authorization_instance TEXT PRIMARY KEY, action_id TEXT NOT NULL,
                   action_digest TEXT NOT NULL, support_digest TEXT NOT NULL,
                   policy TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
                   remaining_executions INTEGER NOT NULL, state TEXT NOT NULL, attempt_id TEXT
                 );
                 CREATE TABLE authorization_receipts (
                   authorization_instance TEXT NOT NULL, action_id TEXT NOT NULL,
                   attempt_id TEXT NOT NULL, phase TEXT NOT NULL, outcome TEXT NOT NULL,
                   action_digest TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
                   PRIMARY KEY(authorization_instance,attempt_id,phase)
                 );
                 CREATE TABLE authorization_dispatches (
                   authorization_instance TEXT NOT NULL, attempt_id TEXT NOT NULL,
                   action_id TEXT NOT NULL, action_digest TEXT NOT NULL,
                   provider_idempotency_key TEXT NOT NULL, target_identity TEXT NOT NULL,
                   audience TEXT NOT NULL, adapter TEXT NOT NULL, boundary_id TEXT NOT NULL,
                   state TEXT NOT NULL, PRIMARY KEY(authorization_instance,attempt_id)
                 );",
            ).unwrap();
        }
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let lease=store.connection().unwrap();

        let lease_boundary:String=lease.query_row(
            "SELECT name FROM pragma_table_info('authorization_leases')
             WHERE name='boundary_id'",
            [], |row| row.get(0)
        ).unwrap();
        let receipt_boundary:String=lease.query_row(
            "SELECT name FROM pragma_table_info('authorization_receipts')
             WHERE name='boundary_id'",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(lease_boundary,"boundary_id");
        assert_eq!(receipt_boundary,"boundary_id");

        let rp_count:i64=lease.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('authorization_dispatches') WHERE name='relying_party_id'",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(rp_count,1);

        let pin_table_count:i64=lease.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type='table' AND name='authorization_native_authority_pins'",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(pin_table_count,1);

        let metadata:String=lease.query_row(
            "SELECT value FROM authorization_store_metadata WHERE key='relying_party_id'",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(metadata,"legacy-local");

        for table in ["authorization_dispatches","authorization_terminal_evidence"] {
            for column in [
                "native_authority_namespace",
                "native_authorization_id",
                "native_replay_derivation_digest",
            ] {
                let count:i64=lease.query_row(
                    &format!("SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name=?1"),
                    params![column],
                    |row| row.get(0),
                ).unwrap();
                assert_eq!(count,1,"{table}.{column} missing after migration");
            }
        }

        let index_count:i64=lease.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type='index' AND name IN
             ('authorization_lease_attempt_id_uq','authorization_dispatch_attempt_id_uq')",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(index_count,2);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn legacy_action_keyed_store_is_migrated_to_explicit_instances() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-migrate-{}.db",std::process::id()));
        {
            let connection=Connection::open(&path).unwrap();
            connection.execute_batch(
                "CREATE TABLE authorization_leases (
                   action_id TEXT PRIMARY KEY, action_digest TEXT NOT NULL,
                   support_digest TEXT NOT NULL, policy TEXT NOT NULL,
                   authority_epoch INTEGER NOT NULL, remaining_executions INTEGER NOT NULL,
                   state TEXT NOT NULL, attempt_id TEXT
                 );
                 CREATE TABLE authorization_receipts (
                   action_id TEXT NOT NULL, attempt_id TEXT NOT NULL, phase TEXT NOT NULL,
                   outcome TEXT NOT NULL, action_digest TEXT NOT NULL,
                   authority_epoch INTEGER NOT NULL,
                   PRIMARY KEY(action_id,attempt_id,phase)
                 );
                 INSERT INTO authorization_leases VALUES
                   ('legacy-action','sha256:action','sha256:support','policy-v1',1,1,'ready',NULL);",
            ).unwrap();
        }
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let lease=store.connection().unwrap();
        let instance:String=lease.query_row(
            "SELECT authorization_instance FROM authorization_leases WHERE action_id='legacy-action'",
            [], |row| row.get(0)
        ).unwrap();
        assert_eq!(instance,"legacy-action");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn concurrent_prepare_has_one_winner() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-concurrent-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let second=SqliteAuthorizationStore::open(&path).unwrap();
        let barrier=Arc::new(std::sync::Barrier::new(2));
        let a2=action.clone(); let w2=witness.clone();
        let b1=barrier.clone(); let b2=barrier.clone();
        let t1=thread::spawn(move||{b1.wait();store.prepare_for_execution(&witness,&action,"frame@1","a")});
        let t2=thread::spawn(move||{b2.wait();second.prepare_for_execution(&w2,&a2,"frame@1","b")});
        let r1=t1.join().unwrap(); let r2=t2.join().unwrap();
        assert_ne!(r1.is_ok(),r2.is_ok());
        let _=std::fs::remove_file(path);
    }
}
