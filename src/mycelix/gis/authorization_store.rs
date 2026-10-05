// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Durable shared authorization-consumption domain.
//!
//! SQLite is the authoritative reservation/consumption state machine. The
//! external effect remains a separate boundary: an uncertain effect becomes
//! Indeterminate and requires explicit reconciliation.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::{Duration, DateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};
use reqwest::Url;

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
    /// Logical operation identity supplied by the recovery authority.
    /// Required by strict operation-bound recovery; legacy witnesses may leave it empty.
    pub operation_id: String,
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
            && match lease.operation_id.as_deref() {
                Some(operation_id) => {
                    !self.operation_id.is_empty() && self.operation_id == operation_id
                }
                None => self.operation_id.is_empty(),
            }
            && self.action_digest == lease.action_digest
            // Recovery policy is authenticated by the recovery authority. The
            // durable store binds the claim to the exact lease/attempt identity,
            // not to the policy namespace used to authorize recovery itself.
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
pub enum ProviderStatusVerificationPurpose {
    Admission,
    PreEntry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderStatusEvidence {
    pub status_identifier: String,
    pub status_source_digest: String,
    pub status_observed_at: String,
    pub status_valid_until: String,
    pub status_evidence_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderStatusVerificationError {
    Revoked,
    Stale,
    Unavailable,
    Unauthenticated,
    InvalidBinding,
    VerificationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderStatusVerifierConfiguration {
    pub verifier_id: String,
    pub verifier_revision: String,
    pub verifier_implementation_id: String,
    pub verifier_implementation_digest: String,
    pub verifier_config_digest: String,
}

impl ProviderStatusVerifierConfiguration {
    pub fn new(
        verifier_id: impl Into<String>,
        verifier_revision: impl Into<String>,
        verifier_implementation_id: impl Into<String>,
        verifier_implementation_digest: impl Into<String>,
        verifier_config_digest: impl Into<String>,
    ) -> Self {
        Self {
            verifier_id: verifier_id.into(),
            verifier_revision: verifier_revision.into(),
            verifier_implementation_id: verifier_implementation_id.into(),
            verifier_implementation_digest: verifier_implementation_digest.into(),
            verifier_config_digest: verifier_config_digest.into(),
        }
    }

    fn validate(&self) -> Result<(), AuthorizationStoreError> {
        if self.verifier_id.is_empty()
            || self.verifier_revision.is_empty()
            || self.verifier_implementation_id.is_empty()
            || self.verifier_implementation_digest.is_empty()
            || self.verifier_config_digest.is_empty()
        {
            return Err(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired.into()
            );
        }
        Ok(())
    }
}

pub trait ProviderStatusVerifier {
    /// Stable identity/configuration selected by the relying party.
    /// Legacy implementations return an unconfigured value and are rejected by
    /// the strict effectful boundary until explicitly pinned.
    fn configuration(&self) -> ProviderStatusVerifierConfiguration {
        ProviderStatusVerifierConfiguration::new("", "", "", "", "")
    }

    fn verify_current_status(
        &self,
        purpose: ProviderStatusVerificationPurpose,
        issuer: &str,
        authority_namespace: &str,
        native_authorization_id: &str,
        status_identifier: &str,
        action_digest: &str,
        target_identity: &str,
        audience: &str,
        adapter: &str,
        expected_source_digest: Option<&str>,
    ) -> Result<ProviderStatusEvidence, ProviderStatusVerificationError>;
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
        self.digest_for_clock(Self::CLOCK_SOURCE_ID)
    }

    fn digest_for_clock(&self, clock_source_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:gis:authorization-clock-policy:v1\n");
        hasher.update(clock_source_id.as_bytes());
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
            if expiry > issued + Duration::seconds(self.max_age_seconds as i64) {
                return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
            }
        }

        let normalize = |value: DateTime<Utc>| {
            value.to_rfc3339_opts(SecondsFormat::Secs, true)
        };
        Ok((normalize(issued), expiry.map(normalize)))
    }
}

pub trait TrustedAuthorizationClock: Send + Sync {
    fn source_id(&self) -> &str;
    fn now_utc(&self) -> Result<DateTime<Utc>, AuthorizationStoreError>;
}

#[derive(Debug, Default)]
pub struct SystemUtcClock;

impl TrustedAuthorizationClock for SystemUtcClock {
    fn source_id(&self) -> &str {
        "system-utc-wall-clock-v1"
    }

    fn now_utc(&self) -> Result<DateTime<Utc>, AuthorizationStoreError> {
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
    /// Exact revision of the relying-party-selected verifier implementation/profile.
    pub verifier_revision: String,
    /// Stable identifier for the exact verifier implementation.
    pub verifier_implementation_id: String,
    /// Digest of the exact verifier implementation.
    pub verifier_implementation_digest: String,
    /// Digest of the exact verifier configuration used for provider evidence.
    pub verifier_config_digest: String,
    /// Digest of the trust anchors/status inputs selected by the relying party.
    pub trust_anchor_digest: String,
    /// Digest of the evidence profile used to classify terminal provider evidence.
    pub evidence_profile_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderAdapterConfiguration {
    /// Stable adapter identity selected by the relying party.
    pub adapter_id: String,
    /// Exact adapter revision selected by the relying party.
    pub adapter_revision: String,
    /// Digest of the exact adapter implementation/configuration.
    pub implementation_digest: String,
}

impl ProviderAdapterConfiguration {
    pub fn new(
        adapter_id: impl Into<String>,
        adapter_revision: impl Into<String>,
        implementation_digest: impl Into<String>,
    ) -> Self {
        Self {
            adapter_id: adapter_id.into(),
            adapter_revision: adapter_revision.into(),
            implementation_digest: implementation_digest.into(),
        }
    }

    fn validate(&self) -> Result<(), AuthorizationStoreError> {
        if self.adapter_id.is_empty()
            || self.adapter_revision.is_empty()
            || self.implementation_digest.is_empty()
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Ok(())
    }
}

impl ProviderVerifierConfiguration {
    fn validate(&self) -> Result<(), AuthorizationStoreError> {
        if self.relying_party_id.is_empty()
            || self.verifier_id.is_empty()
            || self.verifier_revision.is_empty()
            || self.verifier_implementation_id.is_empty()
            || self.verifier_implementation_digest.is_empty()
            || self.verifier_config_digest.is_empty()
            || self.trust_anchor_digest.is_empty()
            || self.evidence_profile_digest.is_empty()
        {
            return Err(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into()
            );
        }
        Ok(())
    }
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
    pub adapter_revision: String,
    pub adapter_implementation_digest: String,
    pub boundary_id: String,
    /// Commitment of the exact durable attempt whose effect is being evidenced.
    pub attempt_binding_digest: String,
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
    /// Frozen native issuer used to select the relying-party pin.
    pub native_issuer: String,
    /// Frozen authority namespace resolved from the issuer pin.
    pub native_authority_namespace: String,
    /// Stable native authorization identifier from which native replay identity was derived.
    pub native_authorization_id: String,
    /// Digest proving how the native replay identity was derived.
    pub native_replay_derivation_digest: String,
    pub status_identifier: String,
    pub status_source_digest: String,
    pub status_observed_at: String,
    pub status_valid_until: String,
    pub status_evidence_digest: String,
    pub action_id: String,
    pub action_digest: String,
    pub provider_idempotency_key: String,
    pub target_identity: String,
    pub audience: String,
    pub adapter: String,
    pub adapter_revision: String,
    pub adapter_implementation_digest: String,
    /// Stable boundary identity. It scopes attempt ownership without entering
    /// the shared action key, so separate boundary instances cannot claim one
    /// another's dispatch records.
    pub boundary_id: String,
    /// Deterministic commitment over the immutable provenance tuple for this attempt.
    /// Lifecycle state is intentionally excluded; transitions are protected separately.
    pub attempt_binding_digest: String,
}

fn compute_attempt_scope_digest(
    boundary_id: &str,
    attempt_id: &str,
) -> Result<String, AuthorizationStoreError> {
    if boundary_id.is_empty() || attempt_id.is_empty() {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:gis:effect-boundary-attempt-scope:v1\n");
    hasher.update((boundary_id.len() as u64).to_be_bytes());
    hasher.update(boundary_id.as_bytes());
    hasher.update((attempt_id.len() as u64).to_be_bytes());
    hasher.update(attempt_id.as_bytes());
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

fn backfill_status_check_boundary_ownership(
    connection: &mut Connection,
) -> Result<(), AuthorizationStoreError> {
    connection.execute(
        "UPDATE authorization_status_checks
         SET boundary_id=(
             SELECT d.boundary_id
             FROM authorization_dispatches d
             WHERE d.authorization_instance=authorization_status_checks.authorization_instance
               AND d.attempt_id=authorization_status_checks.attempt_id
         )
         WHERE (boundary_id IS NULL OR boundary_id='')
           AND EXISTS(
             SELECT 1 FROM authorization_dispatches d
             WHERE d.authorization_instance=authorization_status_checks.authorization_instance
               AND d.attempt_id=authorization_status_checks.attempt_id
               AND d.boundary_id IS NOT NULL AND d.boundary_id <> ''
           )",
        [],
    )?;

    let rows: Vec<(String,String)> = {
        let mut stmt = connection.prepare(
            "SELECT attempt_id,boundary_id
             FROM authorization_status_checks
             WHERE attempt_id <> ''
               AND boundary_id IS NOT NULL AND boundary_id <> ''
               AND (attempt_scope_digest IS NULL OR attempt_scope_digest='')"
        )?;
        let mapped = stmt.query_map([], |row| {
            Ok((row.get::<_,String>(0)?, row.get::<_,String>(1)?))
        })?;
        mapped.collect::<Result<Vec<_>, _>>()?
    };
    for (attempt_id,boundary_id) in rows {
        let scope = compute_attempt_scope_digest(&boundary_id,&attempt_id)?;
        connection.execute(
            "UPDATE authorization_status_checks
             SET attempt_scope_digest=?1
             WHERE attempt_id=?2 AND boundary_id=?3
               AND (attempt_scope_digest IS NULL OR attempt_scope_digest='')",
            params![scope,attempt_id,boundary_id],
        )?;
    }
    Ok(())
}

fn backfill_status_check_operation_ids_from_dispatch(
    connection: &mut Connection,
) -> Result<(), AuthorizationStoreError> {
    connection.execute(
        "UPDATE authorization_status_checks
         SET operation_id=(
             SELECT d.operation_id
             FROM authorization_dispatches d
             WHERE d.authorization_instance=authorization_status_checks.authorization_instance
               AND d.attempt_id=authorization_status_checks.attempt_id
               AND d.operation_id IS NOT NULL
               AND d.operation_id <> ''
         )
         WHERE (operation_id IS NULL OR operation_id='')
           AND EXISTS(
             SELECT 1
             FROM authorization_dispatches d
             WHERE d.authorization_instance=authorization_status_checks.authorization_instance
               AND d.attempt_id=authorization_status_checks.attempt_id
               AND d.operation_id IS NOT NULL
               AND d.operation_id <> ''
           )",
        [],
    )?;
    Ok(())
}

fn backfill_receipt_operation_ids_from_dispatch(
    connection: &mut Connection,
) -> Result<(), AuthorizationStoreError> {
    connection.execute(
        "UPDATE authorization_receipts
         SET operation_id=(
             SELECT d.operation_id
             FROM authorization_dispatches d
             WHERE d.authorization_instance=authorization_receipts.authorization_instance
               AND d.attempt_id=authorization_receipts.attempt_id
               AND d.operation_id IS NOT NULL
               AND d.operation_id <> ''
         )
         WHERE (operation_id IS NULL OR operation_id='')
           AND EXISTS(
             SELECT 1
             FROM authorization_dispatches d
             WHERE d.authorization_instance=authorization_receipts.authorization_instance
               AND d.attempt_id=authorization_receipts.attempt_id
               AND d.operation_id IS NOT NULL
               AND d.operation_id <> ''
           )",
        [],
    )?;
    Ok(())
}

fn backfill_bound_attempt_boundaries_from_dispatch(
    connection: &mut Connection,
) -> Result<(), AuthorizationStoreError> {
    // Receipts and terminal evidence can span schema generations. When a
    // bound dispatch is authoritative, fill only a missing boundary owner.
    // Never overwrite an already-present historical boundary.
    for table in ["authorization_receipts", "authorization_terminal_evidence"] {
        connection.execute(
            &format!(
                "UPDATE {table}
                 SET boundary_id=(
                     SELECT d.boundary_id
                     FROM authorization_dispatches d
                     WHERE d.authorization_instance={table}.authorization_instance
                       AND d.attempt_id={table}.attempt_id
                 )
                 WHERE (boundary_id IS NULL OR boundary_id='')
                   AND EXISTS(
                     SELECT 1
                     FROM authorization_dispatches d
                     WHERE d.authorization_instance={table}.authorization_instance
                       AND d.attempt_id={table}.attempt_id
                       AND d.boundary_id IS NOT NULL
                       AND d.boundary_id <> ''
                   )"
            ),
            [],
        )?;
    }
    Ok(())
}

fn validate_attempt_operation_consistency(
    connection: &Connection,
) -> Result<(), AuthorizationStoreError> {
    let mut owners =
        std::collections::BTreeMap::<(String,String),String>::new();
    for table in [
        "authorization_leases",
        "authorization_receipts",
        "authorization_status_checks",
        "authorization_recovery_markers",
        "authorization_terminal_evidence",
        "authorization_dispatches",
    ] {
        let mut stmt = connection.prepare(&format!(
            "SELECT authorization_instance,attempt_id,operation_id
             FROM {table}
             WHERE attempt_id IS NOT NULL
               AND attempt_id <> ''
               AND operation_id IS NOT NULL
               AND operation_id <> ''"
        ))?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_,String>(0)?,
                row.get::<_,String>(1)?,
                row.get::<_,String>(2)?,
            ))
        })?;
        for row in rows {
            let (authorization_instance,attempt_id,operation_id)=row?;
            let key=(authorization_instance.clone(),attempt_id.clone());
            if let Some(existing)=owners.insert(key,operation_id.clone()) {
                if existing != operation_id {
                    return Err(AuthorizationStoreError::InvalidState(format!(
                        "attempt operation mismatch for {authorization_instance}/{attempt_id}: {existing} vs {operation_id} in {table}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_attempt_boundary_consistency(
    connection: &Connection,
) -> Result<(), AuthorizationStoreError> {
    let mut owners =
        std::collections::BTreeMap::<(String,String),String>::new();
    for table in [
        "authorization_leases",
        "authorization_receipts",
        "authorization_status_checks",
        "authorization_recovery_markers",
        "authorization_terminal_evidence",
        "authorization_dispatches",
    ] {
        let mut stmt = connection.prepare(&format!(
            "SELECT authorization_instance,attempt_id,boundary_id
             FROM {table}
             WHERE attempt_id IS NOT NULL
               AND attempt_id <> ''
               AND boundary_id IS NOT NULL
               AND boundary_id <> ''"
        ))?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_,String>(0)?,
                row.get::<_,String>(1)?,
                row.get::<_,String>(2)?,
            ))
        })?;
        for row in rows {
            let (authorization_instance,attempt_id,boundary_id)=row?;
            let key=(authorization_instance.clone(),attempt_id.clone());
            if let Some(existing)=owners.insert(key,boundary_id.clone()) {
                if existing != boundary_id {
                    return Err(AuthorizationStoreError::InvalidState(format!(
                        "attempt boundary mismatch for {authorization_instance}/{attempt_id}:                          {existing} vs {boundary_id} in {table}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_attempt_boundary_consistency_for_attempt(
    tx: &Transaction<'_>,
    authorization_instance: &str,
    attempt_id: &str,
    expected_boundary: &str,
) -> Result<(), AuthorizationStoreError> {
    if authorization_instance.is_empty()
        || attempt_id.is_empty()
        || expected_boundary.is_empty()
    {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }
    for table in [
        "authorization_leases",
        "authorization_receipts",
        "authorization_status_checks",
        "authorization_recovery_markers",
        "authorization_terminal_evidence",
        "authorization_dispatches",
    ] {
        let mut stmt = tx.prepare(&format!(
            "SELECT boundary_id
             FROM {table}
             WHERE authorization_instance=?1
               AND attempt_id=?2
               AND boundary_id IS NOT NULL
               AND boundary_id <> ''"
        ))?;
        let rows = stmt.query_map(
            params![authorization_instance,attempt_id],
            |row| row.get::<_,String>(0),
        )?;
        for row in rows {
            if row? != expected_boundary {
                return Err(AuthorizationConsumptionError::InvalidBinding.into());
            }
        }
    }
    Ok(())
}

fn validate_attempt_operation_consistency_for_attempt(
    tx: &Transaction<'_>,
    authorization_instance: &str,
    attempt_id: &str,
    expected_operation_id: &str,
) -> Result<(), AuthorizationStoreError> {
    if authorization_instance.is_empty()
        || attempt_id.is_empty()
        || expected_operation_id.is_empty()
    {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }
    for table in [
        "authorization_leases",
        "authorization_receipts",
        "authorization_status_checks",
        "authorization_recovery_markers",
        "authorization_terminal_evidence",
        "authorization_dispatches",
    ] {
        let mut stmt = tx.prepare(&format!(
            "SELECT operation_id
             FROM {table}
             WHERE authorization_instance=?1
               AND attempt_id=?2
               AND operation_id IS NOT NULL
               AND operation_id <> ''"
        ))?;
        let rows = stmt.query_map(
            params![authorization_instance,attempt_id],
            |row| row.get::<_,String>(0),
        )?;
        for row in rows {
            if row? != expected_operation_id {
                return Err(AuthorizationConsumptionError::InvalidBinding.into());
            }
        }
    }
    Ok(())
}

fn backfill_attempt_scope_digests(
    connection: &mut Connection,
) -> Result<(), AuthorizationStoreError> {
    for table in [
        "authorization_leases",
        "authorization_receipts",
        "authorization_status_checks",
        "authorization_recovery_markers",
        "authorization_terminal_evidence",
        "authorization_dispatches",
    ] {
        let rows: Vec<(String,String,Option<String>)> = {
            let mut stmt = connection.prepare(&format!(
                "SELECT attempt_id,boundary_id,attempt_scope_digest
                 FROM {table}
                 WHERE attempt_id IS NOT NULL AND attempt_id <> ''
                   AND boundary_id IS NOT NULL AND boundary_id <> ''"
            ))?;
            let mapped = stmt.query_map([], |row| {
                Ok((
                    row.get::<_,String>(0)?,
                    row.get::<_,String>(1)?,
                    row.get::<_,Option<String>>(2)?,
                ))
            })?;
            mapped.collect::<Result<Vec<_>,_>>()?
        };
        for (attempt_id,boundary_id,stored_scope) in rows {
            let expected=compute_attempt_scope_digest(&boundary_id,&attempt_id)?;
            match stored_scope.as_deref() {
                None | Some("") => {
                    connection.execute(
                        &format!(
                            "UPDATE {table}
                             SET attempt_scope_digest=?1
                             WHERE attempt_id=?2 AND boundary_id=?3
                               AND (attempt_scope_digest IS NULL OR attempt_scope_digest='')"
                        ),
                        params![expected,attempt_id,boundary_id],
                    )?;
                }
                Some(scope) if scope == expected => {}
                Some(_) => {
                    return Err(AuthorizationStoreError::InvalidState(format!(
                        "attempt scope digest mismatch in {table} for attempt {attempt_id}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn append_len_prefixed<H: Digest>(hasher: &mut H, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn append_len_prefixed_bytes(buffer: &mut Vec<u8>, value: &[u8]) {
    buffer.extend_from_slice(&(value.len() as u64).to_be_bytes());
    buffer.extend_from_slice(value);
}

fn compute_attempt_binding_digest(

    authorization_instance: &str,
    attempt_id: &str,
    operation_id: &str,
    native_replay_identity: &str,
    native_issuer: &str,
    native_authority_namespace: &str,
    native_authorization_id: &str,
    native_replay_derivation_digest: &str,
    native_authority_pin_set_id: &str,
    native_authority_pin_set_digest: &str,
    relying_party_id: &str,
    action_id: &str,
    action_digest: &str,
    provider_idempotency_key: &str,
    target_identity: &str,
    audience: &str,
    adapter: &str,
    adapter_revision: &str,
    adapter_implementation_digest: &str,
    status_identifier: &str,
    status_source_digest: &str,
    status_observed_at: &str,
    status_valid_until: &str,
    status_evidence_digest: &str,
    validity_issued_at: &str,
    validity_expires_at: Option<&str>,
    validity_policy_digest: &str,
) -> String {
    let mut material = Vec::with_capacity(512);
    material.extend_from_slice(b"symthaea:gis:attempt-binding:v1\n");
    for value in [
        authorization_instance,
        attempt_id,
        operation_id,
        native_replay_identity,
        native_issuer,
        native_authority_namespace,
        native_authorization_id,
        native_replay_derivation_digest,
        native_authority_pin_set_id,
        native_authority_pin_set_digest,
        relying_party_id,
        action_id,
        action_digest,
        provider_idempotency_key,
        target_identity,
        audience,
        adapter,
        adapter_revision,
        adapter_implementation_digest,
        status_identifier,
        status_source_digest,
        status_observed_at,
        status_valid_until,
        status_evidence_digest,
        validity_issued_at,
    ] {
        append_len_prefixed_bytes(&mut material, value.as_bytes());
    }
    material.push(u8::from(validity_expires_at.is_some()));
    append_len_prefixed_bytes(&mut material, validity_expires_at.unwrap_or("").as_bytes());
    append_len_prefixed_bytes(&mut material, validity_policy_digest.as_bytes());
    format!("sha256:{}", hex::encode(Sha256::digest(material)))
}

fn load_persisted_dispatch_record(
    tx: &Transaction<'_>,
    authorization_instance: &str,
    attempt_id: &str,
    boundary_id: &str,
) -> Result<(DurableDispatchRecord, String), AuthorizationStoreError> {
    if authorization_instance.is_empty() || attempt_id.is_empty() || boundary_id.is_empty() {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }

    let row = tx
        .query_row(
            "SELECT operation_id,native_replay_identity,native_issuer,native_authority_namespace,
                    native_authorization_id,native_replay_derivation_digest,
                    status_identifier,status_source_digest,status_observed_at,status_valid_until,
                    status_evidence_digest,action_id,action_digest,provider_idempotency_key,
                    target_identity,audience,adapter,adapter_revision,
                    adapter_implementation_digest,boundary_id,attempt_binding_digest,state
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
            params![authorization_instance, attempt_id, boundary_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                    row.get::<_, Option<String>>(9)?,
                    row.get::<_, Option<String>>(10)?,
                    row.get::<_, String>(11)?,
                    row.get::<_, String>(12)?,
                    row.get::<_, String>(13)?,
                    row.get::<_, String>(14)?,
                    row.get::<_, String>(15)?,
                    row.get::<_, String>(16)?,
                    row.get::<_, Option<String>>(17)?,
                    row.get::<_, Option<String>>(18)?,
                    row.get::<_, String>(19)?,
                    row.get::<_, String>(20)?,
                    row.get::<_, String>(21)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| AuthorizationStoreError::NotFound(attempt_id.to_owned()))?;

    let (
        operation_id,
        native_replay_identity,
        native_issuer,
        native_authority_namespace,
        native_authorization_id,
        native_replay_derivation_digest,
        status_identifier,
        status_source_digest,
        status_observed_at,
        status_valid_until,
        status_evidence_digest,
        action_id,
        action_digest,
        provider_idempotency_key,
        target_identity,
        audience,
        adapter,
        adapter_revision,
        adapter_implementation_digest,
        stored_boundary_id,
        attempt_binding_digest,
        state,
    ) = row;

    if stored_boundary_id != boundary_id {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }

    let require_non_empty = |value: Option<String>| {
        value
            .filter(|value| !value.is_empty())
            .ok_or_else(|| AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding,
            ))
    };

    let record = DurableDispatchRecord {
        authorization_instance: authorization_instance.to_owned(),
        attempt_id: attempt_id.to_owned(),
        operation_id,
        native_replay_identity,
        native_issuer: require_non_empty(native_issuer)?,
        native_authority_namespace: require_non_empty(native_authority_namespace)?,
        native_authorization_id: require_non_empty(native_authorization_id)?,
        native_replay_derivation_digest: require_non_empty(native_replay_derivation_digest)?,
        status_identifier: require_non_empty(status_identifier)?,
        status_source_digest: require_non_empty(status_source_digest)?,
        status_observed_at: require_non_empty(status_observed_at)?,
        status_valid_until: require_non_empty(status_valid_until)?,
        status_evidence_digest: require_non_empty(status_evidence_digest)?,
        action_id,
        action_digest,
        provider_idempotency_key,
        target_identity,
        audience,
        adapter,
        adapter_revision: require_non_empty(adapter_revision)?,
        adapter_implementation_digest: require_non_empty(adapter_implementation_digest)?,
        boundary_id: stored_boundary_id,
        attempt_binding_digest,
    };

    Ok((record, state))
}

impl DurableDispatchRecord {
    pub fn new(
        authorization_instance: impl Into<String>,
        attempt_id: impl Into<String>,
        operation_id: impl Into<String>,
        native_replay_identity: impl Into<String>,
        native_issuer: impl Into<String>,
        native_authority_namespace: impl Into<String>,
        native_authorization_id: impl Into<String>,
        action_id: impl Into<String>,
        action_digest: impl Into<String>,
        provider_idempotency_key: impl Into<String>,
        effect: &super::ActionEffectBinding,
        adapter_configuration: &ProviderAdapterConfiguration,
        boundary_id: impl Into<String>,
        status: &ProviderStatusEvidence,
        native_replay_derivation_digest: &str,
        native_authority_pin_set_id: &str,
        native_authority_pin_set_digest: &str,
        relying_party_id: &str,
        validity_issued_at: &str,
        validity_expires_at: Option<&str>,
        validity_policy_digest: &str,
    ) -> Self {
        let authorization_instance = authorization_instance.into();
        let attempt_id = attempt_id.into();
        let operation_id = operation_id.into();
        let native_replay_identity = native_replay_identity.into();
        let native_issuer = native_issuer.into();
        let native_authority_namespace = native_authority_namespace.into();
        let native_authorization_id = native_authorization_id.into();
        let action_id = action_id.into();
        let action_digest = action_digest.into();
        let provider_idempotency_key = provider_idempotency_key.into();
        let boundary_id = boundary_id.into();
        let attempt_binding_digest = compute_attempt_binding_digest(
            &authorization_instance,
            &attempt_id,
            &operation_id,
            &native_replay_identity,
            &native_issuer,
            &native_authority_namespace,
            &native_authorization_id,
            native_replay_derivation_digest,
            native_authority_pin_set_id,
            native_authority_pin_set_digest,
            relying_party_id,
            &action_id,
            &action_digest,
            &provider_idempotency_key,
            &effect.target_identity,
            &effect.audience,
            &effect.adapter,
            &adapter_configuration.adapter_revision,
            &adapter_configuration.implementation_digest,
            &status.status_identifier,
            &status.status_source_digest,
            &status.status_observed_at,
            &status.status_valid_until,
            &status.status_evidence_digest,
            validity_issued_at,
            validity_expires_at,
            validity_policy_digest,
        );
        Self {
            authorization_instance,
            attempt_id,
            operation_id,
            native_replay_identity,
            native_issuer,
            native_authority_namespace,
            native_authorization_id,
            native_replay_derivation_digest: native_replay_derivation_digest.to_owned(),
            status_identifier: status.status_identifier.clone(),
            status_source_digest: status.status_source_digest.clone(),
            status_observed_at: status.status_observed_at.clone(),
            status_valid_until: status.status_valid_until.clone(),
            status_evidence_digest: status.status_evidence_digest.clone(),
            action_id,
            action_digest,
            provider_idempotency_key,
            target_identity: effect.target_identity.clone(),
            audience: effect.audience.clone(),
            adapter: effect.adapter.clone(),
            adapter_revision: adapter_configuration.adapter_revision.clone(),
            adapter_implementation_digest: adapter_configuration.implementation_digest.clone(),
            boundary_id,
            attempt_binding_digest,
        }
    }
}
pub struct SqliteAuthorizationStore {
    path: PathBuf,
    relying_party_id: String,
    clock_policy: AuthorizationClockPolicy,
    clock: Arc<dyn TrustedAuthorizationClock>,
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
        Self::open_with_relying_party_clock_and_policy(
            path,
            relying_party_id,
            Arc::new(SystemUtcClock),
            clock_policy,
        )
    }

    pub fn open_with_relying_party_clock_and_policy(
        path: impl AsRef<Path>,
        relying_party_id: impl Into<String>,
        clock: Arc<dyn TrustedAuthorizationClock>,
        clock_policy: AuthorizationClockPolicy,
    ) -> Result<Self, AuthorizationStoreError> {
        clock_policy.validate_configuration()?;
        if clock.source_id().is_empty() {
            return Err(AuthorizationStoreError::InvalidState(
                "trusted clock source id must not be empty".into(),
            ));
        }
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
        let store = Self { path, relying_party_id, clock_policy, clock };
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
             CREATE TABLE IF NOT EXISTS authorization_provider_adapter_pins (
               adapter_id TEXT PRIMARY KEY,
               adapter_revision TEXT NOT NULL,
               implementation_digest TEXT NOT NULL
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
               operation_id TEXT,
               boundary_id TEXT,
               attempt_scope_digest TEXT,
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
               operation_id TEXT,
               provider_idempotency_key TEXT,
               boundary_id TEXT,
               attempt_scope_digest TEXT,
               PRIMARY KEY(authorization_instance, attempt_id, phase)
             );
             CREATE TABLE IF NOT EXISTS authorization_status_checks (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               phase TEXT NOT NULL,
               operation_id TEXT,
               status_identifier TEXT NOT NULL,
               status_source_digest TEXT NOT NULL,
               status_observed_at TEXT NOT NULL,
               status_valid_until TEXT NOT NULL,
               status_evidence_digest TEXT NOT NULL,
               boundary_id TEXT,
               attempt_scope_digest TEXT,
               PRIMARY KEY(authorization_instance,attempt_id,phase)
             );
             CREATE TABLE IF NOT EXISTS authorization_recovery_markers (
               authorization_instance TEXT NOT NULL,
               attempt_id TEXT NOT NULL,
               operation_id TEXT,
               boundary_id TEXT NOT NULL,
               action_digest TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               marker TEXT NOT NULL,
               attempt_scope_digest TEXT,
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
               attempt_scope_digest TEXT,
               action_digest TEXT NOT NULL,
               provider_idempotency_key TEXT NOT NULL,
               target_identity TEXT NOT NULL,
               audience TEXT NOT NULL,
               adapter TEXT,
               adapter_revision TEXT,
               adapter_implementation_digest TEXT,
               outcome TEXT NOT NULL,
               evidence_id TEXT NOT NULL,
               evidence_digest TEXT NOT NULL,
               attempt_binding_digest TEXT NOT NULL,
               verifier_id TEXT NOT NULL,
               verifier_revision TEXT,
               verifier_implementation_id TEXT,
               verifier_implementation_digest TEXT,
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
               status_identifier TEXT,
               status_source_digest TEXT,
               status_observed_at TEXT,
               status_valid_until TEXT,
               status_evidence_digest TEXT,
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
               adapter_revision TEXT NOT NULL,
               adapter_implementation_digest TEXT NOT NULL,
               boundary_id TEXT NOT NULL,
               attempt_scope_digest TEXT,
               attempt_binding_digest TEXT NOT NULL,
               state TEXT NOT NULL,
               PRIMARY KEY(authorization_instance, attempt_id)
             );",
        )?;
        ensure_column(&mut connection, "authorization_leases", "operation_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_leases", "boundary_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_leases", "attempt_scope_digest", "TEXT")?;
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
        ensure_column(&mut connection, "authorization_dispatches", "status_identifier", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "status_source_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "status_observed_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "status_valid_until", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "status_evidence_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "validity_issued_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "validity_expires_at", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "validity_policy_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "relying_party_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_dispatches", "adapter_revision", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_dispatches", "adapter_implementation_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_dispatches", "attempt_binding_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_dispatches", "attempt_scope_digest", "TEXT")?;
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
        ensure_column(&mut connection, "authorization_receipts", "operation_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_receipts", "provider_idempotency_key", "TEXT")?;
        ensure_column(&mut connection, "authorization_receipts", "boundary_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_receipts", "attempt_scope_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_recovery_markers", "operation_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_recovery_markers", "attempt_scope_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "verifier_revision", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "verifier_implementation_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "verifier_implementation_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "verifier_config_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "trust_anchor_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "adapter_revision", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "adapter_implementation_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "attempt_binding_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "attempt_scope_digest", "TEXT")?;
        ensure_column(&mut connection, "authorization_terminal_evidence", "evidence_profile_digest", "TEXT NOT NULL DEFAULT ''")?;
        ensure_column(&mut connection, "authorization_status_checks", "operation_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_status_checks", "boundary_id", "TEXT")?;
        ensure_column(&mut connection, "authorization_status_checks", "attempt_scope_digest", "TEXT")?;
        backfill_status_check_boundary_ownership(&mut connection)?;
        backfill_receipt_operation_ids_from_dispatch(&mut connection)?;
        backfill_status_check_operation_ids_from_dispatch(&mut connection)?;
        backfill_bound_attempt_boundaries_from_dispatch(&mut connection)?;
        backfill_attempt_scope_digests(&mut connection)?;
        validate_attempt_boundary_consistency(&connection)?;
        validate_attempt_operation_consistency(&connection)?;
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
             CREATE UNIQUE INDEX IF NOT EXISTS authorization_lease_operation_uq
               ON authorization_leases(operation_id)
               WHERE operation_id IS NOT NULL AND operation_id <> '';
             DROP INDEX IF EXISTS authorization_dispatch_action_fence_idx;
             CREATE INDEX authorization_dispatch_action_fence_idx
               ON authorization_dispatches(relying_party_id, target_identity, action_digest, state);",
        )?;

        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_native_authority_pin_set(&tx)?;
        let clock_policy_digest = store.clock_policy.digest_for_clock(store.clock.source_id());
        let configured_clock_source: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='authorization_clock_source_id'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match configured_clock_source {
            Some(existing) if existing != store.clock.source_id() => {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "authorization clock source mismatch: store is pinned to {existing}"
                )));
            }
            Some(_) => {}
            None => {
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value) VALUES('authorization_clock_source_id',?1)",
                    params![store.clock.source_id()],
                )?;
            }
        }
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

    fn trusted_utc_now(&self) -> Result<DateTime<Utc>, AuthorizationStoreError> {
        self.clock.now_utc()
    }

    fn connection(&self) -> Result<Connection, AuthorizationStoreError> {
        Ok(Connection::open(&self.path)?)
    }

    pub fn relying_party_id(&self) -> &str {
        &self.relying_party_id
    }

    /// Pin the relying-party-selected provider status source digest.
    ///
    /// The status verifier may authenticate evidence, but it must not choose the
    /// trust source accepted by the boundary. The digest is therefore write-once
    /// and durable alongside the relying-party store policy.
    pub fn pin_provider_status_source_digest(
        &self,
        status_source_digest: &str,
    ) -> Result<(), AuthorizationStoreError> {
        if status_source_digest.is_empty() {
            return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
        }
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_source_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match existing {
            Some(existing) if existing != status_source_digest => {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "provider status source digest is already pinned as {existing}"
                )));
            }
            Some(_) => {}
            None => {
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value)
                     VALUES('provider_status_source_digest',?1)",
                    params![status_source_digest],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Pin the relying-party-selected status verifier identity and configuration digest.
    pub fn pin_provider_status_verifier_configuration(
        &self,
        configuration: &ProviderStatusVerifierConfiguration,
    ) -> Result<(), AuthorizationStoreError> {
        configuration.validate()?;
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing_id: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_id'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let existing_digest: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_config_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let existing_revision: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_revision'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let existing_implementation_id: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_implementation_id'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let existing_implementation_digest: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_implementation_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match (existing_id, existing_digest, existing_revision, existing_implementation_id, existing_implementation_digest) {
            (Some(id), Some(digest), Some(revision), Some(implementation_id), Some(implementation_digest))
                if id == configuration.verifier_id
                    && revision == configuration.verifier_revision
                    && implementation_id == configuration.verifier_implementation_id
                    && implementation_digest == configuration.verifier_implementation_digest
                    && digest == configuration.verifier_config_digest => {}
            (Some(id), Some(digest), None, None, None)
                if id == configuration.verifier_id
                    && digest == configuration.verifier_config_digest => {
                for (key, value) in [
                    ("provider_status_verifier_revision", configuration.verifier_revision.as_str()),
                    ("provider_status_verifier_implementation_id", configuration.verifier_implementation_id.as_str()),
                    ("provider_status_verifier_implementation_digest", configuration.verifier_implementation_digest.as_str()),
                ] {
                    tx.execute(
                        "INSERT INTO authorization_store_metadata(key,value) VALUES(?1,?2)",
                        params![key, value],
                    )?;
                }
            },
            (Some(id), Some(digest), _, _, _) => {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "provider status verifier mismatch: pinned {id}/{digest}"
                )));
            }
            (None, None, None, None, None) => {
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value)
                     VALUES('provider_status_verifier_id',?1)",
                    params![configuration.verifier_id.as_str()],
                )?;
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value)
                     VALUES('provider_status_verifier_revision',?1)",
                    params![configuration.verifier_revision.as_str()],
                )?;
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value)
                     VALUES('provider_status_verifier_implementation_id',?1)",
                    params![configuration.verifier_implementation_id.as_str()],
                )?;
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value)
                     VALUES('provider_status_verifier_implementation_digest',?1)",
                    params![configuration.verifier_implementation_digest.as_str()],
                )?;
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value)
                     VALUES('provider_status_verifier_config_digest',?1)",
                    params![configuration.verifier_config_digest.as_str()],
                )?;
            }
            _ => {
                return Err(AuthorizationStoreError::InvalidState(
                    "provider status verifier metadata is partially configured".into(),
                ));
            }
            }
        tx.commit()?;
        Ok(())
    }

    /// Pin the relying-party-selected terminal provider evidence verifier configuration.
    ///
    /// This is write-once configuration for the durable effect boundary. Presented
    /// verifier metadata cannot silently become a trust anchor for a terminal outcome.
    pub fn pin_provider_evidence_verifier_configuration(
        &self,
        configuration: &ProviderVerifierConfiguration,
    ) -> Result<(), AuthorizationStoreError> {
        configuration.validate()?;
        if configuration.relying_party_id != self.relying_party_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let keys = [
            ("provider_evidence_verifier_relying_party_id", configuration.relying_party_id.as_str()),
            ("provider_evidence_verifier_id", configuration.verifier_id.as_str()),
            ("provider_evidence_verifier_revision", configuration.verifier_revision.as_str()),
            ("provider_evidence_verifier_implementation_id", configuration.verifier_implementation_id.as_str()),
            ("provider_evidence_verifier_implementation_digest", configuration.verifier_implementation_digest.as_str()),
            ("provider_evidence_verifier_config_digest", configuration.verifier_config_digest.as_str()),
            ("provider_evidence_verifier_trust_anchor_digest", configuration.trust_anchor_digest.as_str()),
            ("provider_evidence_verifier_evidence_profile_digest", configuration.evidence_profile_digest.as_str()),
        ];
        let values = [
            configuration.relying_party_id.as_str(),
            configuration.verifier_id.as_str(),
            configuration.verifier_revision.as_str(),
            configuration.verifier_implementation_id.as_str(),
            configuration.verifier_implementation_digest.as_str(),
            configuration.verifier_config_digest.as_str(),
            configuration.trust_anchor_digest.as_str(),
            configuration.evidence_profile_digest.as_str(),
        ];
        let mut existing = Vec::with_capacity(keys.len());
        for (key, _) in keys {
            existing.push(
                tx.query_row(
                    "SELECT value FROM authorization_store_metadata WHERE key=?1",
                    params![key],
                    |row| row.get::<_, String>(0),
                ).optional()?
            );
        }

        if existing.iter().all(Option::is_none) {
            for ((key, _), value) in keys.iter().zip(values.iter()) {
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value) VALUES(?1,?2)",
                    params![key, value],
                )?;
            }
        } else if existing.iter().zip(values.iter()).all(|(stored, expected)| {
            stored.as_deref() == Some(*expected)
        }) {
            // Already pinned to exactly this configuration.
        } else if existing[0].as_deref() == Some(values[0])
            && existing[1].as_deref() == Some(values[1])
            && existing[2].is_none()
            && existing[3].is_none()
            && existing[4].is_none()
            && existing[5].as_deref() == Some(values[5])
            && existing[6].as_deref() == Some(values[6])
            && existing[7].as_deref() == Some(values[7])
        {
            // Complete a historical five-field pin with the newly required
            // revision and implementation provenance. Existing values are never changed.
            for index in 2..5 {
                let (key, _) = keys[index];
                let value = values[index];
                tx.execute(
                    "INSERT INTO authorization_store_metadata(key,value) VALUES(?1,?2)",
                    params![key, value],
                )?;
            }
        } else {
            return Err(AuthorizationStoreError::InvalidState(
                "provider evidence verifier configuration is already pinned differently".into()
            ));
        }

        tx.commit()?;
        Ok(())
    }

    fn pinned_provider_evidence_verifier_configuration(
        &self,
    ) -> Result<ProviderVerifierConfiguration, AuthorizationStoreError> {
        let connection = self.connection()?;
        let read = |key: &str| -> Result<Option<String>, AuthorizationStoreError> {
            Ok(connection
                .query_row(
                    "SELECT value FROM authorization_store_metadata WHERE key=?1",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?)
        };
        let configuration = match (
            read("provider_evidence_verifier_relying_party_id")?,
            read("provider_evidence_verifier_id")?,
            read("provider_evidence_verifier_revision")?,
            read("provider_evidence_verifier_implementation_id")?,
            read("provider_evidence_verifier_implementation_digest")?,
            read("provider_evidence_verifier_config_digest")?,
            read("provider_evidence_verifier_trust_anchor_digest")?,
            read("provider_evidence_verifier_evidence_profile_digest")?,
        ) {
            (Some(relying_party_id), Some(verifier_id), Some(verifier_revision),
             Some(verifier_implementation_id), Some(verifier_implementation_digest),
             Some(verifier_config_digest), Some(trust_anchor_digest), Some(evidence_profile_digest)) =>
                ProviderVerifierConfiguration {
                    relying_party_id,
                    verifier_id,
                    verifier_revision,
                    verifier_implementation_id,
                    verifier_implementation_digest,
                    verifier_config_digest,
                    trust_anchor_digest,
                    evidence_profile_digest,
                },
            _ => return Err(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into()
            ),
        };
        configuration.validate()?;
        if configuration.relying_party_id != self.relying_party_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Ok(configuration)
    }

    fn pinned_provider_status_verifier_configuration(
        &self,
    ) -> Result<ProviderStatusVerifierConfiguration, AuthorizationStoreError> {
        let connection = self.connection()?;
        let verifier_id: Option<String> = connection
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_id'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let verifier_revision: Option<String> = connection
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_revision'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let verifier_implementation_id: Option<String> = connection
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_implementation_id'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let verifier_implementation_digest: Option<String> = connection
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_implementation_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let verifier_config_digest: Option<String> = connection
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_verifier_config_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let configuration = match (
            verifier_id,
            verifier_revision,
            verifier_implementation_id,
            verifier_implementation_digest,
            verifier_config_digest,
        ) {
            (Some(verifier_id), Some(verifier_revision), Some(verifier_implementation_id),
             Some(verifier_implementation_digest), Some(verifier_config_digest)) =>
                ProviderStatusVerifierConfiguration::new(
                    verifier_id,
                    verifier_revision,
                    verifier_implementation_id,
                    verifier_implementation_digest,
                    verifier_config_digest,
                ),
            _ => return Err(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired.into()
            ),
        };
        configuration.validate()?;
        Ok(configuration)
    }

    fn pinned_provider_status_source_digest(
        &self,
    ) -> Result<String, AuthorizationStoreError> {
        let connection = self.connection()?;
        connection
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_source_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                AuthorizationStoreError::Consumption(
                    AuthorizationConsumptionError::ProviderStatusVerificationRequired,
                )
            })
    }

    /// Validate the relying-party-selected status source while holding the
    /// authoritative transaction. This closes the lookup/verification-to-entry
    /// window in which a mutable metadata row could otherwise become stale.
    fn validate_persisted_provider_status_source_digest(
        tx: &Transaction<'_>,
        expected: &str,
    ) -> Result<(), AuthorizationStoreError> {
        if expected.is_empty() {
            return Err(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired.into()
            );
        }
        let stored: Option<String> = tx
            .query_row(
                "SELECT value FROM authorization_store_metadata
                 WHERE key='provider_status_source_digest'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if stored.as_deref() != Some(expected) {
            return Err(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired.into()
            );
        }
        Ok(())
    }

    /// Pin one adapter identity to an exact revision and implementation digest.
    /// The tuple is selected by the relying party and is immutable once used.
    pub fn pin_provider_adapter_configuration(
        &self,
        configuration: &ProviderAdapterConfiguration,
    ) -> Result<(), AuthorizationStoreError> {
        configuration.validate()?;
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, String)> = tx
            .query_row(
                "SELECT adapter_revision,implementation_digest
                 FROM authorization_provider_adapter_pins
                 WHERE adapter_id=?1",
                params![configuration.adapter_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        match existing {
            Some((revision, digest))
                if revision == configuration.adapter_revision
                    && digest == configuration.implementation_digest => {}
            Some((revision, digest)) => {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "adapter {} is already pinned as {revision}/{digest}",
                    configuration.adapter_id
                )));
            }
            None => {
                tx.execute(
                    "INSERT INTO authorization_provider_adapter_pins
                     (adapter_id,adapter_revision,implementation_digest)
                     VALUES (?1,?2,?3)",
                    params![
                        configuration.adapter_id.as_str(),
                        configuration.adapter_revision.as_str(),
                        configuration.implementation_digest.as_str(),
                    ],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn pinned_provider_adapter_configuration(
        &self,
        adapter_id: &str,
    ) -> Result<ProviderAdapterConfiguration, AuthorizationStoreError> {
        if adapter_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        let connection = self.connection()?;
        let configuration = connection
            .query_row(
                "SELECT adapter_id,adapter_revision,implementation_digest
                 FROM authorization_provider_adapter_pins
                 WHERE adapter_id=?1",
                params![adapter_id],
                |row| {
                    Ok(ProviderAdapterConfiguration {
                        adapter_id: row.get(0)?,
                        adapter_revision: row.get(1)?,
                        implementation_digest: row.get(2)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))?;
        configuration.validate()?;
        Ok(configuration)
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

        let normalized_issuer = Self::normalize_native_issuer(issuer);
        let mut stmt = tx.prepare(
            "SELECT issuer,authority_namespace
             FROM authorization_native_authority_pins",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let existing_issuer: String = row.get(0)?;
            let existing_namespace: String = row.get(1)?;
            if Self::normalize_native_issuer(&existing_issuer) == normalized_issuer
                && existing_namespace != authority_namespace
            {
                return Err(AuthorizationStoreError::InvalidState(format!(
                    "authority namespace collision: {issuer} normalizes with {existing_issuer},                      but the established namespaces differ"
                )));
            }
        }

        drop(rows);
        drop(stmt);
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
        let path = url.path().trim_end_matches('/').to_owned();
        url.set_path(&path);
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
        if issuer.is_empty() || namespace.is_empty() {
            return Err(AuthorizationStoreError::InvalidState(
                "invalid native authority pin set: issuer and authority namespace must be non-empty".into(),
            ));
        }
        let normalized = Self::normalize_native_issuer(&issuer);
        if let Some((existing_issuer, existing_namespace)) =
            seen.iter().find(|(existing, _)| Self::normalize_native_issuer(existing) == normalized)
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
            append_len_prefixed_bytes(&mut canonical, issuer.as_bytes());
            append_len_prefixed_bytes(&mut canonical, namespace.as_bytes());
        }

        let pin_set_id = self.native_authority_pin_set_id();
        let mut digest_material = Vec::with_capacity(canonical.len() + pin_set_id.len() + 64);
        digest_material.extend_from_slice(b"symthaea:gis:native-pin-set-digest:v2\n");
        append_len_prefixed_bytes(&mut digest_material, pin_set_id.as_bytes());
        append_len_prefixed_bytes(&mut digest_material, &canonical);
        let pin_set_digest = format!("sha256:v2:{}", hex::encode(Sha256::digest(&digest_material)));
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

    fn validate_native_authority_pin_binding(
        &self,
        tx: &Transaction<'_>,
        issuer: Option<&str>,
        authority_namespace: Option<&str>,
    ) -> Result<(), AuthorizationStoreError> {
        let (Some(issuer), Some(namespace)) = (issuer, authority_namespace) else {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        };
        let normalized = Self::normalize_native_issuer(issuer);
        let mut stmt = tx.prepare(
            "SELECT issuer,authority_namespace
             FROM authorization_native_authority_pins",
        )?;
        let mut rows = stmt.query([])?;
        let mut matched = false;
        while let Some(row) = rows.next()? {
            let pinned_issuer: String = row.get(0)?;
            let pinned_namespace: String = row.get(1)?;
            if Self::normalize_native_issuer(&pinned_issuer) == normalized {
                if pinned_namespace != namespace {
                    return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
                }
                matched = true;
            }
        }
        if !matched {
            return Err(AuthorizationConsumptionError::InvalidNativeReplayProvenance.into());
        }
        Ok(())
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
        let valid = if let Some(hex_digest) = pin_set_digest.strip_prefix("sha256:v2:") {
            let mut digest_material = Vec::with_capacity(canonical.len() + pin_set_id.len() + 64);
            digest_material.extend_from_slice(b"symthaea:gis:native-pin-set-digest:v2\n");
            digest_material.extend_from_slice(&(pin_set_id.len() as u64).to_be_bytes());
            digest_material.extend_from_slice(pin_set_id.as_bytes());
            digest_material.extend_from_slice(&(canonical.len() as u64).to_be_bytes());
            digest_material.extend_from_slice(&canonical);
            hex::encode(Sha256::digest(&digest_material)) == hex_digest
        } else if let Some(hex_digest) = pin_set_digest.strip_prefix("sha256:") {
            // Historical v1 snapshots were content-only; retain validation for
            // durable evidence created before relying-party-scoped v2 digests.
            hex::encode(Sha256::digest(&canonical)) == hex_digest
        } else {
            false
        };
        if !valid {
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
        let normalized = Self::normalize_native_issuer(issuer);
        let connection = self.connection()?;
        let mut stmt = connection.prepare(
            "SELECT issuer,authority_namespace
             FROM authorization_native_authority_pins",
        )?;
        let mut rows = stmt.query([])?;
        let mut matched_namespace: Option<String> = None;
        while let Some(row) = rows.next()? {
            let pinned_issuer: String = row.get(0)?;
            if Self::normalize_native_issuer(&pinned_issuer) != normalized {
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
              remaining_executions, state, attempt_id, operation_id, boundary_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,NULL,NULL)",
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
    #[deprecated(note = "use prepare_for_execution_bound_with_operation for strict operation provenance")]
    pub fn prepare_for_execution_bound(
        &self,
        witness: &ActionAuthorizationWitness,
        action: &EpistemicAction,
        current_frame: &str,
        attempt_id: &str,
        boundary_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        self.prepare_for_execution_bound_internal(
            witness, action, current_frame, attempt_id, boundary_id, None
        )
    }

    /// Canonical effectful preparation. The logical operation ID is frozen
    /// before DispatchPending so exact pre-dispatch recovery can bind it.
    pub fn prepare_for_execution_bound_with_operation(
        &self,
        witness: &ActionAuthorizationWitness,
        action: &EpistemicAction,
        current_frame: &str,
        attempt_id: &str,
        boundary_id: &str,
        operation_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        if operation_id.is_empty() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        self.prepare_for_execution_bound_internal(
            witness,
            action,
            current_frame,
            attempt_id,
            boundary_id,
            Some(operation_id),
        )
    }

    fn prepare_for_execution_bound_internal(
        &self,
        witness: &ActionAuthorizationWitness,
        action: &EpistemicAction,
        current_frame: &str,
        attempt_id: &str,
        boundary_id: &str,
        operation_id: Option<&str>,
    ) -> Result<(), AuthorizationStoreError> {
        if boundary_id.is_empty()
            || attempt_id.is_empty()
            || operation_id.is_some_and(|id| id.is_empty())
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if let Some(operation_id) = operation_id {
            if !witness.is_operation_consistent_with(operation_id) {
                return Err(AuthorizationConsumptionError::InvalidBinding.into());
            }
        }
        let (validity_issued_at, validity_expires_at) = self.clock_policy.validate(
            &witness.issued_at,
            witness.expires_at.as_deref(),
            self.trusted_utc_now()?,
        )?;
        let validity_policy_digest = self.clock_policy.digest_for_clock(self.clock.source_id());

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
        // Attempt and operation identifiers are lifecycle identities, not
        // reusable caller labels. Active leases are covered by the checks below;
        // historical dispatch/recovery records keep the identities consumed
        // after the active lease is released.
        let historical_attempt_owner: Option<String> = tx
            .query_row(
                "SELECT source
                 FROM (
                   SELECT 'authorization_leases' AS source
                   FROM authorization_leases
                   WHERE attempt_id=?1
                     AND attempt_id IS NOT NULL
                     AND attempt_id <> ''
                   UNION ALL
                   SELECT 'authorization_dispatches' AS source
                   FROM authorization_dispatches
                   WHERE attempt_id=?1
                   UNION ALL
                   SELECT 'authorization_recovery_markers' AS source
                   FROM authorization_recovery_markers
                   WHERE attempt_id=?1
                 )
                 LIMIT 1",
                params![attempt_id],
                |row| row.get(0),
            )
            .optional()?;
        if historical_attempt_owner.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if let Some(operation_id) = operation_id {
            let historical_operation_owner: Option<String> = tx
                .query_row(
                    "SELECT source
                     FROM (
                       SELECT 'authorization_leases' AS source
                       FROM authorization_leases
                       WHERE operation_id=?1
                         AND operation_id IS NOT NULL
                         AND operation_id <> ''
                       UNION ALL
                       SELECT 'authorization_dispatches' AS source
                       FROM authorization_dispatches
                       WHERE operation_id=?1
                       UNION ALL
                       SELECT 'authorization_recovery_markers' AS source
                       FROM authorization_recovery_markers
                       WHERE operation_id=?1
                         AND operation_id IS NOT NULL
                         AND operation_id <> ''
                     )
                     LIMIT 1",
                    params![operation_id],
                    |row| row.get(0),
                )
                .optional()?;
            if historical_operation_owner.is_some() {
                return Err(AuthorizationConsumptionError::InvalidBinding.into());
            }
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
             SET state='prepared', attempt_id=?2, operation_id=?3, boundary_id=?4,
                 attempt_scope_digest=?5
             WHERE authorization_instance=?1 AND state='ready'
               AND remaining_executions>0 AND boundary_id IS NULL",
            params![
                witness.authorization_instance.as_str(),
                attempt_id,
                operation_id,
                boundary_id,
                compute_attempt_scope_digest(boundary_id, attempt_id)?,
            ],
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
    fn validate_bound_dispatch_preconditions(
        &self,
        authorization_instance: &str,
        attempt_id: &str,
        action: &EpistemicAction,
        expected_effect: &super::ActionEffectBinding,
        boundary_id: &str,
        operation_id: &str,
        issuer: &str,
        native_authorization_id: &str,
        authority_namespace: &str,
    ) -> Result<(), AuthorizationStoreError> {
        if action.effect_binding.as_ref() != Some(expected_effect)
            || !expected_effect.is_well_formed()
            || authorization_instance.is_empty()
            || attempt_id.is_empty()
            || boundary_id.is_empty()
            || operation_id.is_empty()
            || issuer.is_empty()
            || native_authorization_id.is_empty()
            || authority_namespace.is_empty()
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let current_boundary = load_lease_boundary(&tx, authorization_instance)?;
        if current_boundary.as_deref() != Some(boundary_id) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let lease = load_lease(&tx, authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(authorization_instance.to_owned()))?;
        let persisted_operation_id: Option<String> = tx
            .query_row(
                "SELECT operation_id FROM authorization_leases
                 WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
                params![authorization_instance, attempt_id, boundary_id],
                |row| row.get(0),
            )
            .optional()?;
        if persisted_operation_id.as_deref() != Some(operation_id) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if lease.action_id != action.id
            || lease.action_digest != action.canonical_action_digest()
            || !matches!(
                &lease.state,
                AuthorizationLeaseState::Prepared { attempt_id: id }
                    if id == attempt_id
            )
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        Ok(())
    }

    pub fn mark_dispatch_pending_bound_from_pinned_native_authority<V: ProviderStatusVerifier>(
        &self,
        authorization_instance: &str,
        attempt_id: &str,
        action: &EpistemicAction,
        expected_effect: &super::ActionEffectBinding,
        boundary_id: &str,
        operation_id: &str,
        issuer: &str,
        native_authorization_id: &str,
        status_identifier: &str,
        status_verifier: &V,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        let authority_namespace = self.pinned_native_authority_namespace(issuer)?;
        self.validate_bound_dispatch_preconditions(
            authorization_instance,
            attempt_id,
            action,
            expected_effect,
            boundary_id,
            operation_id,
            issuer,
            native_authorization_id,
            &authority_namespace,
        )?;
        let replay = super::NativeReplayDerivation::derive(
            authority_namespace.clone(),
            native_authorization_id,
        )
        .map_err(|_| AuthorizationConsumptionError::InvalidNativeReplayProvenance)?;
        let adapter_configuration =
            self.pinned_provider_adapter_configuration(&expected_effect.adapter)?;
        let pinned_status_source_digest =
            self.pinned_provider_status_source_digest()?;
        let pinned_status_verifier_configuration =
            self.pinned_provider_status_verifier_configuration()?;
        if status_verifier.configuration() != pinned_status_verifier_configuration {
            self.close_pre_dispatch_status_failure(
                    authorization_instance,
                    attempt_id,
                    boundary_id,
                    operation_id,
                )?;
            return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
        }
        let status = match status_verifier.verify_current_status(
            ProviderStatusVerificationPurpose::Admission,
            issuer,
            &authority_namespace,
            native_authorization_id,
            status_identifier,
            &action.canonical_action_digest(),
            &expected_effect.target_identity,
            &expected_effect.audience,
            &expected_effect.adapter,
            Some(pinned_status_source_digest.as_str()),
        ) {
            Ok(status) => status,
            Err(_) => {
                self.close_pre_dispatch_status_failure(
                    authorization_instance,
                    attempt_id,
                    boundary_id,
                    operation_id,
                )?;
                return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
            }
        };
        self.validate_provider_status_evidence(
            &status, Some(&pinned_status_source_digest), true
        )?;
        self.mark_dispatch_pending_bound_with_provenance(
            authorization_instance,
            attempt_id,
            action,
            expected_effect,
            boundary_id,
            operation_id,
            &replay.native_replay_identity,
            issuer,
            &replay,
            &status,
            &adapter_configuration,
            &pinned_status_verifier_configuration,
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

    fn validate_persisted_provider_status_verifier_configuration(
        &self,
        tx: &Transaction<'_>,
        expected: &ProviderStatusVerifierConfiguration,
    ) -> Result<(), AuthorizationStoreError> {
        for (key, expected_value) in [
            ("provider_status_verifier_id", expected.verifier_id.as_str()),
            ("provider_status_verifier_revision", expected.verifier_revision.as_str()),
            ("provider_status_verifier_implementation_id", expected.verifier_implementation_id.as_str()),
            ("provider_status_verifier_implementation_digest", expected.verifier_implementation_digest.as_str()),
            ("provider_status_verifier_config_digest", expected.verifier_config_digest.as_str()),
        ] {
            let stored: Option<String> = tx
                .query_row(
                    "SELECT value FROM authorization_store_metadata WHERE key=?1",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?;
            if stored.as_deref() != Some(expected_value) {
                return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
            }
        }
        Ok(())
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
        native_issuer: &str,
        native_replay_provenance: &super::NativeReplayDerivation,
        status: &ProviderStatusEvidence,
        adapter_configuration: &ProviderAdapterConfiguration,
        status_verifier_configuration: &ProviderStatusVerifierConfiguration,
    ) -> Result<DurableDispatchRecord, AuthorizationStoreError> {
        if action.effect_binding.as_ref() != Some(expected_effect)
            || boundary_id.is_empty()
            || operation_id.is_empty()
            || native_replay_identity.is_empty()
            || native_issuer.is_empty()
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
        self.validate_provider_status_evidence(status, None, true)?;
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        self.validate_persisted_provider_status_verifier_configuration(
            &tx,
            status_verifier_configuration,
        )?;
        Self::validate_persisted_provider_status_source_digest(
            &tx,
            &status.status_source_digest,
        )?;
        self.validate_native_authority_pin_binding(
            &tx,
            Some(native_issuer),
            Some(native_replay_provenance.authority_namespace.as_str()),
        )?;
        // Revalidate the relying-party adapter pin inside the same authoritative
        // transaction that creates DispatchPending. The adapter configuration
        // resolved before entering this transaction is not itself a trust anchor.
        Self::validate_persisted_adapter_configuration_for_expected(&tx, adapter_configuration)?;

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
        let persisted_operation_id: Option<String> = tx
            .query_row(
                "SELECT operation_id FROM authorization_leases
                 WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
                params![authorization_instance, attempt_id, boundary_id],
                |row| row.get(0),
            )
            .optional()?;
        // The operation identifier is frozen during bound preparation. Never
        // let a caller replace it at DispatchPending, even if every other
        // action/native/status binding still matches.
        if persisted_operation_id.as_deref() != Some(operation_id) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
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

        // Native replay identity is a one-time authority unit. Preflight the
        // durable uniqueness constraint so a collision is reported as a
        // semantic binding refusal rather than leaking a raw SQLite error.
        // The UNIQUE index remains authoritative for races between callers.
        let replay_owner: Option<(String, String, String)> = tx
            .query_row(
                "SELECT authorization_instance,attempt_id,boundary_id
                 FROM authorization_dispatches
                 WHERE native_replay_identity=?1
                 LIMIT 1",
                params![native_replay_identity],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if replay_owner.is_some() {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
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
            self.trusted_utc_now()?,
        ).is_err()
            || lease_validity.2 != self.clock_policy.digest_for_clock(self.clock.source_id())
        {
            let authority_epoch = lease.authority_epoch;
            let changed = tx.execute(
                "UPDATE authorization_leases
                 SET state='expired',attempt_id=NULL,boundary_id=NULL,attempt_scope_digest=NULL
                 WHERE authorization_instance=?1 AND state='prepared' AND attempt_id=?2",
                params![authorization_instance, attempt_id],
            )?;
            if changed != 1 {
                return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
            }
            tx.execute(
                "INSERT OR IGNORE INTO authorization_recovery_markers
                 (authorization_instance,attempt_id,operation_id,boundary_id,action_digest,authority_epoch,marker,attempt_scope_digest)
                 VALUES (?1,?2,NULLIF(?3,''),?4,?5,?6,'not_entered_validity',?7)",
                params![
                    authorization_instance,
                    attempt_id,
                    operation_id,
                    boundary_id,
                    action.canonical_action_digest(),
                    authority_epoch as i64,
                    compute_attempt_scope_digest(boundary_id, attempt_id)?,
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
            native_issuer,
            &native_replay_provenance.authority_namespace,
            &native_replay_provenance.native_authorization_id,
            &action.id,
            &expected_digest,
            &lease
                .provider_idempotency_key_for_native_replay(
                    native_replay_identity,
                    &expected_effect.target_identity,
                )
                .map_err(|_| AuthorizationConsumptionError::InvalidBinding)?,
            expected_effect,
            adapter_configuration,
            boundary_id,
            status,
            &native_replay_provenance.derivation_digest,
            &native_authority_pin_set_id,
            &native_authority_pin_set_digest,
            self.relying_party_id.as_str(),
            lease_validity.0.as_str(),
            lease_validity.1.as_deref(),
            lease_validity.2.as_str(),
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
              native_issuer,native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
              native_authority_pin_set_id,native_authority_pin_set_digest,
              status_identifier,status_source_digest,status_observed_at,status_valid_until,status_evidence_digest,
              validity_issued_at,validity_expires_at,validity_policy_digest,relying_party_id,
              action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,
              adapter_revision,adapter_implementation_digest,boundary_id,attempt_scope_digest,
              attempt_binding_digest,state)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27,?28,?29,?30,?31,'dispatch_pending')",
            params![
                record.authorization_instance, record.attempt_id, record.operation_id,
                record.native_replay_identity,
                native_issuer,
                native_replay_provenance.authority_namespace.as_str(),
                native_replay_provenance.native_authorization_id.as_str(),
                native_replay_provenance.derivation_digest.as_str(),
                native_authority_pin_set_id.as_str(),
                native_authority_pin_set_digest.as_str(),
                status.status_identifier.as_str(),
                status.status_source_digest.as_str(),
                status.status_observed_at.as_str(),
                status.status_valid_until.as_str(),
                status.status_evidence_digest.as_str(),
                lease_validity.0.as_str(),
                lease_validity.1.as_deref(),
                lease_validity.2.as_str(),
                self.relying_party_id.as_str(),
                record.action_id, record.action_digest, record.provider_idempotency_key,
                record.target_identity, record.audience, record.adapter,
                record.adapter_revision, record.adapter_implementation_digest,
                record.boundary_id,
                compute_attempt_scope_digest(&record.boundary_id,&record.attempt_id)?,
                record.attempt_binding_digest],
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
    fn close_pre_dispatch_status_failure(
        &self,
        authorization_instance: &str,
        attempt_id: &str,
        boundary_id: &str,
        operation_id: &str,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let changed = tx.execute(
            "UPDATE authorization_leases
             SET state='ready',attempt_id=NULL,boundary_id=NULL,attempt_scope_digest=NULL
             WHERE authorization_instance=?1 AND state='prepared'
               AND attempt_id=?2 AND boundary_id=?3",
            params![authorization_instance, attempt_id, boundary_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::PreDispatchRecoveryNotAllowed.into());
        }

        let (action_digest, authority_epoch): (String, i64) = tx.query_row(
            "SELECT action_digest,authority_epoch
             FROM authorization_leases
             WHERE authorization_instance=?1",
            params![authorization_instance],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        tx.execute(
            "INSERT OR IGNORE INTO authorization_recovery_markers
             (authorization_instance,attempt_id,operation_id,boundary_id,action_digest,authority_epoch,marker,attempt_scope_digest)
             VALUES (?1,?2,NULLIF(?3,''),?4,?5,?6,'not_entered_status',?7)",
            params![
                authorization_instance,
                attempt_id,
                operation_id,
                boundary_id,
                action_digest,
                authority_epoch,
                compute_attempt_scope_digest(boundary_id, attempt_id)?,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn close_pre_entry_status_failure(
        &self,
        record: &DurableDispatchRecord,
    ) -> Result<(), AuthorizationStoreError> {
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let changed = tx.execute(
            "UPDATE authorization_dispatches
             SET state='not_entered'
             WHERE authorization_instance=?1 AND attempt_id=?2
               AND boundary_id=?3 AND relying_party_id=?4
               AND state='dispatch_pending'",
            params![
                record.authorization_instance.as_str(),
                record.attempt_id.as_str(),
                record.boundary_id.as_str(),
                self.relying_party_id.as_str(),
            ],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into());
        }

        let changed = tx.execute(
            "UPDATE authorization_leases
             SET state='ready',attempt_id=NULL,boundary_id=NULL,attempt_scope_digest=NULL
             WHERE authorization_instance=?1 AND state='dispatch_pending'
               AND attempt_id=?2 AND boundary_id=?3",
            params![
                record.authorization_instance.as_str(),
                record.attempt_id.as_str(),
                record.boundary_id.as_str(),
            ],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::IndeterminateRequiresReconciliation.into());
        }

        let authority_epoch: i64 = tx.query_row(
            "SELECT authority_epoch FROM authorization_leases
             WHERE authorization_instance=?1",
            params![record.authorization_instance.as_str()],
            |row| row.get(0),
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO authorization_recovery_markers
             (authorization_instance,attempt_id,operation_id,boundary_id,action_digest,authority_epoch,marker,attempt_scope_digest)
             VALUES (?1,?2,NULLIF(?3,''),?4,?5,?6,'not_entered_status',?7)",
            params![
                record.authorization_instance.as_str(),
                record.attempt_id.as_str(),
                record.operation_id.as_str(),
                record.boundary_id.as_str(),
                record.action_digest.as_str(),
                authority_epoch,
                compute_attempt_scope_digest(&record.boundary_id,&record.attempt_id)?,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn mark_invoked_bound<V: ProviderStatusVerifier>(
        &self,
        record: &DurableDispatchRecord,
        status_verifier: &V,
    ) -> Result<(), AuthorizationStoreError> {
        let pinned_status_source_digest = self.pinned_provider_status_source_digest()?;
        let pinned_status_verifier_configuration =
            self.pinned_provider_status_verifier_configuration()?;
        if status_verifier.configuration() != pinned_status_verifier_configuration {
            self.close_pre_entry_status_failure(record)?;
            return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
        }
        if record.status_source_digest != pinned_status_source_digest {
            return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
        }
        {
            let mut precheck_connection = self.connection()?;
            let precheck_tx = precheck_connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
            let persisted_state = self.validate_persisted_dispatch_record(&precheck_tx, record)?;
            if persisted_state != "dispatch_pending" {
                return Err(AuthorizationConsumptionError::InvalidBinding.into());
            }
            precheck_tx.commit()?;
        }

        let pre_entry_status = match status_verifier.verify_current_status(
            ProviderStatusVerificationPurpose::PreEntry,
            &record.native_issuer,
            &record.native_authority_namespace,
            &record.native_authorization_id,
            &record.status_identifier,
            &record.action_digest,
            &record.target_identity,
            &record.audience,
            &record.adapter,
            Some(&record.status_source_digest),
        ) {
            Ok(status) => status,
            Err(_) => {
                self.close_pre_entry_status_failure(record)?;
                return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
            }
        };
        if self
            .validate_provider_status_evidence(
                &pre_entry_status,
                Some(&record.status_source_digest),
                true,
            )
            .is_err()
            || pre_entry_status.status_identifier != record.status_identifier
        {
            self.close_pre_entry_status_failure(record)?;
            return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
        }

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        self.validate_persisted_provider_status_verifier_configuration(
            &tx,
            &pinned_status_verifier_configuration,
        )?;
        Self::validate_persisted_provider_status_source_digest(
            &tx,
            &record.status_source_digest,
        )?;
        let persisted_state = self.validate_persisted_dispatch_record(&tx, record)?;
        if persisted_state != "dispatch_pending" {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
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
            Option<String>, Option<String>, Option<String>, Option<String>
        ) = tx.query_row(
            "SELECT native_issuer,native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
                    native_authority_pin_set_id,native_authority_pin_set_digest,
                    validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?)),
        )?;
        Self::validate_persisted_native_replay_provenance(
            record,
            native_provenance.1.as_deref(),
            native_provenance.2.as_deref(),
            native_provenance.3.as_deref(),
        )?;
        self.validate_native_authority_pin_binding(
            &tx,
            native_provenance.0.as_deref(),
            native_provenance.1.as_deref(),
        )?;
        self.validate_native_authority_pin_set_snapshot(
            &tx,
            native_provenance.4.as_deref(),
            native_provenance.5.as_deref(),
        )?;
        self.validate_persisted_authorization_validity(
            native_provenance.6.as_deref(),
            native_provenance.7.as_deref(),
            native_provenance.8.as_deref(),
            false,
        )?;
        if self.validate_persisted_authorization_validity(
            native_provenance.6.as_deref(),
            native_provenance.7.as_deref(),
            native_provenance.8.as_deref(),
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
                 SET state='expired',attempt_id=NULL,boundary_id=NULL,attempt_scope_digest=NULL
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
                 (authorization_instance,attempt_id,operation_id,boundary_id,action_digest,authority_epoch,marker,attempt_scope_digest)
                 VALUES (?1,?2,NULLIF(?3,''),?4,?5,?6,'not_entered_validity',?7)",
                params![
                    record.authorization_instance.as_str(),
                    record.attempt_id.as_str(),
                    record.operation_id.as_str(),
                    record.boundary_id.as_str(),
                    record.action_digest.as_str(),
                    authority_epoch,
                    compute_attempt_scope_digest(&record.boundary_id,&record.attempt_id)?,
                ],
            )?;
            tx.commit()?;
            return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
        }
        tx.execute(
            "INSERT INTO authorization_status_checks
             (authorization_instance,attempt_id,phase,operation_id,status_identifier,status_source_digest,
              status_observed_at,status_valid_until,status_evidence_digest,boundary_id,
              attempt_scope_digest)
             VALUES (?1,?2,'pre_entry',?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                record.authorization_instance.as_str(),
                record.attempt_id.as_str(),
                record.operation_id.as_str(),
                pre_entry_status.status_identifier.as_str(),
                pre_entry_status.status_source_digest.as_str(),
                pre_entry_status.status_observed_at.as_str(),
                pre_entry_status.status_valid_until.as_str(),
                pre_entry_status.status_evidence_digest.as_str(),
                record.boundary_id.as_str(),
                compute_attempt_scope_digest(&record.boundary_id,&record.attempt_id)?,
            ],
        )?;

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

    fn validate_provider_status_evidence(
        &self,
        status: &ProviderStatusEvidence,
        expected_source_digest: Option<&str>,
        enforce_now: bool,
    ) -> Result<(), AuthorizationStoreError> {
        if status.status_identifier.is_empty()
            || status.status_source_digest.is_empty()
            || status.status_observed_at.is_empty()
            || status.status_valid_until.is_empty()
            || status.status_evidence_digest.is_empty()
            || expected_source_digest.is_some_and(|expected| expected != status.status_source_digest)
        {
            return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
        }

        let observed = DateTime::parse_from_rfc3339(&status.status_observed_at)
            .map_err(|_| AuthorizationConsumptionError::ProviderStatusVerificationRequired)?
            .with_timezone(&Utc);
        let valid_until = DateTime::parse_from_rfc3339(&status.status_valid_until)
            .map_err(|_| AuthorizationConsumptionError::ProviderStatusVerificationRequired)?
            .with_timezone(&Utc);
        if valid_until <= observed {
            return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
        }
        if enforce_now {
            let now = self.trusted_utc_now()?;
            let skew = Duration::seconds(self.clock_policy.allowed_skew_seconds as i64);
            if observed > now + skew || now > valid_until + skew {
                return Err(AuthorizationConsumptionError::ProviderStatusVerificationRequired.into());
            }
        }
        Ok(())
    }

    fn validate_recovery_authorization_issued_at(
        &self,
        issued_at: &str,
    ) -> Result<(), AuthorizationStoreError> {
        let issued = DateTime::parse_from_rfc3339(issued_at)
            .map_err(|_| AuthorizationConsumptionError::AuthorizationValidityWindowFailed)?
            .with_timezone(&Utc);
        let now = self.trusted_utc_now()?;
        let skew = Duration::seconds(self.clock_policy.allowed_skew_seconds as i64);
        let max_age = Duration::seconds(self.clock_policy.max_age_seconds as i64);
        if issued > now + skew || now > issued + max_age + skew {
            return Err(AuthorizationConsumptionError::AuthorizationValidityWindowFailed.into());
        }
        Ok(())
    }

    fn validate_persisted_authorization_validity(
        &self,
        issued_at: Option<&str>,
        expires_at: Option<&str>,
        policy_digest: Option<&str>,
        enforce_now: bool,
    ) -> Result<(), AuthorizationStoreError> {
        if policy_digest != Some(self.clock_policy.digest_for_clock(self.clock.source_id()).as_str()) {
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
            self.clock_policy.validate(issued_at, expires_at, self.trusted_utc_now()?)?;
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
            || evidence.adapter_revision != record.adapter_revision
            || evidence.adapter_implementation_digest != record.adapter_implementation_digest
            || evidence.boundary_id != record.boundary_id
            || evidence.attempt_binding_digest != record.attempt_binding_digest
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Ok(())
    }

    fn validate_persisted_provider_verifier_configuration(
        &self,
        tx: &Transaction<'_>,
        expected: &ProviderVerifierConfiguration,
    ) -> Result<(), AuthorizationStoreError> {
        let values = [
            ("provider_evidence_verifier_relying_party_id", expected.relying_party_id.as_str()),
            ("provider_evidence_verifier_id", expected.verifier_id.as_str()),
            ("provider_evidence_verifier_revision", expected.verifier_revision.as_str()),
            ("provider_evidence_verifier_implementation_id", expected.verifier_implementation_id.as_str()),
            ("provider_evidence_verifier_implementation_digest", expected.verifier_implementation_digest.as_str()),
            ("provider_evidence_verifier_config_digest", expected.verifier_config_digest.as_str()),
            ("provider_evidence_verifier_trust_anchor_digest", expected.trust_anchor_digest.as_str()),
            ("provider_evidence_verifier_evidence_profile_digest", expected.evidence_profile_digest.as_str()),
        ];
        for (key, expected_value) in values {
            let stored: Option<String> = tx
                .query_row(
                    "SELECT value FROM authorization_store_metadata WHERE key=?1",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?;
            if stored.as_deref() != Some(expected_value) {
                return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
            }
        }
        Ok(())
    }

    /// Compare a previously resolved adapter configuration against the
    /// relying-party pin while holding the authoritative transaction.
    ///
    /// The adapter is resolved once for the caller-facing path, but the selected
    /// revision and implementation digest are revalidated here immediately before
    /// the durable DispatchPending record is created. A pin change or deletion in
    /// the lookup→commit window therefore fails closed rather than producing a
    /// record carrying stale adapter authority.
    fn validate_persisted_adapter_configuration_for_expected(
        tx: &Transaction<'_>,
        expected: &ProviderAdapterConfiguration,
    ) -> Result<(), AuthorizationStoreError> {
        let configuration = tx
            .query_row(
                "SELECT adapter_revision,implementation_digest
                 FROM authorization_provider_adapter_pins
                 WHERE adapter_id=?1",
                params![expected.adapter_id.as_str()],
                |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?)),
            )
            .optional()?
            .ok_or_else(|| AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))?;
        if configuration.0 != expected.adapter_revision
            || configuration.1 != expected.implementation_digest
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Ok(())
    }

    /// Compare every immutable identity field of a caller-supplied dispatch
    /// record against the durable dispatch row before terminal settlement.
    fn validate_persisted_adapter_configuration(
        tx: &Transaction<'_>,
        record: &DurableDispatchRecord,
    ) -> Result<(), AuthorizationStoreError> {
        let expected = ProviderAdapterConfiguration::new(
            record.adapter.clone(),
            record.adapter_revision.clone(),
            record.adapter_implementation_digest.clone(),
        );
        Self::validate_persisted_adapter_configuration_for_expected(tx, &expected)
    }

    fn validate_persisted_dispatch_record(
        &self,
        tx: &Transaction<'_>,
        record: &DurableDispatchRecord,
    ) -> Result<String, AuthorizationStoreError> {
        validate_attempt_boundary_consistency_for_attempt(
            tx,
            &record.authorization_instance,
            &record.attempt_id,
            &record.boundary_id,
        )?;
        validate_attempt_operation_consistency_for_attempt(
            tx,
            &record.authorization_instance,
            &record.attempt_id,
            &record.operation_id,
        )?;
        let row = tx.query_row(
            "SELECT operation_id,native_replay_identity,native_issuer,native_authority_namespace,
                    native_authorization_id,native_replay_derivation_digest,
                    native_authority_pin_set_id,native_authority_pin_set_digest,
                    relying_party_id,action_id,action_digest,provider_idempotency_key,
                    target_identity,audience,adapter,adapter_revision,adapter_implementation_digest,
                    boundary_id,attempt_scope_digest,attempt_binding_digest,
                    status_identifier,status_source_digest,status_observed_at,status_valid_until,
                    status_evidence_digest,validity_issued_at,validity_expires_at,validity_policy_digest,state
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
            params![record.authorization_instance, record.attempt_id, record.boundary_id],
            |r| Ok((
                r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,Option<String>>(2)?,
                r.get::<_,Option<String>>(3)?, r.get::<_,Option<String>>(4)?, r.get::<_,Option<String>>(5)?,
                r.get::<_,Option<String>>(6)?, r.get::<_,Option<String>>(7)?, r.get::<_,Option<String>>(8)?,
                r.get::<_,String>(9)?, r.get::<_,String>(10)?, r.get::<_,String>(11)?,
                r.get::<_,String>(12)?, r.get::<_,String>(13)?, r.get::<_,String>(14)?,
                r.get::<_,Option<String>>(15)?, r.get::<_,Option<String>>(16)?,
                r.get::<_,String>(17)?, r.get::<_,Option<String>>(18)?, r.get::<_,String>(19)?,
                r.get::<_,Option<String>>(20)?, r.get::<_,Option<String>>(21)?,
                r.get::<_,Option<String>>(22)?, r.get::<_,Option<String>>(23)?,
                r.get::<_,Option<String>>(24)?, r.get::<_,Option<String>>(25)?,
                r.get::<_,Option<String>>(26)?, r.get::<_,Option<String>>(27)?,
                r.get::<_,String>(28)?,
            )),
        ).optional()?.ok_or_else(|| AuthorizationStoreError::NotFound(record.attempt_id.clone()))?;

        let (
            operation_id,native_replay_identity,native_issuer,native_authority_namespace,
            native_authorization_id,native_replay_derivation_digest,
            native_authority_pin_set_id,native_authority_pin_set_digest,
            relying_party_id,action_id,action_digest,provider_idempotency_key,
            target_identity,audience,adapter,adapter_revision,adapter_implementation_digest,
            boundary_id,attempt_scope_digest,attempt_binding_digest,
            status_identifier,status_source_digest,status_observed_at,status_valid_until,
            status_evidence_digest,validity_issued_at,validity_expires_at,validity_policy_digest,state
        )=row;

        if operation_id != record.operation_id
            || native_replay_identity != record.native_replay_identity
            || native_issuer.as_deref() != Some(record.native_issuer.as_str())
            || native_authority_namespace.as_deref() != Some(record.native_authority_namespace.as_str())
            || native_authorization_id.as_deref() != Some(record.native_authorization_id.as_str())
            || native_replay_derivation_digest.as_deref() != Some(record.native_replay_derivation_digest.as_str())
            || relying_party_id.as_deref() != Some(self.relying_party_id.as_str())
            || action_id != record.action_id
            || action_digest != record.action_digest
            || provider_idempotency_key != record.provider_idempotency_key
            || target_identity != record.target_identity
            || audience != record.audience
            || adapter != record.adapter
            || adapter_revision.as_deref() != Some(record.adapter_revision.as_str())
            || adapter_implementation_digest.as_deref() != Some(record.adapter_implementation_digest.as_str())
            || boundary_id != record.boundary_id
            || attempt_scope_digest.as_deref() != Some(
                compute_attempt_scope_digest(&record.boundary_id,&record.attempt_id)?.as_str()
            )
            || attempt_binding_digest != record.attempt_binding_digest
            || status_identifier.as_deref() != Some(record.status_identifier.as_str())
            || status_source_digest.as_deref() != Some(record.status_source_digest.as_str())
            || status_observed_at.as_deref() != Some(record.status_observed_at.as_str())
            || status_valid_until.as_deref() != Some(record.status_valid_until.as_str())
            || status_evidence_digest.as_deref() != Some(record.status_evidence_digest.as_str())
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let Some(adapter_revision) = adapter_revision.as_deref() else {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        };
        let Some(adapter_implementation_digest) = adapter_implementation_digest.as_deref() else {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        };
        Self::validate_persisted_adapter_configuration(tx, record)?;

        let expected = compute_attempt_binding_digest(
            &record.authorization_instance,&record.attempt_id,&record.operation_id,
            &record.native_replay_identity,&record.native_issuer,&record.native_authority_namespace,
            &record.native_authorization_id,&record.native_replay_derivation_digest,
            native_authority_pin_set_id.as_deref().unwrap_or(""),
            native_authority_pin_set_digest.as_deref().unwrap_or(""),
            &self.relying_party_id,&record.action_id,&record.action_digest,
            &record.provider_idempotency_key,&record.target_identity,&record.audience,&record.adapter,
            adapter_revision,adapter_implementation_digest,
            &record.status_identifier,&record.status_source_digest,&record.status_observed_at,
            &record.status_valid_until,&record.status_evidence_digest,
            validity_issued_at.as_deref().unwrap_or(""),validity_expires_at.as_deref(),
            validity_policy_digest.as_deref().unwrap_or(""),
        );
        if attempt_binding_digest.is_empty() || expected != record.attempt_binding_digest
            || expected != attempt_binding_digest
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        Ok(state)
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
        // Phase 1: validate the durable binding and service an already-terminal
        // idempotent read without invoking foreign verifier code.
        {
            let mut connection = self.connection()?;
            let tx = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
            let persisted_state = self.validate_persisted_dispatch_record(&tx, record)?;
            if matches!(persisted_state.as_str(), "succeeded" | "failed") {
                if let Some(receipt) =
                    load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")?
                {
                    let pinned_verifier = self.pinned_provider_evidence_verifier_configuration()?;
                    validate_persisted_terminal_evidence(
                        self,
                        &tx,
                        record,
                        &receipt,
                        &pinned_verifier,
                    )?;
                    return Ok(receipt);
                }
                return Err(AuthorizationConsumptionError::AttemptMismatch.into());
            }
            tx.commit()?;
        }

        let pinned_verifier = self.pinned_provider_evidence_verifier_configuration()?;
        if !matches!(evidence.kind, ProviderEvidenceKind::TerminalOutcome)
            || matches!(evidence.outcome, ExecutionOutcome::Indeterminate)
        {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }
        let verified = verifier
            .verify(ProviderVerificationPurpose::TerminalOutcome, record, evidence)
            .map_err(|_| AuthorizationConsumptionError::ProviderEvidenceVerificationRequired)?;
        self.validate_verified_terminal_outcome(record, &verified)?;
        if verified.configuration != pinned_verifier {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }

        // A verifier result is never authoritative by itself: mutable pins or
        // durable attempt fields may change while foreign verification executes.
        // Only the second transaction can admit terminal state.
        // Phase 2: acquire the authoritative write transaction only after
        // verification, then revalidate every durable binding before settlement.
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let persisted_state = self.validate_persisted_dispatch_record(&tx, record)?;
        self.validate_persisted_provider_verifier_configuration(&tx, &verified.configuration)?;
        let row: (
            String, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>,
            Option<String>, Option<String>, Option<String>, Option<String>
        ) = tx.query_row(
            "SELECT state,native_issuer,native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
                    native_authority_pin_set_id,native_authority_pin_set_digest,
                    validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
            params![record.authorization_instance, record.attempt_id, record.boundary_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?,r.get(9)?)),
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
            record, row.2.as_deref(), row.3.as_deref(), row.4.as_deref()
        )?;
        self.validate_native_authority_pin_binding(
            &tx, row.1.as_deref(), row.2.as_deref()
        )?;
        self.validate_native_authority_pin_set_snapshot(
            &tx, row.5.as_deref(), row.6.as_deref()
        )?;
        if persisted_state != row.0 {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        if !matches!(row.0.as_str(), "dispatch_pending" | "invoked" | "indeterminate") {
            if let Some(receipt) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")? {
                validate_persisted_terminal_evidence(
                    self,
                    &tx,
                    record,
                    &receipt,
                    &pinned_verifier,
                )?;
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
        if lease.operation_id.as_deref() != Some(record.operation_id.as_str())
            || lease.action_id != record.action_id
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
                validate_persisted_terminal_evidence(
                    self,
                    &tx,
                    record,
                    &receipt,
                    &pinned_verifier,
                )?;
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
              native_issuer,native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
              native_authority_pin_set_id,native_authority_pin_set_digest,
              validity_issued_at,validity_expires_at,validity_policy_digest,relying_party_id,boundary_id,attempt_scope_digest,
               action_digest,provider_idempotency_key,target_identity,audience,adapter,
              adapter_revision,adapter_implementation_digest,outcome,evidence_id,
              evidence_digest,attempt_binding_digest,verifier_id,verifier_revision,verifier_implementation_id,
              verifier_implementation_digest,verifier_config_digest,trust_anchor_digest,evidence_profile_digest,verification_digest)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27,?28,?29,?30,?31,?32,?33,?34,?35)",
            params![
                record.authorization_instance, record.attempt_id, record.operation_id,
                record.native_replay_identity, row.1, row.2, row.3, row.4, row.5, row.6,
                row.7, row.8, row.9,
                self.relying_party_id.as_str(), record.boundary_id,
                 compute_attempt_scope_digest(&record.boundary_id,&record.attempt_id)?,
                 record.action_digest,
                record.provider_idempotency_key, record.target_identity, record.audience, record.adapter,
                record.adapter_revision, record.adapter_implementation_digest,
                if matches!(evidence.outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                evidence.evidence_id, evidence.evidence_digest, record.attempt_binding_digest,
                verified.configuration.verifier_id,
                verified.configuration.verifier_revision,
                verified.configuration.verifier_implementation_id,
                verified.configuration.verifier_implementation_digest,
                verified.configuration.verifier_config_digest, verified.configuration.trust_anchor_digest,
                verified.configuration.evidence_profile_digest, verified.verification_digest,
            ],
        )?;
        insert_receipt_with_boundary(&tx, &receipt, "final", Some(&record.boundary_id))?;
        let changed = tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?4
               AND state IN ('dispatch_pending','invoked','indeterminate')",
            params![record.authorization_instance, record.attempt_id,
                if matches!(evidence.outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                record.boundary_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
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
        if lease.operation_id.as_deref() != Some(record.operation_id.as_str())
            || lease.action_id != record.action_id || lease.action_digest != record.action_digest
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
        let changed = tx.execute(
            "UPDATE authorization_dispatches SET state=?3
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?4
               AND state IN ('dispatch_pending','invoked','indeterminate')",
            params![record.authorization_instance, record.attempt_id, dispatch_state, record.boundary_id],
        )?;
        if changed != 1 {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }
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
        self.validate_recovery_authorization_issued_at(&witness.issued_at)?;

        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let lease = load_lease(&tx, &witness.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(witness.authorization_instance.clone()))?;

        validate_attempt_boundary_consistency_for_attempt(
            &tx,
            &witness.authorization_instance,
            &witness.attempt_id,
            &witness.boundary_id,
        )?;

        if let Some((marker_operation_id, boundary_id, action_digest, authority_epoch, stored_scope)) = tx
            .query_row(
                "SELECT operation_id,boundary_id,action_digest,authority_epoch,attempt_scope_digest
                 FROM authorization_recovery_markers
                 WHERE authorization_instance=?1 AND attempt_id=?2 AND marker='not_entered'",
                params![witness.authorization_instance.as_str(), witness.attempt_id.as_str()],
                |row| Ok((
                    row.get::<_,Option<String>>(0)?,
                    row.get::<_,String>(1)?,
                    row.get::<_,String>(2)?,
                    row.get::<_,i64>(3)?,
                    row.get::<_,Option<String>>(4)?,
                )),
            )
            .optional()?
        {
            let expected_scope = compute_attempt_scope_digest(
                &witness.boundary_id,
                &witness.attempt_id,
            )?;
            let stored_operation_id = marker_operation_id.as_deref().unwrap_or("");
            if lease.authorization_instance == witness.authorization_instance
                && lease.action_digest == witness.action_digest
                && !witness.policy.is_empty()
                && lease.authority_epoch == witness.authority_epoch
                && match lease.operation_id.as_deref() {
                    Some(operation_id) => {
                        !witness.operation_id.is_empty() && operation_id == witness.operation_id
                    }
                    None => witness.operation_id.is_empty(),
                }
                && stored_operation_id == witness.operation_id
                && matches!(lease.state, AuthorizationLeaseState::Ready)
                && boundary_id == witness.boundary_id
                && action_digest == witness.action_digest
                && authority_epoch >= 0
                && authority_epoch as u64 == witness.authority_epoch
                && stored_scope.as_deref() == Some(expected_scope.as_str())
            {
                return Ok(false);
            }
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        if !witness.is_bound_to(&lease) {
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
             SET state='ready', attempt_id=NULL, boundary_id=NULL, attempt_scope_digest=NULL
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
             (authorization_instance,attempt_id,operation_id,boundary_id,action_digest,authority_epoch,marker,attempt_scope_digest)
             VALUES (?1,?2,NULLIF(?3,''),?4,?5,?6,'not_entered',?7)",
            params![
                witness.authorization_instance.as_str(),
                witness.attempt_id.as_str(),
                witness.operation_id.as_str(),
                witness.boundary_id.as_str(),
                witness.action_digest.as_str(),
                witness.authority_epoch as i64,
                compute_attempt_scope_digest(&witness.boundary_id,&witness.attempt_id)?,
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
        tx: &Transaction<'_>,
        record: &DurableDispatchRecord,
        outcome: ExecutionOutcome,
        verified: &VerifiedProviderOutcome,
    ) -> Result<ExecutionReceipt, AuthorizationStoreError> {
        if matches!(outcome, ExecutionOutcome::Indeterminate) {
            return Err(AuthorizationConsumptionError::AttemptMismatch.into());
        }

        let persisted_state = self.validate_persisted_dispatch_record(tx, record)?;
        self.validate_persisted_provider_verifier_configuration(tx, &verified.configuration)?;

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
            Option<String>, Option<String>, Option<String>, Option<String>
        ) = tx.query_row(
            "SELECT native_issuer,native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
                    native_authority_pin_set_id,native_authority_pin_set_digest,
                    validity_issued_at,validity_expires_at,validity_policy_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance, record.attempt_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?)),
        )?;
        Self::validate_persisted_native_replay_provenance(
            record,
            native_provenance.1.as_deref(),
            native_provenance.2.as_deref(),
            native_provenance.3.as_deref(),
        )?;
        self.validate_native_authority_pin_binding(
            &tx,
            native_provenance.0.as_deref(),
            native_provenance.1.as_deref(),
        )?;
        self.validate_native_authority_pin_set_snapshot(
            &tx,
            native_provenance.4.as_deref(),
            native_provenance.5.as_deref(),
        )?;
        self.validate_persisted_authorization_validity(
            native_provenance.6.as_deref(),
            native_provenance.7.as_deref(),
            native_provenance.8.as_deref(),
            false,
        )?;

        if let Some(r) = load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "reconciled")? {
            let pinned_verifier = self.pinned_provider_evidence_verifier_configuration()?;
            validate_persisted_terminal_evidence(
                self,
                &tx,
                record,
                &r,
                &pinned_verifier,
            )?;
            return Ok(r);
        }

        let current_boundary = load_lease_boundary(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if current_boundary != record.boundary_id {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let mut lease = load_lease(&tx, &record.authorization_instance)?
            .ok_or_else(|| AuthorizationStoreError::NotFound(record.authorization_instance.clone()))?;
        if lease.operation_id.as_deref() != Some(record.operation_id.as_str())
            || lease.action_id != record.action_id || lease.action_digest != record.action_digest
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
              native_issuer,native_authority_namespace,native_authorization_id,native_replay_derivation_digest,
              native_authority_pin_set_id,native_authority_pin_set_digest,
              validity_issued_at,validity_expires_at,validity_policy_digest,relying_party_id,boundary_id,attempt_scope_digest,
              action_digest,provider_idempotency_key,target_identity,audience,adapter,
              adapter_revision,adapter_implementation_digest,outcome,evidence_id,
              evidence_digest,attempt_binding_digest,verifier_id,verifier_revision,verifier_implementation_id,
              verifier_implementation_digest,verifier_config_digest,trust_anchor_digest,evidence_profile_digest,verification_digest)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27,?28,?29,?30,?31,?32,?33,?34,?35)",
            params![
                record.authorization_instance, record.attempt_id, record.operation_id,
                record.native_replay_identity, native_provenance.0, native_provenance.1,
                native_provenance.2, native_provenance.3, native_provenance.4, native_provenance.5,
                native_provenance.6, native_provenance.7, native_provenance.8,
                self.relying_party_id.as_str(), record.boundary_id,
                compute_attempt_scope_digest(&record.boundary_id,&record.attempt_id)?,
                record.action_digest, record.provider_idempotency_key,
                record.target_identity, record.audience, verified.evidence.adapter,
                record.adapter_revision, record.adapter_implementation_digest,
                if matches!(outcome, ExecutionOutcome::Succeeded) { "succeeded" } else { "failed" },
                verified.evidence.evidence_id, verified.evidence.evidence_digest,
                record.attempt_binding_digest,
                verified.configuration.verifier_id,
                verified.configuration.verifier_revision,
                verified.configuration.verifier_implementation_id,
                verified.configuration.verifier_implementation_digest,
                verified.configuration.verifier_config_digest,
                verified.configuration.trust_anchor_digest,
                verified.configuration.evidence_profile_digest,
                verified.verification_digest,
            ],
        )?;     Ok(receipt)
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
        let pinned_verifier = self.pinned_provider_evidence_verifier_configuration()?;

        // Phase 1: validate the frozen dispatch contract without retaining a
        // write lock across the external provider verification call. A durable
        // terminal receipt is an established historical result, so replay reads
        // its bound evidence without re-running foreign verifier code.
        {
            let mut connection = self.connection()?;
            let tx = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
            let persisted_state = self.validate_persisted_dispatch_record(&tx, record)?;
            if matches!(persisted_state.as_str(), "succeeded" | "failed") {
                if let Some(receipt) =
                    load_receipt(&tx, &record.authorization_instance, &record.attempt_id, "final")?
                {
                    validate_persisted_terminal_evidence(
                        self,
                        &tx,
                        record,
                        &receipt,
                        &pinned_verifier,
                    )?;
                    return Ok(receipt);
                }
                return Err(AuthorizationConsumptionError::AttemptMismatch.into());
            }
            if persisted_state != "indeterminate" {
                return Err(AuthorizationConsumptionError::AttemptMismatch.into());
            }
            tx.commit()?;
        }

        if !matches!(evidence.kind, ProviderEvidenceKind::TerminalOutcome)
            || matches!(evidence.outcome, ExecutionOutcome::Indeterminate)
        {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }
        let verified = verifier
            .verify(ProviderVerificationPurpose::TerminalOutcome, record, evidence)
            .map_err(|_| AuthorizationConsumptionError::ProviderEvidenceVerificationRequired)?;
        self.validate_verified_terminal_outcome(record, &verified)?;
        if verified.configuration != pinned_verifier {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }

        // Phase 2: the verifier result is advisory until the authoritative
        // transaction rechecks the durable record and relying-party verifier pin.
        let mut connection = self.connection()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        self.validate_persisted_dispatch_record(&tx, record)?;
        self.validate_persisted_provider_verifier_configuration(&tx, &verified.configuration)?;
        if !matches!(verified.evidence.outcome, ExecutionOutcome::Succeeded | ExecutionOutcome::Failed) {
            return Err(AuthorizationConsumptionError::ProviderEvidenceVerificationRequired.into());
        }

        let receipt = self.reconcile_indeterminate_bound_verified_inner(
            &tx, record, verified.evidence.outcome, &verified
        )?;
        tx.commit()?;
        Ok(receipt)
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
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id, state
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
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id, state
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id=?1",
                    )?,
                    vec![boundary_filter.unwrap().to_owned()],
                ),
                (None, Some(_)) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id, state
                         FROM authorization_leases
                         WHERE state IN ('dispatch_pending','invoked')
                           AND boundary_id IS NULL AND attempt_id=?1",
                    )?,
                    vec![attempt_filter.unwrap().to_owned()],
                ),
                (None, None) => (
                    tx.prepare(
                        "SELECT authorization_instance, action_id, attempt_id, action_digest, authority_epoch, boundary_id, state
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
                    row.get::<_, String>(6)?,
                ))
            })?;
            for row in rows {
                recovered.push(row?);
            }
            drop(stmt);
        }

        for (instance, action_id, attempt_id, action_digest, authority_epoch, lease_boundary, lease_state) in &recovered {
            let persisted_bound_record = if let Some(boundary_id) = lease_boundary.as_deref() {
                match dispatch_boundary(&tx, instance, attempt_id)? {
                    Some(dispatch_boundary_id) if dispatch_boundary_id == boundary_id => {}
                    Some(_) => return Err(AuthorizationConsumptionError::InvalidBinding.into()),
                    None => return Err(AuthorizationConsumptionError::InvalidBinding.into()),
                }
                let (record, dispatch_state) =
                    load_persisted_dispatch_record(&tx, instance, attempt_id, boundary_id)?;
                if dispatch_state.as_str() != lease_state.as_str()
                    || dispatch_state != "dispatch_pending" && dispatch_state != "invoked"
                {
                    return Err(AuthorizationConsumptionError::InvalidBinding.into());
                }
                self.validate_persisted_dispatch_record(&tx, &record)?;
                Some(record)
            } else {
                None
            };
            let changed = tx.execute(
                "UPDATE authorization_leases
                 SET state='indeterminate', attempt_id=?2
                 WHERE authorization_instance=?1
                   AND state IN ('prepared','dispatch_pending','invoked')
                   AND attempt_id=?2",
                params![instance, attempt_id],
            )?;
            if changed != 1 {
                return Err(AuthorizationConsumptionError::AttemptMismatch.into());
            }
            let recovered_provider_key: String = if let Some(record) = persisted_bound_record.as_ref() {
                record.provider_idempotency_key.clone()
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
                operation_id: persisted_bound_record.as_ref().map(|record| record.operation_id.clone()),
                action_digest: action_digest.clone(),
                provider_idempotency_key: recovered_provider_key,
                attempt_id: attempt_id.clone(),
                authority_epoch: *authority_epoch,
                outcome: ExecutionOutcome::Indeterminate,
            };
            let dispatch_changed = tx.execute(
                "UPDATE authorization_dispatches SET state='indeterminate'
                 WHERE authorization_instance=?1 AND attempt_id=?2
                   AND state IN ('dispatch_pending','invoked')
                   AND (?3 IS NULL OR boundary_id=?3)",
                params![instance, attempt_id, lease_boundary],
            )?;
            if lease_boundary.is_some() && dispatch_changed != 1 {
                return Err(AuthorizationConsumptionError::AttemptMismatch.into());
            }
            tx.execute(
                "INSERT OR IGNORE INTO authorization_receipts
                 (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,
                  boundary_id,attempt_scope_digest)
                 VALUES (?1,?2,?3,'indeterminate','indeterminate',?4,?5,?6,?7)",
                params![
                    receipt.authorization_instance,
                    receipt.action_id,
                    receipt.attempt_id,
                    receipt.action_digest,
                    receipt.authority_epoch as i64,
                    lease_boundary,
                    lease_boundary.as_ref().map(|boundary|
                        compute_attempt_scope_digest(boundary.as_str(),&receipt.attempt_id)
                    ).transpose()?,
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
           operation_id TEXT,
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
          authority_epoch,remaining_executions,state,attempt_id,operation_id,boundary_id)
         SELECT action_id,action_id,action_digest,support_digest,policy,
                authority_epoch,remaining_executions,state,attempt_id,NULL,NULL
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
                remaining_executions,state,attempt_id,operation_id FROM authorization_leases
         WHERE authorization_instance=?1",
        params![id],
        |r| {
            let state: String = r.get(7)?;
            let attempt: Option<String> = r.get(8)?;
            let operation_id: Option<String> = r.get(9)?;
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
                operation_id,
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
    let attempt_scope_digest = match (current_boundary, state_attempt(&lease.state)) {
        (Some(boundary), Some(attempt)) => Some(compute_attempt_scope_digest(boundary, attempt)?),
        _ => None,
    };
    let changed = tx.execute(
        "UPDATE authorization_leases
         SET state=?2,attempt_id=?3,operation_id=?4,remaining_executions=?5,boundary_id=?6,
             attempt_scope_digest=?7
         WHERE authorization_instance=?1",
        params![
            lease.authorization_instance,
            encode_state(&lease.state),
            state_attempt(&lease.state),
            if current_boundary.is_some() { lease.operation_id.as_deref() } else { None },
            lease.remaining_executions as i64,
            current_boundary,
            attempt_scope_digest
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
    let attempt_scope_digest = match boundary_id {
        Some(boundary) => Some(compute_attempt_scope_digest(boundary, &r.attempt_id)?),
        None => None,
    };
    let operation_id = boundary_id.map(|_| {
        r.operation_id
            .as_deref()
            .filter(|value| !value.is_empty())
    });
    if boundary_id.is_some() && operation_id.is_none() {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }
    tx.execute(
        "INSERT INTO authorization_receipts
         (authorization_instance,action_id,attempt_id,phase,outcome,action_digest,authority_epoch,
          operation_id,provider_idempotency_key,boundary_id,attempt_scope_digest)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
        params![
            r.authorization_instance, r.action_id, r.attempt_id, phase, outcome,
            r.action_digest, r.authority_epoch as i64, operation_id, r.provider_idempotency_key,
            boundary_id, attempt_scope_digest
        ],
    )?;
    Ok(())
}
fn load_receipt(
    tx: &Transaction<'_>, authorization_instance: &str, attempt_id: &str, phase: &str,
) -> Result<Option<ExecutionReceipt>, AuthorizationStoreError> {
    let row: Option<(
        String, String, String, String, String, i64,
        Option<String>, Option<String>, Option<String>, Option<String>,
    )> = tx.query_row(
        "SELECT authorization_instance,action_id,attempt_id,outcome,action_digest,authority_epoch,
                operation_id,provider_idempotency_key,boundary_id,attempt_scope_digest
         FROM authorization_receipts
         WHERE authorization_instance=?1 AND attempt_id=?2 AND phase=?3",
        params![authorization_instance,attempt_id,phase],
        |r| Ok((
            r.get(0)?,
            r.get(1)?,
            r.get(2)?,
            r.get(3)?,
            r.get(4)?,
            r.get(5)?,
            r.get(6)?,
            r.get(7)?,
            r.get(8)?,
            r.get(9)?,
        )),
    ).optional()?;

    let Some((
        authorization_instance,
        action_id,
        attempt_id,
        outcome,
        action_digest,
        authority_epoch,
        operation_id,
        provider_idempotency_key,
        boundary_id,
        attempt_scope_digest,
    )) = row else {
        return Ok(None);
    };

    let mut operation_id = None;

    let outcome = match outcome.as_str() {
        "succeeded" => ExecutionOutcome::Succeeded,
        "failed" => ExecutionOutcome::Failed,
        "indeterminate" => ExecutionOutcome::Indeterminate,
        _ => return Err(AuthorizationStoreError::InvalidState(
            "invalid persisted authorization receipt outcome".into()
        )),
    };

    let persisted_authority_epoch: i64 = tx
        .query_row(
            "SELECT authority_epoch
             FROM authorization_leases
             WHERE authorization_instance=?1",
            params![authorization_instance.as_str()],
            |r| r.get(0),
        )
        .optional()?
        .ok_or_else(|| AuthorizationConsumptionError::InvalidBinding)?;
    if persisted_authority_epoch < 0
        || persisted_authority_epoch as u64 != authority_epoch as u64
    {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }

    if let Some(boundary) = boundary_id.as_deref() {
        let expected_scope=compute_attempt_scope_digest(boundary,&attempt_id)?;
        if attempt_scope_digest.as_deref() != Some(expected_scope.as_str()) {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }

        let dispatch: Option<(String,String,String,String,String,Option<String>)> = tx.query_row(
            "SELECT operation_id,action_id,action_digest,provider_idempotency_key,boundary_id,attempt_scope_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![authorization_instance,attempt_id],
            |r| Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
            )),
        ).optional()?;

        let Some((
            dispatch_operation_id,
            dispatch_action_id,
            dispatch_action_digest,
            dispatch_provider_key,
            dispatch_boundary,
            dispatch_scope,
        )) = dispatch else {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        };

        if operation_id.as_deref() != Some(dispatch_operation_id.as_str())
            || dispatch_action_id != action_id
            || dispatch_action_digest != action_digest
            || dispatch_provider_key != provider_idempotency_key.as_deref().unwrap_or("")
            || dispatch_boundary != boundary
            || dispatch_scope.as_deref() != Some(expected_scope.as_str())
        {
            return Err(AuthorizationConsumptionError::InvalidBinding.into());
        }
        operation_id = Some(dispatch_operation_id);
    } else if attempt_scope_digest.as_ref().is_some_and(|scope| !scope.is_empty()) {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }

    let provider_idempotency_key = provider_idempotency_key
        .filter(|key| !key.is_empty())
        .unwrap_or_else(|| {
            AuthorizationLease::new_with_instance(
                authorization_instance.clone(),
                action_id.clone(),
                action_digest.clone(),
                String::new(),
                String::new(),
                authority_epoch as u64,
                1,
            ).provider_idempotency_key()
        });

    Ok(Some(ExecutionReceipt {
        authorization_instance,
        action_id,
        operation_id,
        attempt_id,
        outcome,
        action_digest,
        provider_idempotency_key,
        authority_epoch: authority_epoch as u64,
    }))
}


fn validate_persisted_terminal_evidence(
    store: &SqliteAuthorizationStore,
    tx: &Transaction<'_>,
    record: &DurableDispatchRecord,
    receipt: &ExecutionReceipt,
    pinned_verifier: &ProviderVerifierConfiguration,
) -> Result<(), AuthorizationStoreError> {
    let pin_set: (Option<String>, Option<String>) = tx.query_row(
        "SELECT native_authority_pin_set_id,native_authority_pin_set_digest
         FROM authorization_dispatches
         WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
        params![record.authorization_instance, record.attempt_id, record.boundary_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let dispatch_validity: (Option<String>, Option<String>, Option<String>) = tx.query_row(
        "SELECT validity_issued_at,validity_expires_at,validity_policy_digest
         FROM authorization_dispatches
         WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
        params![record.authorization_instance, record.attempt_id, record.boundary_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;

    let row: Option<(
        String, String, Option<String>, Option<String>, Option<String>,
        Option<String>, Option<String>, Option<String>, Option<String>,
        String, Option<String>, String, String, String, String,
        String, String, String, String, String, String, String,
        String, String, String, String, String, String, String, String,
    )> = tx.query_row(
        "SELECT operation_id,native_replay_identity,native_issuer,native_authority_namespace,
                native_authorization_id,native_replay_derivation_digest,
                native_authority_pin_set_id,native_authority_pin_set_digest,relying_party_id,
                boundary_id,attempt_scope_digest,action_digest,provider_idempotency_key,
                target_identity,audience,adapter,adapter_revision,
                adapter_implementation_digest,outcome,evidence_id,evidence_digest,
                attempt_binding_digest,verifier_id,verifier_revision,verifier_implementation_id,
                verifier_implementation_digest,verifier_config_digest,trust_anchor_digest,
                evidence_profile_digest,verification_digest
         FROM authorization_terminal_evidence
         WHERE authorization_instance=?1 AND attempt_id=?2",
        params![record.authorization_instance.as_str(), record.attempt_id.as_str()],
        |r| Ok((
            r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?,
            r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?,
            r.get(10)?, r.get(11)?, r.get(12)?, r.get(13)?, r.get(14)?,
            r.get(15)?, r.get(16)?, r.get(17)?, r.get(18)?, r.get(19)?,
            r.get(20)?, r.get(21)?, r.get(22)?, r.get(23)?, r.get(24)?,
            r.get(25)?, r.get(26)?, r.get(27)?, r.get(28)?, r.get(29)?,
        )),
    ).optional()?;

    let terminal_validity: (Option<String>, Option<String>, Option<String>) = tx.query_row(
        "SELECT validity_issued_at,validity_expires_at,validity_policy_digest
         FROM authorization_terminal_evidence
         WHERE authorization_instance=?1 AND attempt_id=?2",
        params![record.authorization_instance.as_str(), record.attempt_id.as_str()],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;

    let Some((
        operation_id,native_replay_identity,native_issuer,native_authority_namespace,
        native_authorization_id,native_replay_derivation_digest,
        native_authority_pin_set_id,native_authority_pin_set_digest,relying_party_id,
        boundary_id,attempt_scope_digest,action_digest,provider_idempotency_key,
        target_identity,audience,adapter,adapter_revision,adapter_implementation_digest,
        outcome,evidence_id,evidence_digest,attempt_binding_digest,
        verifier_id,verifier_revision,verifier_implementation_id,
        verifier_implementation_digest,verifier_config_digest,trust_anchor_digest,
        evidence_profile_digest,verification_digest,
    )) = row else {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    };

    let expected_scope =
        compute_attempt_scope_digest(&record.boundary_id, &record.attempt_id)?;
    let expected_outcome = match receipt.outcome {
        ExecutionOutcome::Succeeded => "succeeded",
        ExecutionOutcome::Failed => "failed",
        ExecutionOutcome::Indeterminate => "indeterminate",
    };

    store.validate_native_authority_pin_set_snapshot(
        tx,
        pin_set.0.as_deref(),
        pin_set.1.as_deref(),
    )?;

    if operation_id != record.operation_id
        || native_replay_identity != record.native_replay_identity
        || native_issuer.as_deref() != Some(record.native_issuer.as_str())
        || native_authority_namespace.as_deref() != Some(record.native_authority_namespace.as_str())
        || native_authorization_id.as_deref() != Some(record.native_authorization_id.as_str())
        || native_replay_derivation_digest.as_deref() != Some(record.native_replay_derivation_digest.as_str())
        || native_authority_pin_set_id != pin_set.0
        || native_authority_pin_set_digest != pin_set.1
        || terminal_validity != dispatch_validity
        || relying_party_id.as_deref() != Some(store.relying_party_id.as_str())
        || boundary_id != record.boundary_id
        || attempt_scope_digest.as_deref() != Some(expected_scope.as_str())
        || action_digest != record.action_digest
        || provider_idempotency_key != record.provider_idempotency_key
        || target_identity != record.target_identity
        || audience != record.audience
        || adapter.as_str() != record.adapter.as_str()
        || adapter_revision.as_str() != record.adapter_revision.as_str()
        || adapter_implementation_digest.as_str() != record.adapter_implementation_digest.as_str()
        || outcome != expected_outcome
        || evidence_id.is_empty()
        || evidence_digest.is_empty()
        || attempt_binding_digest != record.attempt_binding_digest
        || verifier_id != pinned_verifier.verifier_id
        || verifier_revision.as_str() != pinned_verifier.verifier_revision.as_str()
        || verifier_implementation_id.as_str()
            != pinned_verifier.verifier_implementation_id.as_str()
        || verifier_implementation_digest.as_str()
            != pinned_verifier.verifier_implementation_digest.as_str()
        || verifier_config_digest != pinned_verifier.verifier_config_digest
        || trust_anchor_digest != pinned_verifier.trust_anchor_digest
        || evidence_profile_digest != pinned_verifier.evidence_profile_digest
        || verification_digest.is_empty()
    {
        return Err(AuthorizationConsumptionError::InvalidBinding.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelix::gis::{ActionEffectBinding, NativeReplayDerivation};
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        thread,
    };

    fn fixture(path: &Path) -> (SqliteAuthorizationStore, EpistemicAction, ActionAuthorizationWitness) {
        let fixed_now = "2026-10-03T12:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let store = SqliteAuthorizationStore::open_with_relying_party_clock_and_policy(
            path,
            "legacy-local",
            Arc::new(FixedClock {
                now: fixed_now,
                source_id: AuthorizationClockPolicy::CLOCK_SOURCE_ID,
            }),
            AuthorizationClockPolicy::default(),
        ).unwrap();
        let action = EpistemicAction::new("durable-action","intervention",super::super::ActionRisk::Critical);
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(), authorization_instance:action.id.clone(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:00:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new(action.id.clone(),digest,"sha256:support","policy-v1",1,1)).unwrap();
        (store,action,witness)
    }

    struct FixedClock {
        now: DateTime<Utc>,
        source_id: &'static str,
    }

    impl TrustedAuthorizationClock for FixedClock {
        fn source_id(&self) -> &str {
            self.source_id
        }

        fn now_utc(&self) -> Result<DateTime<Utc>, AuthorizationStoreError> {
            Ok(self.now)
        }
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
                    verifier_revision: "test-verifier/rev1".into(),
                    verifier_implementation_id: "test-verifier".into(),
                    verifier_implementation_digest: "sha256:test-verifier-implementation".into(),
                    verifier_config_digest: "sha256:test-verifier-config".into(),
                    trust_anchor_digest: "sha256:test-trust-anchors".into(),
                    evidence_profile_digest: "sha256:test-evidence-profile".into(),
                },
                verification_digest: "sha256:test-verification".into(),
            })
        }
    }

    struct CountingProviderVerifier {
        calls: Arc<AtomicUsize>,
    }

    impl ProviderEvidenceVerifier for CountingProviderVerifier {
        fn verify(
            &self,
            _purpose: ProviderVerificationPurpose,
            record: &DurableDispatchRecord,
            evidence: &ProviderTerminalEvidence,
        ) -> Result<VerifiedProviderOutcome, ProviderVerificationError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(VerifiedProviderOutcome {
                evidence: evidence.clone(),
                configuration: ProviderVerifierConfiguration {
                    relying_party_id: "legacy-local".into(),
                    verifier_id: "test-verifier/v1".into(),
                    verifier_revision: "test-verifier-rev/1".into(),
                    verifier_implementation_id: "test-verifier-impl".into(),
                    verifier_implementation_digest: "sha256:test-verifier-implementation".into(),
                    verifier_config_digest: "sha256:test-verifier-config".into(),
                    trust_anchor_digest: "sha256:test-trust-anchors".into(),
                    evidence_profile_digest: "sha256:test-evidence-profile".into(),
                },
                verification_digest: format!("sha256:counted:{}", record.attempt_id),
            })
        }
    }

    struct TestProviderVerifierForRp {
        relying_party_id: String,
    }

    impl ProviderEvidenceVerifier for TestProviderVerifierForRp {
        fn verify(
            &self,
            purpose: ProviderVerificationPurpose,
            record: &DurableDispatchRecord,
            evidence: &ProviderTerminalEvidence,
        ) -> Result<VerifiedProviderOutcome, ProviderVerificationError> {
            if !matches!(purpose, ProviderVerificationPurpose::TerminalOutcome)
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
                    relying_party_id: self.relying_party_id.clone(),
                    verifier_id: "test-verifier/v1".into(),
                    verifier_revision: "test-verifier/rev1".into(),
                    verifier_implementation_id: "test-verifier".into(),
                    verifier_implementation_digest: "sha256:test-verifier-implementation".into(),
                    verifier_config_digest: "sha256:test-verifier-config".into(),
                    trust_anchor_digest: "sha256:test-trust-anchors".into(),
                    evidence_profile_digest: "sha256:test-evidence-profile".into(),
                },
                verification_digest: "sha256:test-verification".into(),
            })
        }
    }

    struct AdmissionStatusFailsVerifier;

    impl ProviderStatusVerifier for AdmissionStatusFailsVerifier {
        fn verify_current_status(
            &self,
            purpose: ProviderStatusVerificationPurpose,
            _issuer: &str,
            _authority_namespace: &str,
            _native_authorization_id: &str,
            _status_identifier: &str,
            _action_digest: &str,
            _target_identity: &str,
            _audience: &str,
            _adapter: &str,
            _expected_source_digest: Option<&str>,
        ) -> Result<ProviderStatusEvidence, ProviderStatusVerificationError> {
            if matches!(purpose, ProviderStatusVerificationPurpose::Admission) {
                Err(ProviderStatusVerificationError::Unavailable)
            } else {
                Err(ProviderStatusVerificationError::VerificationFailed)
            }
        }
    }

    struct TestProviderStatusVerifier;

    impl ProviderStatusVerifier for TestProviderStatusVerifier {
        fn configuration(&self) -> ProviderStatusVerifierConfiguration {
            ProviderStatusVerifierConfiguration::new(
                "test-status-verifier/v1",
                "test-status-verifier/rev1",
                "test-status-verifier",
                "sha256:test-status-verifier-implementation",
                "sha256:test-status-verifier-config",
            )
        }

        fn verify_current_status(
            &self,
            purpose: ProviderStatusVerificationPurpose,
            _issuer: &str,
            _authority_namespace: &str,
            native_authorization_id: &str,
            status_identifier: &str,
            _action_digest: &str,
            _target_identity: &str,
            _audience: &str,
            _adapter: &str,
            expected_source_digest: Option<&str>,
        ) -> Result<ProviderStatusEvidence, ProviderStatusVerificationError> {
            let expected_native_authorization_id =
                status_identifier.strip_prefix("status:");
            if !matches!(
                purpose,
                ProviderStatusVerificationPurpose::Admission
                    | ProviderStatusVerificationPurpose::PreEntry
            ) || status_identifier.is_empty()
                || expected_native_authorization_id != Some(native_authorization_id)
            {
                return Err(ProviderStatusVerificationError::InvalidBinding);
            }

            const SOURCE: &str = "sha256:test-status-source";
            if expected_source_digest.is_some_and(|value| value != SOURCE) {
                return Err(ProviderStatusVerificationError::InvalidBinding);
            }

            let observed = trusted_utc_now()
                .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
            let valid_until = observed + Duration::seconds(30 * 60);
            Ok(ProviderStatusEvidence {
                status_identifier: status_identifier.to_owned(),
                status_source_digest: SOURCE.to_owned(),
                status_observed_at: observed.to_rfc3339_opts(SecondsFormat::Secs, true),
                status_valid_until: valid_until.to_rfc3339_opts(SecondsFormat::Secs, true),
                status_evidence_digest: "sha256:test-status-evidence".into(),
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
            adapter_revision: record.adapter_revision.clone(),
            adapter_implementation_digest: record.adapter_implementation_digest.clone(),
            boundary_id: record.boundary_id.clone(),
            attempt_binding_digest: record.attempt_binding_digest.clone(),
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
        store.pin_provider_status_source_digest("sha256:test-status-source")?;
        store.pin_provider_status_verifier_configuration(
            &ProviderStatusVerifierConfiguration::new(
                "test-status-verifier/v1",
                "test-status-verifier/rev1",
                "test-status-verifier",
                "sha256:test-status-verifier-implementation",
                "sha256:test-status-verifier-config",
            )
        )?;
        store.pin_provider_evidence_verifier_configuration(
            &ProviderVerifierConfiguration {
                relying_party_id: store.relying_party_id().to_owned(),
                verifier_id: "test-verifier/v1".into(),
                verifier_revision: "test-verifier/rev1".into(),
                verifier_implementation_id: "test-verifier".into(),
                verifier_implementation_digest: "sha256:test-verifier-implementation".into(),
                verifier_config_digest: "sha256:test-verifier-config".into(),
                trust_anchor_digest: "sha256:test-trust-anchors".into(),
                evidence_profile_digest: "sha256:test-evidence-profile".into(),
            }
        )?;
        store.pin_provider_adapter_configuration(
            &ProviderAdapterConfiguration::new(
                expected_effect.adapter.clone(),
                "test-adapter/v1",
                "sha256:test-adapter-implementation",
            )
        )?;
        const NAMESPACE: &str = "test-authority/v1";
        store.pin_native_authority_namespace(ISSUER, NAMESPACE)?;
        let status_identifier = format!("status:{}", native_authorization_id.as_ref());
        store.mark_dispatch_pending_bound_from_pinned_native_authority(
            authorization_instance,
            attempt_id,
            action,
            expected_effect,
            boundary_id,
            operation_id.as_ref(),
            ISSUER,
            native_authorization_id.as_ref(),
            &status_identifier,
            &TestProviderStatusVerifier,
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
        store.pin_provider_adapter_configuration(
            &ProviderAdapterConfiguration::new(
                expected_effect.adapter.clone(),
                "test-adapter/v1",
                "sha256:test-adapter-implementation",
            )
        )?;
        store.pin_native_authority_namespace(ISSUER, authority_namespace)?;
        let status_identifier = format!("status:{}", native_authorization_id);
        store.mark_dispatch_pending_bound_from_pinned_native_authority(
            authorization_instance,
            attempt_id,
            action,
            expected_effect,
            boundary_id,
            operation_id,
            ISSUER,
            native_authorization_id,
            &status_identifier,
            &TestProviderStatusVerifier,
        )
    }

    #[test]
    fn malformed_effect_binding_is_rejected_before_status_lookup() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-malformed-effect-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=ActionEffectBinding::new("","prod","adapter-A");
        let action=EpistemicAction::new(
            "malformed-effect","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"malformed-effect".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T07:00:00Z".into(),
            expires_at:Some("2026-10-04T07:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-malformed-effect","boundary-malformed"
        ).unwrap();

        let err=store.mark_dispatch_pending_bound_from_pinned_native_authority(
            &witness.authorization_instance,
            "attempt-malformed-effect",
            &action,
            &effect,
            "boundary-malformed",
            "operation-malformed-effect",
            "untrusted-issuer",
            "native-malformed",
            "status:native-malformed",
            &AdmissionStatusFailsVerifier,
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        assert!(store.connection().unwrap().query_row::<i64,_,_>(
            "SELECT COUNT(*) FROM authorization_dispatches",
            [],
            |row| row.get(0)
        ).unwrap() == 0);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn unbound_prepare_rejects_effect_bound_actions() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-effect-prepare-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new(
            "effect-bound-prepare","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
    fn terminal_verifier_is_not_called_for_unvalidated_dispatch_record() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-verifier-order-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"legacy-local").unwrap();
        store.pin_provider_evidence_verifier_configuration(&ProviderVerifierConfiguration {
            relying_party_id:"legacy-local".into(),
            verifier_id:"test-verifier/v1".into(),
            verifier_revision:"test-verifier-rev/1".into(),
            verifier_implementation_id:"test-verifier-impl".into(),
            verifier_implementation_digest:"sha256:test-verifier-implementation".into(),
            verifier_config_digest:"sha256:test-verifier-config".into(),
            trust_anchor_digest:"sha256:test-trust-anchors".into(),
            evidence_profile_digest:"sha256:test-evidence-profile".into(),
        }).unwrap();

        let effect=ActionEffectBinding::new("target-order","prod","adapter-order");
        store.pin_provider_adapter_configuration(
            &super::ProviderAdapterConfiguration::new(
                "adapter-order","test-adapter/v1","sha256:test-adapter-implementation"
            )
        ).unwrap();
        let action=EpistemicAction::new(
            "verifier-order","effect",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"verifier-order".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound_with_operation(
            &witness,&action,"frame@1","attempt-order","boundary-order","operation-order"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-order",&action,&effect,
            "boundary-order","operation-order","native-order"
        ).unwrap();
        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let mut forged=record.clone();
        forged.target_identity="attacker-target".into();
        let calls=Arc::new(AtomicUsize::new(0));
        let err=store.commit_bound_verified(
            &forged,&evidence,&CountingProviderVerifier { calls: calls.clone() }
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding)
        ));
        assert_eq!(calls.load(Ordering::SeqCst),0);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn reconciliation_verifier_is_not_called_for_unvalidated_dispatch_record() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-reconcile-verifier-order-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"legacy-local").unwrap();
        store.pin_provider_evidence_verifier_configuration(&ProviderVerifierConfiguration {
            relying_party_id:"legacy-local".into(),
            verifier_id:"test-verifier/v1".into(),
            verifier_revision:"test-verifier-rev/1".into(),
            verifier_implementation_id:"test-verifier-impl".into(),
            verifier_implementation_digest:"sha256:test-verifier-implementation".into(),
            verifier_config_digest:"sha256:test-verifier-config".into(),
            trust_anchor_digest:"sha256:test-trust-anchors".into(),
            evidence_profile_digest:"sha256:test-evidence-profile".into(),
        }).unwrap();

        let effect=ActionEffectBinding::new("target-reconcile-order","prod","adapter-reconcile-order");
        store.pin_provider_adapter_configuration(
            &super::ProviderAdapterConfiguration::new(
                "adapter-reconcile-order","test-adapter/v1","sha256:test-adapter-implementation"
            )
        ).unwrap();
        let action=EpistemicAction::new(
            "reconcile-verifier-order","effect",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"reconcile-verifier-order".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound_with_operation(
            &witness,&action,"frame@1","attempt-reconcile-order",
            "boundary-reconcile-order","operation-reconcile-order"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-reconcile-order",&action,&effect,
            "boundary-reconcile-order","operation-reconcile-order","native-reconcile-order"
        ).unwrap();
        store.commit_bound(&record,ExecutionOutcome::Indeterminate).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let mut forged=record.clone();
        forged.target_identity="attacker-reconcile-target".into();
        let calls=Arc::new(AtomicUsize::new(0));
        let err=store.reconcile_indeterminate_bound_verified(
            &forged,
            &evidence,
            &CountingProviderVerifier { calls: calls.clone() }
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding)
        ));
        assert_eq!(calls.load(Ordering::SeqCst),0);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_commit_requires_relying_party_pinned_verifier_configuration() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-verifier-pin-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-verifier-pin","prod","adapter-verifier-pin"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"verifier-pin".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T06:40:00Z".into(),
            expires_at:Some("2026-10-04T06:40:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-verifier-pin","boundary-verifier-pin"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-verifier-pin",
            &action,&effect,"boundary-verifier-pin",
            "operation:verifier-pin","native-verifier-pin"
        ).unwrap();

        store.connection().unwrap().execute(
            "DELETE FROM authorization_store_metadata
             WHERE key LIKE 'provider_evidence_verifier_%'",
            [],
        ).unwrap();

        let err=store.commit_bound_verified(
            &record,
            &verified_evidence(&record,ExecutionOutcome::Succeeded),
            &TestProviderVerifier
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            )
        ));

        store.pin_provider_evidence_verifier_configuration(&ProviderVerifierConfiguration {
            relying_party_id: store.relying_party_id().to_owned(),
            verifier_id: "test-verifier/v1".into(),
            verifier_revision: "test-verifier/rev1".into(),
            verifier_implementation_id: "test-verifier".into(),
            verifier_implementation_digest: "sha256:test-verifier-implementation".into(),
            verifier_config_digest: "sha256:test-verifier-config".into(),
            trust_anchor_digest: "sha256:test-trust-anchors".into(),
            evidence_profile_digest: "sha256:test-evidence-profile".into(),
        }).unwrap();

        let mismatch=store.commit_bound_verified(
            &record,
            &verified_evidence(&record,ExecutionOutcome::Succeeded),
            &TestProviderVerifierForRp { relying_party_id: "wrong-rp".into() }
        ).unwrap_err();
        assert!(matches!(
            mismatch,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            )
        ));

        let receipt=store.commit_bound_verified(
            &record,
            &verified_evidence(&record,ExecutionOutcome::Succeeded),
            &TestProviderVerifier
        ).unwrap();
        assert_eq!(receipt.outcome,ExecutionOutcome::Succeeded);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn provider_adapter_configuration_is_write_once_and_required() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-adapter-pin-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-adapter-pin"
        ).unwrap();
        let pinned=ProviderAdapterConfiguration::new(
            "adapter/v1","rev-1","sha256:adapter-impl-1"
        );
        store.pin_provider_adapter_configuration(&pinned).unwrap();
        store.pin_provider_adapter_configuration(&pinned).unwrap();
        assert!(matches!(
            store.pin_provider_adapter_configuration(&ProviderAdapterConfiguration::new(
                "adapter/v1","rev-2","sha256:adapter-impl-2"
            )),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        assert!(matches!(
            store.pinned_provider_adapter_configuration("missing/adapter"),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn stale_adapter_configuration_is_rejected_at_dispatch_commit() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-auth-adapter-stale-commit-{}.db",
            std::process::id()
        ));
        let store = SqliteAuthorizationStore::open(&path).unwrap();
        let effect = ActionEffectBinding::new(
            "target-adapter-stale",
            "prod",
            "adapter-adapter-stale",
        );
        let action = EpistemicAction::new(
            "adapter-stale-commit",
            "intervention",
            super::super::ActionRisk::Critical,
        )
        .with_effect_binding(effect.clone());
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance: "adapter-stale-commit".into(),
            action_id: action.id.clone(),
            action_digest: digest.clone(),
            frame: "frame@1".into(),
            support_digest: "sha256:support".into(),
            policy: "policy-v1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-02T20:00:00Z".into(),
            expires_at: Some("2026-10-04T12:00:00Z".into()),
            authority_epoch: 1,
        };

        store
            .register_lease(&AuthorizationLease::new_with_instance(
                witness.authorization_instance.clone(),
                action.id.clone(),
                digest.clone(),
                witness.support_digest.clone(),
                witness.policy.clone(),
                1,
                1,
            ))
            .unwrap();
        store
            .pin_provider_status_source_digest("sha256:test-status-source")
            .unwrap();
        store
            .pin_provider_status_verifier_configuration(
                &TestProviderStatusVerifier.configuration(),
            )
            .unwrap();
        let adapter_configuration = ProviderAdapterConfiguration::new(
            effect.adapter.clone(),
            "test-adapter/v1",
            "sha256:test-adapter-implementation",
        );
        store
            .pin_provider_adapter_configuration(&adapter_configuration)
            .unwrap();
        store
            .pin_native_authority_namespace("test-issuer", "test-authority/v1")
            .unwrap();
        store
            .prepare_for_execution_bound_with_operation(
                &witness,
                &action,
                "frame@1",
                "attempt-adapter-stale",
                "boundary-adapter-stale",
                "operation-adapter-stale",
            )
            .unwrap();

        let native_authorization_id = "native-adapter-stale";
        let authority_namespace = store
            .pinned_native_authority_namespace("test-issuer")
            .unwrap();
        let replay = NativeReplayDerivation::derive(
            authority_namespace,
            native_authorization_id,
        )
        .unwrap();
        let status_verifier = TestProviderStatusVerifier;
        let status_identifier = format!("status:{native_authorization_id}");
        let status = status_verifier
            .verify_current_status(
                ProviderStatusVerificationPurpose::Admission,
                "test-issuer",
                &replay.authority_namespace,
                native_authorization_id,
                &status_identifier,
                &digest,
                &effect.target_identity,
                &effect.audience,
                &effect.adapter,
                Some("sha256:test-status-source"),
            )
            .unwrap();
        let status_verifier_configuration = status_verifier.configuration();

        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE authorization_provider_adapter_pins
                 SET adapter_revision='tampered-before-commit'
                 WHERE adapter_id=?1",
                params![effect.adapter.as_str()],
            )
            .unwrap();

        let err = store
            .mark_dispatch_pending_bound_with_provenance(
                &witness.authorization_instance,
                "attempt-adapter-stale",
                &action,
                &effect,
                "boundary-adapter-stale",
                "operation-adapter-stale",
                &replay.native_replay_identity,
                "test-issuer",
                &replay,
                &status,
                &adapter_configuration,
                &status_verifier_configuration,
            )
            .unwrap_err();

        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row::<i64, _, _>(
                    "SELECT COUNT(*) FROM authorization_dispatches",
                    [],
                    |row| row.get(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row::<String, _, _>(
                    "SELECT state FROM authorization_leases WHERE authorization_instance=?1",
                    params![witness.authorization_instance.as_str()],
                    |row| row.get(0)
                )
                .unwrap(),
            "prepared"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn adapter_pin_tampering_is_rejected_before_provider_entry() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-adapter-preentry-tamper-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-adapter-preentry","prod","adapter-adapter-preentry"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"adapter-preentry-tamper".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T07:35:00Z".into(),
            expires_at:Some("2026-10-04T07:35:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-adapter-preentry","boundary-adapter-preentry"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-adapter-preentry",
            &action,&effect,"boundary-adapter-preentry",
            "operation:adapter-preentry","native-adapter-preentry"
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_provider_adapter_pins
             SET adapter_revision='tampered-revision'
             WHERE adapter_id=?1",
            params![record.adapter.as_str()],
        ).unwrap();

        let err=store.mark_invoked_bound(&record,&TestProviderStatusVerifier).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"dispatch_pending");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn adapter_pin_tampering_is_rejected_before_terminal_settlement() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-adapter-tamper-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-adapter-tamper","prod","adapter-adapter-tamper"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"adapter-tamper".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T07:30:00Z".into(),
            expires_at:Some("2026-10-04T07:30:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-adapter-tamper","boundary-adapter-tamper"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-adapter-tamper",
            &action,&effect,"boundary-adapter-tamper",
            "operation:adapter-tamper","native-adapter-tamper"
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_provider_adapter_pins
             SET implementation_digest='sha256:tampered-adapter'
             WHERE adapter_id=?1",
            params![record.adapter.as_str()],
        ).unwrap();

        let err=store.commit_bound_verified(
            &record,&verified_evidence(&record,ExecutionOutcome::Succeeded),&TestProviderVerifier
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn historical_verifier_pin_can_be_completed_once() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-verifier-pin-upgrade-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-verifier-pin-upgrade"
        ).unwrap();
        let metadata=[
            ("provider_evidence_verifier_relying_party_id","rp-verifier-pin-upgrade"),
            ("provider_evidence_verifier_id","test-verifier/v1"),
            ("provider_evidence_verifier_config_digest","sha256:test-verifier-config"),
            ("provider_evidence_verifier_trust_anchor_digest","sha256:test-trust-anchors"),
            ("provider_evidence_verifier_evidence_profile_digest","sha256:test-evidence-profile"),
        ];
        for (key,value) in metadata {
            store.connection().unwrap().execute(
                "INSERT INTO authorization_store_metadata(key,value) VALUES(?1,?2)",
                params![key,value],
            ).unwrap();
        }
        let upgraded=ProviderVerifierConfiguration {
            relying_party_id:"rp-verifier-pin-upgrade".into(),
            verifier_id:"test-verifier/v1".into(),
            verifier_revision:"test-verifier/rev1".into(),
            verifier_implementation_id:"test-verifier".into(),
            verifier_implementation_digest:"sha256:test-verifier-implementation".into(),
            verifier_config_digest:"sha256:test-verifier-config".into(),
            trust_anchor_digest:"sha256:test-trust-anchors".into(),
            evidence_profile_digest:"sha256:test-evidence-profile".into(),
        };
        store.pin_provider_evidence_verifier_configuration(&upgraded).unwrap();
        assert_eq!(
            store.pinned_provider_evidence_verifier_configuration().unwrap(),
            upgraded
        );
        let changed=ProviderVerifierConfiguration {
            verifier_implementation_digest:"sha256:changed".into(),
            ..upgraded
        };
        assert!(matches!(
            store.pin_provider_evidence_verifier_configuration(&changed),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_receipt_replay_rejects_tampered_validity_provenance() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-replay-validity-tamper-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-terminal-replay-validity-tamper",
            "prod",
            "adapter-terminal-replay-validity-tamper"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"terminal-replay-validity-tamper".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:40:00Z".into(),
            expires_at:Some("2026-10-05T07:40:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,
            &action,
            "frame@1",
            "attempt-terminal-replay-validity-tamper",
            "boundary-terminal-replay-validity-tamper"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-terminal-replay-validity-tamper",
            &action,
            &effect,
            "boundary-terminal-replay-validity-tamper",
            "operation:terminal-replay-validity-tamper",
            "native-terminal-replay-validity-tamper"
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let _first=store.commit_bound_verified(
            &record,
            &evidence,
            &TestProviderVerifier
        ).unwrap();

        let persisted_receipt_operation: Option<String> = store.connection().unwrap().query_row(
            "SELECT operation_id
             FROM authorization_receipts
             WHERE authorization_instance=?1 AND attempt_id=?2 AND phase='final'",
            params![record.authorization_instance.as_str(),record.attempt_id.as_str()],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(persisted_receipt_operation.as_deref(), Some(record.operation_id.as_str()));

        store.connection().unwrap().execute(
            "UPDATE authorization_receipts
             SET operation_id='operation:tampered-receipt'
             WHERE authorization_instance=?1 AND attempt_id=?2 AND phase='final'",
            params![record.authorization_instance.as_str(),record.attempt_id.as_str()],
        ).unwrap();
        assert!(matches!(
            store.commit_bound_verified(
                &record,
                &evidence,
                &TestProviderVerifier,
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));
        store.connection().unwrap().execute(
            "UPDATE authorization_receipts
             SET operation_id=?3
             WHERE authorization_instance=?1 AND attempt_id=?2 AND phase='final'",
            params![
                record.authorization_instance.as_str(),
                record.attempt_id.as_str(),
                record.operation_id.as_str()
            ],
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_terminal_evidence
             SET validity_policy_digest='sha256:tampered-terminal-validity'
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();

        struct RefusingVerifier;
        impl ProviderEvidenceVerifier for RefusingVerifier {
            fn verify(
                &self,
                _purpose:ProviderVerificationPurpose,
                _record:&DurableDispatchRecord,
                _evidence:&ProviderTerminalEvidence,
            ) -> Result<VerifiedProviderOutcome,ProviderVerificationError> {
                Err(ProviderVerificationError::VerificationFailed)
            }
        }

        let err=store.commit_bound_verified(
            &record,
            &evidence,
            &RefusingVerifier,
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));

        let persisted:String=store.connection().unwrap().query_row(
            "SELECT validity_policy_digest
             FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(persisted,"sha256:tampered-terminal-validity");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn bound_commit_verified_rejects_tampered_lease_operation_id() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-lease-operation-tamper-verified-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-lease-operation-tamper-verified",
            "prod",
            "adapter-lease-operation-tamper-verified"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"lease-operation-tamper-verified".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:40:00Z".into(),
            expires_at:Some("2026-10-05T07:40:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1",
            "attempt-lease-operation-tamper-verified",
            "boundary-lease-operation-tamper-verified"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-lease-operation-tamper-verified",
            &action,
            &effect,
            "boundary-lease-operation-tamper-verified",
            "operation:lease-operation-tamper-verified",
            "native-lease-operation-tamper-verified"
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_leases
             SET operation_id='operation:tampered-lease'
             WHERE authorization_instance=?1",
            params![record.authorization_instance],
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let err=store.commit_bound_verified(
            &record,&evidence,&TestProviderVerifier
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));

        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"dispatch_pending");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn bound_reconciliation_rejects_tampered_lease_operation_id() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-lease-operation-tamper-reconcile-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-lease-operation-tamper-reconcile",
            "prod",
            "adapter-lease-operation-tamper-reconcile"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"lease-operation-tamper-reconcile".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:40:00Z".into(),
            expires_at:Some("2026-10-05T07:40:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1",
            "attempt-lease-operation-tamper-reconcile",
            "boundary-lease-operation-tamper-reconcile"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-lease-operation-tamper-reconcile",
            &action,
            &effect,
            "boundary-lease-operation-tamper-reconcile",
            "operation:lease-operation-tamper-reconcile",
            "native-lease-operation-tamper-reconcile"
        ).unwrap();
        store.commit_bound(&record,ExecutionOutcome::Indeterminate).unwrap();
        store.connection().unwrap().execute(
            "UPDATE authorization_leases
             SET operation_id='operation:tampered-lease'
             WHERE authorization_instance=?1",
            params![record.authorization_instance],
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let err=store.reconcile_indeterminate_bound_verified(
            &record,&evidence,&TestProviderVerifier
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));

        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"indeterminate");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn bound_indeterminate_commit_rejects_tampered_lease_operation_id() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-lease-operation-tamper-indeterminate-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-lease-operation-tamper-indeterminate",
            "prod",
            "adapter-lease-operation-tamper-indeterminate"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"lease-operation-tamper-indeterminate".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:40:00Z".into(),
            expires_at:Some("2026-10-05T07:40:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1",
            "attempt-lease-operation-tamper-indeterminate",
            "boundary-lease-operation-tamper-indeterminate"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-lease-operation-tamper-indeterminate",
            &action,
            &effect,
            "boundary-lease-operation-tamper-indeterminate",
            "operation:lease-operation-tamper-indeterminate",
            "native-lease-operation-tamper-indeterminate"
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_leases
             SET operation_id='operation:tampered-lease'
             WHERE authorization_instance=?1",
            params![record.authorization_instance],
        ).unwrap();

        let err=store.commit_bound(
            &record,ExecutionOutcome::Indeterminate
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));

        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"dispatch_pending");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_verifier_configuration_is_write_once() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-verifier-write-once-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-verifier-write-once"
        ).unwrap();
        let pinned=ProviderVerifierConfiguration {
            relying_party_id:"rp-verifier-write-once".into(),
            verifier_id:"test-verifier/v1".into(),
            verifier_revision: "test-verifier/rev1".into(),
            verifier_implementation_id: "test-verifier".into(),
            verifier_implementation_digest: "sha256:test-verifier-implementation".into(),
            verifier_config_digest:"sha256:test-verifier-config".into(),
            trust_anchor_digest:"sha256:test-trust-anchors".into(),
            evidence_profile_digest:"sha256:test-evidence-profile".into(),
        };
        store.pin_provider_evidence_verifier_configuration(&pinned).unwrap();
        store.pin_provider_evidence_verifier_configuration(&pinned).unwrap();

        let changed=ProviderVerifierConfiguration {
            verifier_config_digest:"sha256:changed-config".into(),
            ..pinned.clone()
        };
        assert!(matches!(
            store.pin_provider_evidence_verifier_configuration(&changed),
            Err(AuthorizationStoreError::InvalidState(_))
        ));

        let changed_revision=ProviderVerifierConfiguration {
            verifier_revision:"test-verifier/rev2".into(),
            ..pinned.clone()
        };
        assert!(matches!(
            store.pin_provider_evidence_verifier_configuration(&changed_revision),
            Err(AuthorizationStoreError::InvalidState(_))
        ));

        let changed_implementation=ProviderVerifierConfiguration {
            verifier_implementation_digest:"sha256:changed-implementation".into(),
            ..pinned.clone()
        };
        assert!(matches!(
            store.pin_provider_evidence_verifier_configuration(&changed_implementation),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_verifier_implementation_pin_drift_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-verifier-implementation-drift-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-verifier-implementation-drift","prod","adapter-verifier-drift"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"verifier-implementation-drift".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T06:40:00Z".into(),
            expires_at:Some("2026-10-04T06:40:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-verifier-drift","boundary-verifier-drift"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-verifier-drift",
            &action,&effect,"boundary-verifier-drift",
            "operation:verifier-drift","native-verifier-drift"
        ).unwrap();

        store.connection().unwrap().execute(
            "UPDATE authorization_store_metadata
             SET value='sha256:tampered-verifier-implementation'
             WHERE key='provider_evidence_verifier_implementation_digest'",
            [],
        ).unwrap();

        let err=store.commit_bound_verified(
            &record,&verified_evidence(&record,ExecutionOutcome::Succeeded),&TestProviderVerifier
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            )
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_verifier_pin_drift_after_external_verification_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-verifier-external-drift-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-verifier-external-drift","prod","adapter-verifier-external-drift"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"verifier-external-drift".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T06:40:00Z".into(),
            expires_at:Some("2026-10-05T06:40:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-verifier-external-drift",
            "boundary-verifier-external-drift"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-verifier-external-drift",
            &action,&effect,"boundary-verifier-external-drift",
            "operation:verifier-external-drift","native-verifier-external-drift"
        ).unwrap();

        struct MutatingTerminalVerifier { path: std::path::PathBuf }
        impl ProviderEvidenceVerifier for MutatingTerminalVerifier {
            fn verify(
                &self,
                purpose: ProviderVerificationPurpose,
                _record: &DurableDispatchRecord,
                evidence: &ProviderTerminalEvidence,
            ) -> Result<VerifiedProviderOutcome,ProviderVerificationError> {
                if !matches!(purpose,ProviderVerificationPurpose::TerminalOutcome) {
                    return Err(ProviderVerificationError::VerificationFailed);
                }
                let connection=Connection::open(&self.path)
                    .map_err(|_| ProviderVerificationError::VerificationFailed)?;
                connection.execute(
                    "UPDATE authorization_store_metadata
                     SET value='sha256:tampered-verifier-external'
                     WHERE key='provider_evidence_verifier_implementation_digest'",
                    [],
                ).map_err(|_| ProviderVerificationError::VerificationFailed)?;

                Ok(VerifiedProviderOutcome {
                    evidence:evidence.clone(),
                    configuration:ProviderVerifierConfiguration {
                        relying_party_id:"legacy-local".into(),
                        verifier_id:"test-verifier/v1".into(),
                        verifier_revision:"test-verifier/rev1".into(),
                        verifier_implementation_id:"test-verifier".into(),
                        verifier_implementation_digest:"sha256:test-verifier-implementation".into(),
                        verifier_config_digest:"sha256:test-verifier-config".into(),
                        trust_anchor_digest:"sha256:test-trust-anchors".into(),
                        evidence_profile_digest:"sha256:test-evidence-profile".into(),
                    },
                    verification_digest:"sha256:test-verification".into(),
                })
            }
        }

        let err=store.commit_bound_verified(
            &record,
            &verified_evidence(&record,ExecutionOutcome::Succeeded),
            &MutatingTerminalVerifier { path:path.clone() },
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            )
        ));

        let persisted:String=store.connection().unwrap().query_row(
            "SELECT value FROM authorization_store_metadata
             WHERE key='provider_evidence_verifier_implementation_digest'",
            [],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(persisted,"sha256:tampered-verifier-external");
        assert_eq!(
            store.connection().unwrap().query_row::<i64,_,_>(
                "SELECT COUNT(*) FROM authorization_terminal_evidence
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![record.authorization_instance,record.attempt_id],
                |r| r.get(0)
            ).unwrap(),
            0
        );
        assert_eq!(
            store.connection().unwrap().query_row::<String,_,_>(
                "SELECT state FROM authorization_dispatches
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![record.authorization_instance,record.attempt_id],
                |r| r.get(0)
            ).unwrap(),
            "dispatch_pending"
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_receipt_replay_does_not_reinvoke_external_verifier() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-replay-no-verifier-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-terminal-replay-no-verifier","prod","adapter-terminal-replay-no-verifier"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"terminal-replay-no-verifier".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:00:00Z".into(),
            expires_at:Some("2026-10-05T07:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1",
            "attempt-terminal-replay-no-verifier",
            "boundary-terminal-replay-no-verifier"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-terminal-replay-no-verifier",
            &action,
            &effect,
            "boundary-terminal-replay-no-verifier",
            "operation:terminal-replay-no-verifier",
            "native-terminal-replay-no-verifier"
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let first=store.commit_bound_verified(&record,&evidence,&TestProviderVerifier).unwrap();

        struct RefusingVerifier;
        impl ProviderEvidenceVerifier for RefusingVerifier {
            fn verify(
                &self,
                _purpose:ProviderVerificationPurpose,
                _record:&DurableDispatchRecord,
                _evidence:&ProviderTerminalEvidence,
            ) -> Result<VerifiedProviderOutcome,ProviderVerificationError> {
                Err(ProviderVerificationError::VerificationFailed)
            }
        }

        let replay=store.commit_bound_verified(&record,&evidence,&RefusingVerifier).unwrap();
        assert_eq!(replay,first);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn reconciliation_terminal_receipt_replay_does_not_reinvoke_external_verifier() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-reconcile-terminal-replay-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-reconcile-terminal-replay","prod","adapter-reconcile-terminal-replay"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"reconcile-terminal-replay".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:10:00Z".into(),
            expires_at:Some("2026-10-05T07:10:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1",
            "attempt-reconcile-terminal-replay",
            "boundary-reconcile-terminal-replay"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-reconcile-terminal-replay",
            &action,
            &effect,
            "boundary-reconcile-terminal-replay",
            "operation:reconcile-terminal-replay",
            "native-reconcile-terminal-replay"
        ).unwrap();
        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let first=store.commit_bound_verified(&record,&evidence,&TestProviderVerifier).unwrap();

        struct RefusingVerifier;
        impl ProviderEvidenceVerifier for RefusingVerifier {
            fn verify(
                &self,
                _purpose:ProviderVerificationPurpose,
                _record:&DurableDispatchRecord,
                _evidence:&ProviderTerminalEvidence,
            ) -> Result<VerifiedProviderOutcome,ProviderVerificationError> {
                Err(ProviderVerificationError::VerificationFailed)
            }
        }

        let replay=store.reconcile_indeterminate_bound_verified(
            &record,&evidence,&RefusingVerifier
        ).unwrap();
        assert_eq!(replay,first);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_receipt_replay_rejects_missing_terminal_evidence() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-replay-missing-evidence-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-terminal-replay-missing-evidence",
            "prod",
            "adapter-terminal-replay-missing-evidence"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"terminal-replay-missing-evidence".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:05:00Z".into(),
            expires_at:Some("2026-10-05T07:05:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,
            &action,
            "frame@1",
            "attempt-terminal-replay-missing-evidence",
            "boundary-terminal-replay-missing-evidence"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-terminal-replay-missing-evidence",
            &action,
            &effect,
            "boundary-terminal-replay-missing-evidence",
            "operation:terminal-replay-missing-evidence",
            "native-terminal-replay-missing-evidence"
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let first=store.commit_bound_verified(&record,&evidence,&TestProviderVerifier).unwrap();
        assert_eq!(
            store.connection().unwrap().query_row::<i64,_,_>(
                "SELECT COUNT(*) FROM authorization_terminal_evidence
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![record.authorization_instance,record.attempt_id],
                |r| r.get(0)
            ).unwrap(),
            1
        );
        store.connection().unwrap().execute(
            "DELETE FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();

        struct RefusingVerifier;
        impl ProviderEvidenceVerifier for RefusingVerifier {
            fn verify(
                &self,
                _purpose:ProviderVerificationPurpose,
                _record:&DurableDispatchRecord,
                _evidence:&ProviderTerminalEvidence,
            ) -> Result<VerifiedProviderOutcome,ProviderVerificationError> {
                Err(ProviderVerificationError::VerificationFailed)
            }
        }

        let err=store.commit_bound_verified(
            &record,
            &evidence,
            &RefusingVerifier,
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        assert_eq!(
            store.connection().unwrap().query_row::<i64,_,_>(
                "SELECT COUNT(*) FROM authorization_terminal_evidence
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![record.authorization_instance,record.attempt_id],
                |r| r.get(0)
            ).unwrap(),
            0
        );
        assert_eq!(first.outcome,ExecutionOutcome::Succeeded);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_receipt_replay_rejects_missing_pin_set_snapshot() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-replay-pin-snapshot-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-terminal-replay-pin-snapshot",
            "prod",
            "adapter-terminal-replay-pin-snapshot"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"terminal-replay-pin-snapshot".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:20:00Z".into(),
            expires_at:Some("2026-10-05T07:20:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,
            &action,
            "frame@1",
            "attempt-terminal-replay-pin-snapshot",
            "boundary-terminal-replay-pin-snapshot"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-terminal-replay-pin-snapshot",
            &action,
            &effect,
            "boundary-terminal-replay-pin-snapshot",
            "operation:terminal-replay-pin-snapshot",
            "native-terminal-replay-pin-snapshot"
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        store.commit_bound_verified(
            &record,
            &evidence,
            &TestProviderVerifier
        ).unwrap();

        let (pin_set_id,pin_set_digest): (String,String)=store.connection().unwrap().query_row(
            "SELECT native_authority_pin_set_id,native_authority_pin_set_digest
             FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |row| Ok((row.get(0)?,row.get(1)?)),
        ).unwrap();
        let deleted=store.connection().unwrap().execute(
            "DELETE FROM authorization_native_authority_pin_sets
             WHERE pin_set_id=?1 AND pin_set_digest=?2",
            params![pin_set_id,pin_set_digest],
        ).unwrap();
        assert_eq!(deleted,1);

        let calls=Arc::new(AtomicUsize::new(0));
        let err=store.commit_bound_verified(
            &record,
            &evidence,
            &CountingProviderVerifier { calls:calls.clone() },
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            )
        ));
        assert_eq!(calls.load(Ordering::SeqCst),0);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_receipt_replay_rejects_tampered_authority_epoch() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-replay-epoch-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-terminal-replay-epoch",
            "prod",
            "adapter-terminal-replay-epoch"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"terminal-replay-epoch".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T07:15:00Z".into(),
            expires_at:Some("2026-10-05T07:15:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,
            &action,
            "frame@1",
            "attempt-terminal-replay-epoch",
            "boundary-terminal-replay-epoch"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-terminal-replay-epoch",
            &action,
            &effect,
            "boundary-terminal-replay-epoch",
            "operation:terminal-replay-epoch",
            "native-terminal-replay-epoch"
        ).unwrap();

        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let first=store.commit_bound_verified(
            &record,
            &evidence,
            &TestProviderVerifier
        ).unwrap();
        assert_eq!(first.authority_epoch,1);

        store.connection().unwrap().execute(
            "UPDATE authorization_receipts
             SET authority_epoch=999
             WHERE authorization_instance=?1 AND attempt_id=?2 AND phase='final'",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();

        let calls=Arc::new(AtomicUsize::new(0));
        let err=store.commit_bound_verified(
            &record,
            &evidence,
            &CountingProviderVerifier { calls:calls.clone() },
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        assert_eq!(calls.load(Ordering::SeqCst),0);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn reconciliation_verifier_pin_drift_after_external_verification_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-reconcile-verifier-external-drift-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-reconcile-verifier-external-drift",
            "prod",
            "adapter-reconcile-verifier-external-drift"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"reconcile-verifier-external-drift".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-04T06:45:00Z".into(),
            expires_at:Some("2026-10-05T06:45:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,
            &action,
            "frame@1",
            "attempt-reconcile-verifier-external-drift",
            "boundary-reconcile-verifier-external-drift"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,
            &witness.authorization_instance,
            "attempt-reconcile-verifier-external-drift",
            &action,
            &effect,
            "boundary-reconcile-verifier-external-drift",
            "operation:reconcile-verifier-external-drift",
            "native-reconcile-verifier-external-drift"
        ).unwrap();
        store.commit_bound(&record,ExecutionOutcome::Indeterminate).unwrap();

        struct MutatingTerminalVerifier { path: std::path::PathBuf }
        impl ProviderEvidenceVerifier for MutatingTerminalVerifier {
            fn verify(
                &self,
                purpose: ProviderVerificationPurpose,
                _record: &DurableDispatchRecord,
                evidence: &ProviderTerminalEvidence,
            ) -> Result<VerifiedProviderOutcome,ProviderVerificationError> {
                if !matches!(purpose,ProviderVerificationPurpose::TerminalOutcome) {
                    return Err(ProviderVerificationError::VerificationFailed);
                }
                let connection=Connection::open(&self.path)
                    .map_err(|_| ProviderVerificationError::VerificationFailed)?;
                connection.execute(
                    "UPDATE authorization_store_metadata
                     SET value='sha256:tampered-reconcile-verifier-external'
                     WHERE key='provider_evidence_verifier_implementation_digest'",
                    [],
                ).map_err(|_| ProviderVerificationError::VerificationFailed)?;

                Ok(VerifiedProviderOutcome {
                    evidence:evidence.clone(),
                    configuration:ProviderVerifierConfiguration {
                        relying_party_id:"legacy-local".into(),
                        verifier_id:"test-verifier/v1".into(),
                        verifier_revision:"test-verifier/rev1".into(),
                        verifier_implementation_id:"test-verifier".into(),
                        verifier_implementation_digest:"sha256:test-verifier-implementation".into(),
                        verifier_config_digest:"sha256:test-verifier-config".into(),
                        trust_anchor_digest:"sha256:test-trust-anchors".into(),
                        evidence_profile_digest:"sha256:test-evidence-profile".into(),
                    },
                    verification_digest:"sha256:test-verification".into(),
                })
            }
        }

        let err=store.reconcile_indeterminate_bound_verified(
            &record,
            &verified_evidence(&record,ExecutionOutcome::Succeeded),
            &MutatingTerminalVerifier { path:path.clone() },
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            )
        ));

        let persisted:String=store.connection().unwrap().query_row(
            "SELECT value FROM authorization_store_metadata
             WHERE key='provider_evidence_verifier_implementation_digest'",
            [],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(persisted,"sha256:tampered-reconcile-verifier-external");
        assert_eq!(
            store.connection().unwrap().query_row::<i64,_,_>(
                "SELECT COUNT(*) FROM authorization_terminal_evidence
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![record.authorization_instance,record.attempt_id],
                |r| r.get(0)
            ).unwrap(),
            0
        );
        assert_eq!(
            store.connection().unwrap().query_row::<String,_,_>(
                "SELECT state FROM authorization_dispatches
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![record.authorization_instance,record.attempt_id],
                |r| r.get(0)
            ).unwrap(),
            "indeterminate"
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_verifier_configuration_mismatch_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-verifier-config-mismatch-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-verifier-config","prod","adapter-verifier-config"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"verifier-config-mismatch".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T06:45:00Z".into(),
            expires_at:Some("2026-10-04T06:45:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-verifier-config","boundary-verifier-config"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-verifier-config",
            &action,&effect,"boundary-verifier-config",
            "operation:verifier-config","native-verifier-config"
        ).unwrap();

        let err=store.commit_bound_verified(
            &record,&verified_evidence(&record,ExecutionOutcome::Succeeded),&TestProviderVerifierForRp { relying_party_id: "wrong-rp".into() }
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            )
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_evidence_storage_is_insert_only_for_attempt() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-insert-only-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-terminal-insert-only","prod","adapter-terminal"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let binding:String=store.connection().unwrap().query_row(
            "SELECT attempt_binding_digest FROM authorization_terminal_evidence
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(binding,record.attempt_binding_digest);
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
    fn native_issuer_provenance_is_persisted_and_validated() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-native-issuer-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-native-issuer","prod","adapter-native-issuer"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"native-issuer".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T07:00:00Z".into(),
            expires_at:Some("2026-10-04T07:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-native-issuer","boundary-native-issuer"
        ).unwrap();
        let record=mark_dispatch_pending_bound_from_native_authority_for_test(
            &store,&witness.authorization_instance,"attempt-native-issuer",
            &action,&effect,"boundary-native-issuer","operation:native-issuer",
            "https://issuer.example","native-auth-native-issuer"
        ).unwrap();

        let issuer:String=store.connection().unwrap().query_row(
            "SELECT native_issuer FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(issuer,"https://issuer.example");

        store.connection().unwrap().execute(
            "UPDATE authorization_dispatches SET native_issuer='https://other.example'
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();
        assert!(matches!(
            store.mark_invoked_bound(&record, &TestProviderStatusVerifier),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_evidence_persists_exact_adapter_identity() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-adapter-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-terminal-adapter","prod","adapter-terminal"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
    fn identical_pin_contents_have_distinct_relying_party_scoped_digests() {
        let path_a=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-pin-digest-rp-a-{}.db",std::process::id()
        ));
        let path_b=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-pin-digest-rp-b-{}.db",std::process::id()
        ));
        let store_a=SqliteAuthorizationStore::open_with_relying_party(
            &path_a,"rp-digest-a"
        ).unwrap();
        let store_b=SqliteAuthorizationStore::open_with_relying_party(
            &path_b,"rp-digest-b"
        ).unwrap();
        store_a.pin_native_authority_namespace("issuer","authority/v1").unwrap();
        store_b.pin_native_authority_namespace("issuer","authority/v1").unwrap();

        let read_digest=|store:&SqliteAuthorizationStore| -> String {
            let mut connection=store.connection().unwrap();
            let tx=connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
            let (_,digest)=store.persist_native_authority_pin_set_snapshot(&tx).unwrap();
            tx.commit().unwrap();
            digest
        };
        let digest_a=read_digest(&store_a);
        let digest_b=read_digest(&store_b);
        assert_ne!(digest_a,digest_b);

        let _=std::fs::remove_file(path_a);
        let _=std::fs::remove_file(path_b);
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
    fn normalized_issuer_aliases_share_native_replay_identity_when_namespace_is_pinned() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-issuer-replay-alias-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-issuer-replay-alias"
        ).unwrap();

        store.pin_native_authority_namespace(
            "HTTPS://Issuer.Example.:443/","authority/v1"
        ).unwrap();

        let namespace_a=store
            .pinned_native_authority_namespace("HTTPS://Issuer.Example.:443/")
            .unwrap();
        let namespace_b=store
            .pinned_native_authority_namespace("https:issuer.example")
            .unwrap();

        let replay_a=super::super::NativeReplayDerivation::derive(
            namespace_a,"grant-1"
        ).unwrap();
        let replay_b=super::super::NativeReplayDerivation::derive(
            namespace_b,"grant-1"
        ).unwrap();

        assert_eq!(namespace_a,namespace_b);
        assert_eq!(replay_a.native_replay_identity,replay_b.native_replay_identity);
        assert_eq!(replay_a.derivation_digest,replay_b.derivation_digest);

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
        let normalized_a=SqliteAuthorizationStore::normalize_native_issuer("HTTPS://Issuer.Example.:443/");
        let normalized_b=SqliteAuthorizationStore::normalize_native_issuer("https://issuer.example");
        let normalized_short_url=SqliteAuthorizationStore::normalize_native_issuer("https:issuer.example");
        let normalized_explicit_url=SqliteAuthorizationStore::normalize_native_issuer("https://issuer.example");
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

        let effect=ActionEffectBinding::new("target-pin-snapshot","prod","adapter");
        let action=EpistemicAction::new(
            "pin-snapshot-action","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
            store.commit_bound_verified(&record,&evidence,&TestProviderVerifierForRp { relying_party_id: store.relying_party_id().to_owned() }),
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
        store.commit_bound_verified(&record,&evidence,&TestProviderVerifierForRp { relying_party_id: store.relying_party_id().to_owned() }).unwrap();

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
        let mut digest_material=Vec::new();
        digest_material.extend_from_slice(b"symthaea:gis:native-pin-set-digest:v2\n");
        append_len_prefixed_bytes(&mut digest_material,snapshot.0.as_bytes());
        append_len_prefixed_bytes(&mut digest_material,&snapshot_bytes);
        assert_eq!(
            format!("sha256:v2:{}",hex::encode(Sha256::digest(digest_material))),
            snapshot.1
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn trusted_clock_source_is_injectable_and_pinned_across_reopen() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-trusted-clock-{}.db",std::process::id()
        ));
        let policy=AuthorizationClockPolicy {
            max_age_seconds: 3600,
            allowed_skew_seconds: 30,
            require_expiry: true,
        };
        let now=DateTime::parse_from_rfc3339("2026-10-03T06:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let store=SqliteAuthorizationStore::open_with_relying_party_clock_and_policy(
            &path,
            "rp-trusted-clock",
            Arc::new(FixedClock { now, source_id:"fixed-test-clock/v1" }),
            policy,
        ).unwrap();

        let effect=ActionEffectBinding::new(
            "target-trusted-clock","prod","adapter"
        );
        let action=EpistemicAction::new(
            "trusted-clock-action","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"trusted-clock-action".into(),
            action_id:action.id.clone(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T06:00:00Z".into(),
            expires_at:Some("2026-10-03T07:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "trusted-clock-action",action.id.clone(),digest,
            "support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-trusted-clock","boundary-clock"
        ).unwrap();

        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party_clock_and_policy(
                &path,
                "rp-trusted-clock",
                Arc::new(FixedClock {
                    now,
                    source_id:"different-clock/v1"
                }),
                policy,
            ),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
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
        let effect=ActionEffectBinding::new(
            "target-validity-immutable","prod","adapter"
        );
        let action=EpistemicAction::new(
            "validity-immutable","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
            &record,&verified_evidence(&record,ExecutionOutcome::Failed),&TestProviderVerifierForRp { relying_party_id: store.relying_party_id().to_owned() }
        ).unwrap();

        let extended=ActionAuthorizationWitness {
            operation_id: None,
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
        let effect=ActionEffectBinding::new(
            "target-validity-dispatch","prod","adapter"
        );
        let action=EpistemicAction::new(
            "validity-dispatch","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let marker_operation_id: Option<String> = store.connection().unwrap().query_row(
            "SELECT operation_id
             FROM authorization_recovery_markers
             WHERE authorization_instance=?1 AND attempt_id=?2 AND marker='not_entered_validity'",
            params![witness.authorization_instance.as_str(),"attempt-validity-dispatch"],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(marker_operation_id.as_deref(), Some("operation:validity-dispatch"));

        let renewed=ActionAuthorizationWitness {
            operation_id: None,
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
        let effect=ActionEffectBinding::new("target-validity","prod","adapter");
        let action=EpistemicAction::new(
            "validity-admission","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let effect=ActionEffectBinding::new("target-validity-preentry","prod","adapter");
        let action=EpistemicAction::new(
            "validity-preentry","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
            store.mark_invoked_bound(&record, &TestProviderStatusVerifier),
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
        let status_operation_id: Option<String> = store.connection().unwrap().query_row(
            "SELECT operation_id
             FROM authorization_status_checks
             WHERE authorization_instance=?1 AND attempt_id=?2 AND phase='pre_entry'",
            params![record.authorization_instance.as_str(),record.attempt_id.as_str()],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(status_operation_id.as_deref(), Some("operation:validity-preentry"));
        let marker_operation_id: Option<String> = store.connection().unwrap().query_row(
            "SELECT operation_id
             FROM authorization_recovery_markers
             WHERE authorization_instance=?1 AND attempt_id=?2 AND marker='not_entered_validity'",
            params![record.authorization_instance.as_str(),record.attempt_id.as_str()],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(marker_operation_id.as_deref(), Some("operation:validity-preentry"));

        let _=std::fs::remove_file(path);
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_provenance_less_bound_apis_fail_closed() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-legacy-fence-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let calls=Arc::new(AtomicUsize::new(0));
        assert!(matches!(
            store.commit_bound_verified(
                &record,
                &evidence,
                &CountingProviderVerifier { calls: calls.clone() }
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            ))
        ));
        assert_eq!(calls.load(Ordering::SeqCst),0);

        store.commit_bound(&record,ExecutionOutcome::Indeterminate).unwrap();
        let calls=Arc::new(AtomicUsize::new(0));
        assert!(matches!(
            store.reconcile_indeterminate_bound_verified(
                &record,
                &evidence,
                &CountingProviderVerifier { calls: calls.clone() }
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderEvidenceVerificationRequired
            ))
        ));
        assert_eq!(calls.load(Ordering::SeqCst),0);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_evidence_attempt_commitment_cannot_be_substituted() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-terminal-binding-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new("target-terminal-binding","prod","adapter-terminal");
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"terminal-binding".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T07:20:00Z".into(),
            expires_at:Some("2026-10-04T07:20:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-terminal-binding","boundary-terminal-binding"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-terminal-binding",
            &action,&effect,"boundary-terminal-binding",
            "operation:terminal-binding","native-terminal-binding"
        ).unwrap();
        let mut evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        evidence.attempt_binding_digest="sha256:forged-attempt-binding".into();

        let err=store.commit_bound_verified(
            &record,&evidence,&TestProviderVerifier
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"dispatch_pending");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn verified_terminal_evidence_is_exact_attempt_and_sink_bound() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-verifier-binding-{}.db",std::process::id()));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"reconcile-key".into(), action_id:action.id.clone(),
            action_digest:action.canonical_action_digest(), support_digest:witness.support_digest,
            frame:witness.frame, policy:witness.policy, authority_epoch:witness.authority_epoch,
        };
        store.prepare_for_execution_bound(&witness,&action,"frame@1","attempt-reconcile-key","boundary-A").unwrap();
        let record=mark_dispatch_pending_bound_for_test(&store,
            &witness.authorization_instance,"attempt-reconcile-key",&action,&effect,"boundary-A",
            "operation:reconcile-key","native-grant:reconcile"
        ).unwrap();
        store.mark_invoked_bound(&record, &TestProviderStatusVerifier).unwrap();
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        let effect=ActionEffectBinding::new("target-verifier-rp","prod","adapter-A");
        let action=EpistemicAction::new("verifier-rp","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
                        verifier_revision: "test-verifier/rev1".into(),
                        verifier_implementation_id: "test-verifier".into(),
                        verifier_implementation_digest: "sha256:test-verifier-implementation".into(),
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

        store.pin_provider_status_source_digest("sha256:test-status-source").unwrap();
        assert!(matches!(
            store.mark_dispatch_pending_bound_from_pinned_native_authority(
                "missing","attempt-missing",
                &EpistemicAction::new("missing-action","intervention",super::super::ActionRisk::Critical),
                &ActionEffectBinding::new("target-missing","prod","adapter"),
                "boundary","operation","issuer.unpinned","native-auth",
                "status:missing",&TestProviderStatusVerifier
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

        let effect=ActionEffectBinding::new("target-pin","prod","adapter-pin");
        let action=EpistemicAction::new("native-pin-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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

        store.pin_provider_status_source_digest("sha256:test-status-source").unwrap();
        let record=store.mark_dispatch_pending_bound_from_pinned_native_authority(
            "native-pin","attempt-native-pin",&action,&effect,"boundary-pin",
            "operation:native-pin","issuer.example","native-auth-pin",
            "status:native-auth-pin",&TestProviderStatusVerifier
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
            store.mark_invoked_bound(&wrong_operation, &TestProviderStatusVerifier),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))
        ));

        let mut wrong_native=record.clone();
        wrong_native.native_replay_identity.push_str("-forged");
        assert!(matches!(
            store.mark_invoked_bound(&wrong_native, &TestProviderStatusVerifier),
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


    #[test]
    fn durable_terminal_state_survives_restart_and_remains_idempotent() {
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
            operation_id: None,
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
    fn strict_pre_dispatch_recovery_is_idempotent_with_operation_binding() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-auth-pre-dispatch-recovery-operation-{}.db",
            std::process::id()
        ));
        let store = SqliteAuthorizationStore::open_with_relying_party(
            &path,
            "rp-pre-dispatch-recovery-operation",
        )
        .unwrap();
        let action = EpistemicAction::new(
            "pre-dispatch-operation-recovery",
            "effect",
            super::super::ActionRisk::Critical,
        );
        let digest = action.canonical_action_digest();
        let operation_id = "operation:pre-dispatch-recovery";
        let boundary_id = "boundary:pre-dispatch-recovery";
        let attempt_id = "attempt:pre-dispatch-recovery";
        let witness = ActionAuthorizationWitness {
            operation_id: Some(operation_id.into()),
            authorization_instance: "pre-dispatch-operation-recovery".into(),
            action_id: action.id.clone(),
            action_digest: digest.clone(),
            frame: "frame@1".into(),
            support_digest: "support".into(),
            policy: "policy@1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-03T11:00:00Z".into(),
            expires_at: Some("2026-10-03T12:00:00Z".into()),
            authority_epoch: 1,
        };

        store
            .register_lease(&AuthorizationLease::new_with_instance(
                witness.authorization_instance.clone(),
                action.id.clone(),
                digest,
                witness.support_digest.clone(),
                witness.policy.clone(),
                witness.authority_epoch,
                1,
            ))
            .unwrap();
        store
            .prepare_for_execution_bound_with_operation(
                &witness,
                &action,
                "frame@1",
                attempt_id,
                boundary_id,
                operation_id,
            )
            .unwrap();

        assert_eq!(
            store.recover_pre_dispatch_attempt(&RecoveryAuthorizationWitness {
                authorization_instance: witness.authorization_instance.clone(),
                attempt_id: attempt_id.into(),
                operation_id: operation_id.into(),
                boundary_id: boundary_id.into(),
                action_digest: witness.action_digest.clone(),
                policy: "recovery-policy-v1".into(),
                authority_epoch: witness.authority_epoch,
                issued_at: "2026-10-03T11:30:00Z".into(),
            })
            .unwrap(),
            true
        );

        let marker_operation_id: Option<String> = store
            .connection()
            .unwrap()
            .query_row(
                "SELECT operation_id
                 FROM authorization_recovery_markers
                 WHERE authorization_instance=?1 AND attempt_id=?2 AND marker='not_entered'",
                params![witness.authorization_instance.as_str(), attempt_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(marker_operation_id.as_deref(), Some(operation_id));

        store.connection().unwrap().execute(
            "UPDATE authorization_leases
             SET operation_id='operation:tampered'
             WHERE authorization_instance=?1",
            params![witness.authorization_instance.as_str()],
        ).unwrap();
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&RecoveryAuthorizationWitness {
                authorization_instance: witness.authorization_instance.clone(),
                attempt_id: attempt_id.into(),
                operation_id: operation_id.into(),
                boundary_id: boundary_id.into(),
                action_digest: witness.action_digest.clone(),
                policy: "recovery-policy-v1".into(),
                authority_epoch: witness.authority_epoch,
                issued_at: "2026-10-03T11:45:00Z".into(),
            }),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        );
        store.connection().unwrap().execute(
            "UPDATE authorization_leases
             SET operation_id=?2
             WHERE authorization_instance=?1",
            params![witness.authorization_instance.as_str(), operation_id],
        ).unwrap();
        assert_eq!(
            store
                .recover_pre_dispatch_attempt(&RecoveryAuthorizationWitness {
                    authorization_instance: witness.authorization_instance,
                    attempt_id: attempt_id.into(),
                    operation_id: operation_id.into(),
                    boundary_id: boundary_id.into(),
                    action_digest: witness.action_digest,
                    policy: "recovery-policy-v1".into(),
                    authority_epoch: witness.authority_epoch,
                    issued_at: "2026-10-03T11:45:00Z".into(),
                })
                .unwrap(),
            false
        );

        let _ = std::fs::remove_file(path);
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("bound-dispatch-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        assert!(matches!(store.mark_invoked_bound(&tampered, &TestProviderStatusVerifier),Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding))));
        store.mark_invoked_bound(&record, &TestProviderStatusVerifier).unwrap();
        assert!(store.mark_invoked_bound(&record, &TestProviderStatusVerifier).is_err());
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("same-action-fence","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();

        let witness_a=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"fence-a".into(), action_id:action.id.clone(),
            action_digest:digest.clone(), support_digest:"support-a".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            operation_id: None,
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
    fn provider_status_source_pin_is_write_once() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-source-pin-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        store.pin_provider_status_source_digest("sha256:status-source-a").unwrap();
        store.pin_provider_status_source_digest("sha256:status-source-a").unwrap();
        assert!(matches!(
            store.pin_provider_status_source_digest("sha256:status-source-b"),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn historical_status_verifier_pin_can_be_completed_once() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-verifier-upgrade-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        for (key,value) in [
            ("provider_status_verifier_id","status-verifier/v1"),
            ("provider_status_verifier_config_digest","sha256:status-verifier-config"),
        ] {
            store.connection().unwrap().execute(
                "INSERT INTO authorization_store_metadata(key,value) VALUES(?1,?2)",
                params![key,value],
            ).unwrap();
        }
        let upgraded=ProviderStatusVerifierConfiguration::new(
            "status-verifier/v1","status-verifier/rev1",
            "status-verifier","sha256:status-verifier-implementation",
            "sha256:status-verifier-config"
        );
        store.pin_provider_status_verifier_configuration(&upgraded).unwrap();
        assert_eq!(
            store.pinned_provider_status_verifier_configuration().unwrap(),
            upgraded
        );
        let changed=ProviderStatusVerifierConfiguration::new(
            "status-verifier/v1","status-verifier/rev2",
            "status-verifier","sha256:status-verifier-implementation",
            "sha256:status-verifier-config"
        );
        assert!(matches!(
            store.pin_provider_status_verifier_configuration(&changed),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn provider_status_verifier_configuration_is_write_once() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-verifier-pin-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let config=ProviderStatusVerifierConfiguration::new(
            "status-verifier/v1","status-verifier/rev1",
            "status-verifier","sha256:status-verifier-implementation",
            "sha256:status-verifier-config"
        );
        store.pin_provider_status_verifier_configuration(&config).unwrap();
        store.pin_provider_status_verifier_configuration(&config).unwrap();
        assert!(matches!(
            store.pin_provider_status_verifier_configuration(
                &ProviderStatusVerifierConfiguration::new(
                    "status-verifier/v2","status-verifier/rev2",
                    "status-verifier","sha256:status-verifier-implementation",
                    "sha256:other"
                )
            ),
            Err(AuthorizationStoreError::InvalidState(_))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn status_verifier_pin_drift_after_verification_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-verifier-drift-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-status-drift").unwrap();
        store.pin_native_authority_namespace("issuer.drift","issuer.drift/authority/v1").unwrap();
        store.pin_provider_status_source_digest("sha256:test-status-source").unwrap();
        store.pin_provider_status_verifier_configuration(
            &ProviderStatusVerifierConfiguration::new(
                "test-status-verifier/v1","test-status-verifier/rev1",
                "test-status-verifier","sha256:test-status-verifier-implementation",
                "sha256:test-status-verifier-config"
            )
        ).unwrap();
        store.pin_provider_adapter_configuration(
            &ProviderAdapterConfiguration::new(
                "adapter-status-drift","test-adapter/v1",
                "sha256:test-adapter-implementation"
            )
        ).unwrap();

        let effect=ActionEffectBinding::new("target-status-drift","prod","adapter-status-drift");
        let action=EpistemicAction::new(
            "status-verifier-drift","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"status-verifier-drift".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T06:20:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-status-verifier-drift","boundary-drift"
        ).unwrap();

        struct MutatingStatusVerifier { path: std::path::PathBuf }
        impl ProviderStatusVerifier for MutatingStatusVerifier {
            fn configuration(&self) -> ProviderStatusVerifierConfiguration {
                ProviderStatusVerifierConfiguration::new(
                    "test-status-verifier/v1","test-status-verifier/rev1",
                    "test-status-verifier","sha256:test-status-verifier-implementation",
                    "sha256:test-status-verifier-config"
                )
            }

            fn verify_current_status(
                &self,
                _purpose:ProviderStatusVerificationPurpose,
                _issuer:&str,
                _authority_namespace:&str,
                native_authorization_id:&str,
                status_identifier:&str,
                _action_digest:&str,
                _target_identity:&str,
                _audience:&str,
                _adapter:&str,
                _expected_source_digest:Option<&str>,
            ) -> Result<ProviderStatusEvidence,ProviderStatusVerificationError> {
                let conn=Connection::open(&self.path)
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                conn.execute(
                    "UPDATE authorization_store_metadata
                     SET value='sha256:tampered-status-implementation'
                     WHERE key='provider_status_verifier_implementation_digest'",
                    []
                ).map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                let observed=trusted_utc_now()
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                Ok(ProviderStatusEvidence {
                    status_identifier:status_identifier.to_owned(),
                    status_source_digest:"sha256:test-status-source".into(),
                    status_observed_at:observed.to_rfc3339_opts(SecondsFormat::Secs,true),
                    status_valid_until:(observed+Duration::seconds(1800))
                        .to_rfc3339_opts(SecondsFormat::Secs,true),
                    status_evidence_digest:format!("sha256:status:{native_authorization_id}"),
                })
            }
        }

        let err=store.mark_dispatch_pending_bound_from_pinned_native_authority(
            &witness.authorization_instance,
            "attempt-status-verifier-drift",
            &action,
            &effect,
            "boundary-drift",
            "operation:status-verifier-drift",
            "issuer.drift",
            "native-status-drift",
            "status:native-status-drift",
            &MutatingStatusVerifier { path:path.clone() },
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired
            )
        ));
        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_leases WHERE authorization_instance=?1",
            params![witness.authorization_instance],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"prepared");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn native_authority_pin_drift_after_admission_verification_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-auth-native-pin-admission-drift-{}.db",
            std::process::id()
        ));
        let store = SqliteAuthorizationStore::open_with_relying_party(
            &path,
            "rp-native-pin-admission",
        )
        .unwrap();
        store
            .pin_native_authority_namespace(
                "issuer.native-admission",
                "issuer.native-admission/authority/v1",
            )
            .unwrap();
        store
            .pin_provider_status_source_digest("sha256:test-status-source")
            .unwrap();
        store
            .pin_provider_status_verifier_configuration(
                &TestProviderStatusVerifier.configuration(),
            )
            .unwrap();
        store
            .pin_provider_adapter_configuration(&ProviderAdapterConfiguration::new(
                "adapter-native-admission",
                "test-adapter/v1",
                "sha256:test-adapter-implementation",
            ))
            .unwrap();

        let effect = ActionEffectBinding::new(
            "target-native-admission",
            "prod",
            "adapter-native-admission",
        );
        let action = EpistemicAction::new(
            "native-pin-admission",
            "intervention",
            super::super::ActionRisk::Critical,
        )
        .with_effect_binding(effect.clone());
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            operation_id: None,
            action_id: action.id.clone(),
            authorization_instance: "native-pin-admission".into(),
            action_digest: digest.clone(),
            frame: "frame@1".into(),
            support_digest: "support".into(),
            policy: "policy@1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-03T06:20:00Z".into(),
            expires_at: Some("2026-10-04T12:00:00Z".into()),
            authority_epoch: 1,
        };
        store
            .register_lease(&AuthorizationLease::new_with_instance(
                witness.authorization_instance.clone(),
                action.id.clone(),
                digest,
                witness.support_digest.clone(),
                witness.policy.clone(),
                1,
                1,
            ))
            .unwrap();
        store
            .prepare_for_execution_bound(
                &witness,
                &action,
                "frame@1",
                "attempt-native-pin-admission",
                "boundary-native-pin-admission",
            )
            .unwrap();

        struct MutatingNativePinStatusVerifier {
            path: std::path::PathBuf,
        }

        impl ProviderStatusVerifier for MutatingNativePinStatusVerifier {
            fn configuration(&self) -> ProviderStatusVerifierConfiguration {
                TestProviderStatusVerifier.configuration()
            }

            fn verify_current_status(
                &self,
                _purpose: ProviderStatusVerificationPurpose,
                _issuer: &str,
                _authority_namespace: &str,
                native_authorization_id: &str,
                status_identifier: &str,
                _action_digest: &str,
                _target_identity: &str,
                _audience: &str,
                _adapter: &str,
                _expected_source_digest: Option<&str>,
            ) -> Result<ProviderStatusEvidence, ProviderStatusVerificationError> {
                let connection = Connection::open(&self.path)
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                connection
                    .execute(
                        "UPDATE authorization_native_authority_pins
                         SET authority_namespace='issuer.native-admission/tampered'
                         WHERE issuer='issuer.native-admission'",
                        [],
                    )
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                let observed = trusted_utc_now()
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                Ok(ProviderStatusEvidence {
                    status_identifier: status_identifier.to_owned(),
                    status_source_digest: "sha256:test-status-source".into(),
                    status_observed_at: observed.to_rfc3339_opts(SecondsFormat::Secs, true),
                    status_valid_until: (observed + Duration::seconds(1800))
                        .to_rfc3339_opts(SecondsFormat::Secs, true),
                    status_evidence_digest: format!(
                        "sha256:status:{native_authorization_id}"
                    ),
                })
            }
        }

        let err = store
            .mark_dispatch_pending_bound_from_pinned_native_authority(
                &witness.authorization_instance,
                "attempt-native-pin-admission",
                &action,
                &effect,
                "boundary-native-pin-admission",
                "operation:native-pin-admission",
                "issuer.native-admission",
                "native-native-pin-admission",
                "status:native-native-pin-admission",
                &MutatingNativePinStatusVerifier { path: path.clone() },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidNativeReplayProvenance
            )
        ));
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row::<i64, _, _>(
                    "SELECT COUNT(*) FROM authorization_dispatches",
                    [],
                    |row| row.get(0),
                )
                .unwrap(),
            0
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn status_source_pin_drift_after_admission_verification_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-source-admission-drift-{}.db",
            std::process::id()
        ));
        let store =
            SqliteAuthorizationStore::open_with_relying_party(&path, "rp-status-source-admission")
                .unwrap();
        store
            .pin_native_authority_namespace(
                "issuer.source-admission",
                "issuer.source-admission/authority/v1",
            )
            .unwrap();
        store
            .pin_provider_status_source_digest("sha256:test-status-source")
            .unwrap();
        store
            .pin_provider_status_verifier_configuration(
                &TestProviderStatusVerifier.configuration(),
            )
            .unwrap();
        store
            .pin_provider_adapter_configuration(&ProviderAdapterConfiguration::new(
                "adapter-source-admission",
                "test-adapter/v1",
                "sha256:test-adapter-implementation",
            ))
            .unwrap();

        let effect = ActionEffectBinding::new(
            "target-source-admission",
            "prod",
            "adapter-source-admission",
        );
        let action = EpistemicAction::new(
            "status-source-admission",
            "intervention",
            super::super::ActionRisk::Critical,
        )
        .with_effect_binding(effect.clone());
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            operation_id: None,
            action_id: action.id.clone(),
            authorization_instance: "status-source-admission".into(),
            action_digest: digest,
            frame: "frame@1".into(),
            support_digest: "support".into(),
            policy: "policy@1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-03T06:20:00Z".into(),
            expires_at: Some("2026-10-04T12:00:00Z".into()),
            authority_epoch: 1,
        };
        store
            .register_lease(&AuthorizationLease::new_with_instance(
                witness.authorization_instance.clone(),
                action.id.clone(),
                witness.action_digest.clone(),
                witness.support_digest.clone(),
                witness.policy.clone(),
                1,
                1,
            ))
            .unwrap();
        store
            .prepare_for_execution_bound(
                &witness,
                &action,
                "frame@1",
                "attempt-status-source-admission",
                "boundary-source-admission",
            )
            .unwrap();

        struct MutatingAdmissionStatusSourceVerifier {
            path: std::path::PathBuf,
        }

        impl ProviderStatusVerifier for MutatingAdmissionStatusSourceVerifier {
            fn configuration(&self) -> ProviderStatusVerifierConfiguration {
                TestProviderStatusVerifier.configuration()
            }

            fn verify_current_status(
                &self,
                _purpose: ProviderStatusVerificationPurpose,
                _issuer: &str,
                _authority_namespace: &str,
                native_authorization_id: &str,
                status_identifier: &str,
                _action_digest: &str,
                _target_identity: &str,
                _audience: &str,
                _adapter: &str,
                _expected_source_digest: Option<&str>,
            ) -> Result<ProviderStatusEvidence, ProviderStatusVerificationError> {
                let connection = Connection::open(&self.path)
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                connection
                    .execute(
                        "UPDATE authorization_store_metadata
                         SET value='sha256:tampered-status-source'
                         WHERE key='provider_status_source_digest'",
                        [],
                    )
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                let observed = trusted_utc_now()
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                Ok(ProviderStatusEvidence {
                    status_identifier: status_identifier.to_owned(),
                    status_source_digest: "sha256:test-status-source".into(),
                    status_observed_at: observed.to_rfc3339_opts(SecondsFormat::Secs, true),
                    status_valid_until: (observed + Duration::seconds(1800))
                        .to_rfc3339_opts(SecondsFormat::Secs, true),
                    status_evidence_digest: format!(
                        "sha256:status:{native_authorization_id}"
                    ),
                })
            }
        }

        let err = store
            .mark_dispatch_pending_bound_from_pinned_native_authority(
                &witness.authorization_instance,
                "attempt-status-source-admission",
                &action,
                &effect,
                "boundary-source-admission",
                "operation:status-source-admission",
                "issuer.source-admission",
                "native-status-source-admission",
                "status:native-status-source-admission",
                &MutatingAdmissionStatusSourceVerifier { path: path.clone() },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired
            )
        ));
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row::<i64, _, _>(
                    "SELECT COUNT(*) FROM authorization_dispatches",
                    [],
                    |row| row.get(0),
                )
                .unwrap(),
            0
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn status_source_pin_drift_after_pre_entry_verification_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-source-drift-{}.db",
            std::process::id()
        ));
        let store =
            SqliteAuthorizationStore::open_with_relying_party(&path, "rp-status-source-drift")
                .unwrap();
        store
            .pin_native_authority_namespace("issuer.source-drift", "issuer.source-drift/authority/v1")
            .unwrap();
        store
            .pin_provider_status_source_digest("sha256:test-status-source")
            .unwrap();
        store
            .pin_provider_status_verifier_configuration(
                &TestProviderStatusVerifier.configuration(),
            )
            .unwrap();
        store
            .pin_provider_adapter_configuration(&ProviderAdapterConfiguration::new(
                "adapter-source-drift",
                "test-adapter/v1",
                "sha256:test-adapter-implementation",
            ))
            .unwrap();

        let effect = ActionEffectBinding::new(
            "target-source-drift",
            "prod",
            "adapter-source-drift",
        );
        let action = EpistemicAction::new(
            "status-source-drift",
            "intervention",
            super::super::ActionRisk::Critical,
        )
        .with_effect_binding(effect.clone());
        let digest = action.canonical_action_digest();
        let witness = ActionAuthorizationWitness {
            operation_id: None,
            action_id: action.id.clone(),
            authorization_instance: "status-source-drift".into(),
            action_digest: digest.clone(),
            frame: "frame@1".into(),
            support_digest: "support".into(),
            policy: "policy@1".into(),
            decision: "execute".into(),
            issued_at: "2026-10-03T06:20:00Z".into(),
            expires_at: Some("2026-10-04T12:00:00Z".into()),
            authority_epoch: 1,
        };
        store
            .register_lease(&AuthorizationLease::new_with_instance(
                witness.authorization_instance.clone(),
                action.id.clone(),
                digest,
                witness.support_digest.clone(),
                witness.policy.clone(),
                1,
                1,
            ))
            .unwrap();
        store
            .prepare_for_execution_bound(
                &witness,
                &action,
                "frame@1",
                "attempt-status-source-drift",
                "boundary-source-drift",
            )
            .unwrap();

        let record = store
            .mark_dispatch_pending_bound_from_pinned_native_authority(
                &witness.authorization_instance,
                "attempt-status-source-drift",
                &action,
                &effect,
                "boundary-source-drift",
                "operation:status-source-drift",
                "issuer.source-drift",
                "native-status-source-drift",
                "status:native-status-source-drift",
                &TestProviderStatusVerifier,
            )
            .unwrap();

        struct MutatingStatusSourceVerifier {
            path: std::path::PathBuf,
        }

        impl ProviderStatusVerifier for MutatingStatusSourceVerifier {
            fn configuration(&self) -> ProviderStatusVerifierConfiguration {
                TestProviderStatusVerifier.configuration()
            }

            fn verify_current_status(
                &self,
                _purpose: ProviderStatusVerificationPurpose,
                _issuer: &str,
                _authority_namespace: &str,
                native_authorization_id: &str,
                status_identifier: &str,
                _action_digest: &str,
                _target_identity: &str,
                _audience: &str,
                _adapter: &str,
                _expected_source_digest: Option<&str>,
            ) -> Result<ProviderStatusEvidence, ProviderStatusVerificationError> {
                let connection = Connection::open(&self.path)
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                connection
                    .execute(
                        "UPDATE authorization_store_metadata
                         SET value='sha256:tampered-status-source'
                         WHERE key='provider_status_source_digest'",
                        [],
                    )
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;

                let observed = trusted_utc_now()
                    .map_err(|_| ProviderStatusVerificationError::VerificationFailed)?;
                Ok(ProviderStatusEvidence {
                    status_identifier: status_identifier.to_owned(),
                    status_source_digest: "sha256:test-status-source".into(),
                    status_observed_at: observed.to_rfc3339_opts(SecondsFormat::Secs, true),
                    status_valid_until: (observed + Duration::seconds(1800))
                        .to_rfc3339_opts(SecondsFormat::Secs, true),
                    status_evidence_digest: format!(
                        "sha256:status:{native_authorization_id}"
                    ),
                })
            }
        }

        let err = store
            .mark_invoked_bound(
                &record,
                &MutatingStatusSourceVerifier { path: path.clone() },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired
            )
        ));
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row::<String, _, _>(
                    "SELECT state FROM authorization_dispatches
                     WHERE authorization_instance=?1 AND attempt_id=?2",
                    params![
                        record.authorization_instance.as_str(),
                        record.attempt_id.as_str()
                    ],
                    |row| row.get(0),
                )
                .unwrap(),
            "dispatch_pending"
        );
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row::<String, _, _>(
                    "SELECT state FROM authorization_leases WHERE authorization_instance=?1",
                    params![record.authorization_instance.as_str()],
                    |row| row.get(0),
                )
                .unwrap(),
            "dispatch_pending"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn status_lookup_is_not_attempted_before_structural_dispatch_validation() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-ordering-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-status-ordering"
        ).unwrap();
        store.pin_native_authority_namespace(
            "issuer.ordering","issuer.ordering/authority/v1"
        ).unwrap();

        let effect=ActionEffectBinding::new(
            "target-status-ordering","prod","adapter"
        );
        let action=EpistemicAction::new(
            "status-ordering","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"status-ordering".into(),
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
            "status-ordering",action.id.clone(),digest,
            "support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-status-ordering","boundary-status"
        ).unwrap();

        let wrong_effect=ActionEffectBinding::new(
            "target-different","prod","adapter"
        );
        assert!(matches!(
            store.mark_dispatch_pending_bound_from_pinned_native_authority(
                "status-ordering","attempt-status-ordering",
                &action,&wrong_effect,"boundary-status","operation:status-ordering",
                "issuer.ordering","native-status-ordering",
                "status:native-status-ordering",&AdmissionStatusFailsVerifier
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));

        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_leases
             WHERE authorization_instance=?1",
            params![witness.authorization_instance],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"prepared");

        let dispatches:i64=store.connection().unwrap().query_row(
            "SELECT COUNT(*) FROM authorization_dispatches
             WHERE authorization_instance=?1",
            params![witness.authorization_instance],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(dispatches,0);
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn admission_status_failure_releases_prepared_reservation_without_dispatch() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-admission-failure-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-status-admission-failure"
        ).unwrap();
        store.pin_provider_status_source_digest("sha256:test-status-source").unwrap();
        store.pin_provider_status_verifier_configuration(
            &ProviderStatusVerifierConfiguration::new(
                "test-status-verifier/v1","test-status-verifier/rev1",
                "test-status-verifier","sha256:test-status-verifier-implementation",
                "sha256:test-status-verifier-config"
            )
        ).unwrap();
        store.pin_native_authority_namespace(
            "issuer.status-admission","issuer.status-admission/authority/v1"
        ).unwrap();

        let effect=ActionEffectBinding::new(
            "target-status-admission-failure","prod","adapter"
        );
        let action=EpistemicAction::new(
            "status-admission-failure","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"status-admission-failure".into(),
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
            "status-admission-failure",action.id.clone(),digest,
            "support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-status-admission-failure","boundary-status"
        ).unwrap();

        assert!(matches!(
            store.mark_dispatch_pending_bound_from_pinned_native_authority(
                "status-admission-failure","attempt-status-admission-failure",
                &action,&effect,"boundary-status","operation:status-admission-failure",
                "issuer.status-admission","native-status-admission-failure",
                "status:native-status-admission-failure",&AdmissionStatusFailsVerifier
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired
            ))
        ));

        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_leases
             WHERE authorization_instance=?1",
            params![witness.authorization_instance],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"ready");

        let dispatches:i64=store.connection().unwrap().query_row(
            "SELECT COUNT(*) FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![witness.authorization_instance,"attempt-status-admission-failure"],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(dispatches,0);

        let marker:String=store.connection().unwrap().query_row(
            "SELECT marker FROM authorization_recovery_markers
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![witness.authorization_instance,"attempt-status-admission-failure"],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(marker,"not_entered_status");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_entry_status_failure_closes_attempt_without_provider_entry() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-failure-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(
            &path,"rp-status-failure"
        ).unwrap();
        store.pin_provider_status_source_digest("sha256:test-status-source").unwrap();
        store.pin_provider_status_verifier_configuration(
            &ProviderStatusVerifierConfiguration::new(
                "test-status-verifier/v1","test-status-verifier/rev1",
                "test-status-verifier","sha256:test-status-verifier-implementation",
                "sha256:test-status-verifier-config"
            )
        ).unwrap();
        store.pin_native_authority_namespace("issuer.status","issuer.status/authority/v1").unwrap();

        let effect=ActionEffectBinding::new("target-status-failure","prod","adapter");
        let action=EpistemicAction::new(
            "status-failure","intervention",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"status-failure".into(),
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
            "status-failure",action.id.clone(),digest,"support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-status-failure","boundary-status"
        ).unwrap();
        let record=store.mark_dispatch_pending_bound_from_pinned_native_authority(
            "status-failure","attempt-status-failure",&action,&effect,"boundary-status",
            "operation:status-failure","issuer.status","native-status-failure",
            "status:native-status-failure",&TestProviderStatusVerifier
        ).unwrap();

        assert!(matches!(
            store.mark_invoked_bound(&record,&AdmissionStatusFailsVerifier),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::ProviderStatusVerificationRequired
            ))
        ));

        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_dispatches
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"not_entered");

        let marker:String=store.connection().unwrap().query_row(
            "SELECT marker FROM authorization_recovery_markers
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(marker,"not_entered_status");
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn executed_action_instance_remains_closed_to_fresh_authority() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-action-closed-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-closed").unwrap();
        store.pin_provider_status_source_digest("sha256:test-status-source").unwrap();
        store.pin_provider_status_verifier_configuration(&ProviderStatusVerifierConfiguration::new(
            "test-status-verifier/v1","test-status-verifier/rev1",
            "test-status-verifier","sha256:test-status-verifier-implementation",
            "sha256:test-status-verifier-config")).unwrap();
        store.pin_native_authority_namespace("issuer.closed","issuer.closed/authority/v1").unwrap();

        let effect=ActionEffectBinding::new("target-closed","prod","adapter-closed");
        let action=EpistemicAction::new("closed-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();

        let witness_a=ActionAuthorizationWitness {
            operation_id: None,
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
            "operation:closed-a","issuer.closed","native-closed-a",
            "status:native-closed-a",&TestProviderStatusVerifier
        ).unwrap();
        store.mark_invoked_bound(&record_a, &TestProviderStatusVerifier).unwrap();
        store.commit_bound_verified(
            &record_a,
            &verified_evidence(&record_a,ExecutionOutcome::Succeeded),
            &TestProviderVerifierForRp { relying_party_id: store.relying_party_id().to_owned() },
        ).unwrap();

        let witness_b=ActionAuthorizationWitness {
            operation_id: None,
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
                "operation:closed-b","issuer.closed","native-closed-b",
                "status:native-closed-b",&TestProviderStatusVerifier
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");

        let action_a=EpistemicAction::new("native-a","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let action_b=EpistemicAction::new("native-b","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let witness_a=ActionAuthorizationWitness {
            operation_id: None,
            authorization_instance:"native-a".into(), action_id:action_a.id.clone(),
            action_digest:action_a.canonical_action_digest(), support_digest:"support-a".into(),
            frame:"frame@1".into(), policy:"policy@1".into(), authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            operation_id: None,
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
        assert!(matches!(
            duplicate,
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));

        let connection=store.connection().unwrap();
        let state:String=connection.query_row(
            "SELECT state FROM authorization_leases
             WHERE authorization_instance=?1",
            params![witness_b.authorization_instance.as_str()],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(state,"prepared");
        let dispatches:i64=connection.query_row(
            "SELECT COUNT(*) FROM authorization_dispatches
             WHERE native_replay_identity='native-replay:shared'",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(dispatches,1);

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn legacy_transitions_cannot_bypass_boundary_owned_attempts() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-legacy-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("boundary-legacy-fence","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
        store.mark_invoked_bound(&record, &TestProviderStatusVerifier).unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn stale_recovery_authorization_is_rejected_before_state_change() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-recovery-validity-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new(
            "recovery-validity","intervention",super::super::ActionRisk::Critical
        );
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"recovery-validity".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"support".into(),
            policy:"policy@1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(),
            expires_at:Some("2026-10-04T10:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "recovery-validity",action.id.clone(),digest.clone(),"support","policy@1",1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-recovery-validity","boundary-A"
        ).unwrap();

        let stale=RecoveryAuthorizationWitness {
            authorization_instance:witness.authorization_instance.clone(),
            attempt_id:"attempt-recovery-validity".into(),
            operation_id:"".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest.clone(),
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2020-01-01T00:00:00Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&stale),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::AuthorizationValidityWindowFailed
            ))
        ));
        let state:String=store.connection().unwrap().query_row(
            "SELECT state FROM authorization_leases WHERE authorization_instance=?1",
            params![witness.authorization_instance],
            |r| r.get(0)
        ).unwrap();
        assert_eq!(state,"prepared");

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn operation_bound_recovery_requires_matching_operation_identity() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-operation-recovery-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new(
            "operation-recovery","intervention",super::super::ActionRisk::Critical
        );
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"operation-recovery".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(),
            expires_at:Some("2026-10-04T10:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "operation-recovery",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound_with_operation(
            &witness,&action,"frame@1","attempt-operation-recovery","boundary-A","operation-A"
        ).unwrap();

        let missing_operation=RecoveryAuthorizationWitness {
            authorization_instance:witness.authorization_instance.clone(),
            attempt_id:"attempt-operation-recovery".into(),
            operation_id:"".into(),
            boundary_id:"boundary-A".into(),
            action_digest:digest.clone(),
            policy:"recovery-policy-v1".into(),
            authority_epoch:1,
            issued_at:"2026-10-03T10:00:01Z".into(),
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&missing_operation),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));

        let wrong_operation=RecoveryAuthorizationWitness {
            operation_id:"operation-B".into(),
            ..missing_operation.clone()
        };
        assert!(matches!(
            store.recover_pre_dispatch_attempt(&wrong_operation),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));

        let correct=RecoveryAuthorizationWitness {
            operation_id:"operation-A".into(),
            ..missing_operation
        };
        assert!(store.recover_pre_dispatch_attempt(&correct).unwrap());

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn pre_dispatch_recovery_releases_prepared_attempt_and_records_marker() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-pre-recovery-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("pre-recovery","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
            operation_id:"".into(),
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
            &ActionEffectBinding::new("target-A","prod","adapter-A"),
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
            operation_id: None,
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
            operation_id:"".into(),
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
            operation_id:"".into(),
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
            operation_id:"".into(),
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
            operation_id:"".into(),
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("pre-recovery-after-dispatch","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
            operation_id:"".into(),
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
        let effect=ActionEffectBinding::new("target-A","prod","adapter-A");
        let action=EpistemicAction::new("boundary-recovery-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
    fn boundary_bulk_recovery_does_not_convert_prepared_to_indeterminate() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-boundary-bulk-prepared-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new(
            "boundary-bulk-prepared",
            "intervention",
            super::super::ActionRisk::Critical
        );
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"approval-bulk-prepared".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-04T07:10:00Z".into(),
            expires_at:Some("2026-10-05T07:10:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,
            &action,
            "frame@1",
            "attempt-bulk-prepared",
            "boundary-bulk-prepared"
        ).unwrap();

        assert_eq!(
            store.recover_incomplete_attempts_for_boundary("boundary-bulk-prepared").unwrap(),
            0
        );
        assert_eq!(
            store.connection().unwrap().query_row::<String,_,_>(
                "SELECT state FROM authorization_leases
                 WHERE authorization_instance=?1",
                params![witness.authorization_instance.as_str()],
                |row| row.get(0)
            ).unwrap(),
            "prepared"
        );
        assert_eq!(
            store.connection().unwrap().query_row::<i64,_,_>(
                "SELECT COUNT(*) FROM authorization_receipts
                 WHERE authorization_instance=?1 AND attempt_id=?2",
                params![witness.authorization_instance.as_str(),"attempt-bulk-prepared"],
                |row| row.get(0)
            ).unwrap(),
            0
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn boundary_ownership_survives_crash_before_dispatch_record() {
        let path=std::env::temp_dir().join(format!("symthaea-gis-auth-boundary-prepared-{}.db",std::process::id()));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action=EpistemicAction::new("boundary-prepared","intervention",super::super::ActionRisk::Critical);
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
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
            operation_id:"".into(),
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
                &ActionEffectBinding::new("target-A","prod","adapter-A"),
                "boundary-A"
            ,
            format!("operation:{}", "attempt-prepared"),
            format!("native-replay:{}", &witness.authorization_instance)),
            Err(AuthorizationStoreError::Consumption(AuthorizationConsumptionError::AttemptMismatch))
        ));
        let _=std::fs::remove_file(path);
    }


    #[test]
    fn released_attempt_and_operation_ids_cannot_be_reused() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-identity-history-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let attempt_id="attempt-history";
        let operation_id="operation:history";
        store.prepare_for_execution_bound_with_operation(
            &witness,
            &action,
            "frame@1",
            attempt_id,
            "boundary-history",
            operation_id,
        ).unwrap();

        let recovery=RecoveryAuthorizationWitness {
            authorization_instance:witness.authorization_instance.clone(),
            attempt_id:attempt_id.into(),
            operation_id:operation_id.into(),
            boundary_id:"boundary-history".into(),
            action_digest:witness.action_digest.clone(),
            policy:"recovery-policy-v1".into(),
            authority_epoch:witness.authority_epoch,
            issued_at:witness.issued_at.clone(),
        };
        assert!(store.recover_pre_dispatch_attempt(&recovery).unwrap());

        let err=store.prepare_for_execution_bound_with_operation(
            &witness,
            &action,
            "frame@1",
            attempt_id,
            "boundary-history",
            operation_id,
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));

        let new_attempt="attempt-history-new";
        let new_operation="operation:history-new";
        let mut fresh_witness=witness.clone();
        fresh_witness.operation_id=Some(new_operation.into());
        store.prepare_for_execution_bound_with_operation(
            &fresh_witness,
            &action,
            "frame@1",
            new_attempt,
            "boundary-history",
            new_operation,
        ).unwrap();
        assert!(matches!(
            store.prepare_for_execution_bound_with_operation(
                &fresh_witness,
                &action,
                "frame@1",
                "attempt-history-second",
                "boundary-history",
                operation_id,
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));

        let _=std::fs::remove_file(path);
    }

    #[test]
    fn prepared_operation_id_cannot_be_reused_across_authorizations() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-operation-reuse-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let action_a=EpistemicAction::new(
            "operation-a","a",super::super::ActionRisk::Critical
        );
        let action_b=EpistemicAction::new(
            "operation-b","b",super::super::ActionRisk::Critical
        );
        let digest_a=action_a.canonical_action_digest();
        let digest_b=action_b.canonical_action_digest();
        let witness_a=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action_a.id.clone(),
            authorization_instance:"operation-approval-a".into(),
            action_digest:digest_a.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action_b.id.clone(),
            authorization_instance:"operation-approval-b".into(),
            action_digest:digest_b.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:01Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness_a.authorization_instance.clone(),action_a.id.clone(),digest_a,
            witness_a.support_digest.clone(),witness_a.policy.clone(),1,1
        )).unwrap();
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness_b.authorization_instance.clone(),action_b.id.clone(),digest_b,
            witness_b.support_digest.clone(),witness_b.policy.clone(),1,1
        )).unwrap();

        store.prepare_for_execution_bound_with_operation(
            &witness_a,&action_a,"frame@1","attempt-operation-a",
            "boundary-A","operation-shared"
        ).unwrap();
        assert!(matches!(
            store.prepare_for_execution_bound_with_operation(
                &witness_b,&action_b,"frame@1","attempt-operation-b",
                "boundary-B","operation-shared"
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn attempt_scope_digest_is_stable_and_boundary_sensitive() {
        let same_a = compute_attempt_scope_digest("boundary-A","attempt-1").unwrap();
        let same_b = compute_attempt_scope_digest("boundary-A","attempt-1").unwrap();
        let other_boundary = compute_attempt_scope_digest("boundary-B","attempt-1").unwrap();
        let other_attempt = compute_attempt_scope_digest("boundary-A","attempt-2").unwrap();
        assert_eq!(same_a, same_b);
        assert_ne!(same_a, other_boundary);
        assert_ne!(same_a, other_attempt);
        assert!(same_a.starts_with("sha256:"));
    }


    #[test]
    fn status_check_boundary_scope_is_repaired_from_authoritative_dispatch() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-status-scope-backfill-{}.db",std::process::id()
        ));
        {
            let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-status").unwrap();
            let conn=store.connection().unwrap();
            conn.execute(
                "INSERT INTO authorization_dispatches(
                    authorization_instance,attempt_id,operation_id,native_replay_identity,
                    action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,
                    adapter_revision,adapter_implementation_digest,boundary_id,
                    attempt_binding_digest,state)
                 VALUES(
                    'auth-status','attempt-status','op-status','replay-status',
                    'action-status','digest-status','provider-status','target-status','aud-status','adapter-status',
                    'adapter/v1','sha256:impl','boundary-authoritative',
                    'sha256:binding','dispatch_pending'
                 )",
                [],
            ).unwrap();
            conn.execute(
                "INSERT INTO authorization_status_checks(
                    authorization_instance,attempt_id,phase,status_identifier,status_source_digest,
                    status_observed_at,status_valid_until,status_evidence_digest)
                 VALUES(
                    'auth-status','attempt-status','pre_entry','status:attempt-status','sha256:source',
                    '2026-10-03T10:00:00Z','2026-10-03T10:30:00Z','sha256:evidence'
                 )",
                [],
            ).unwrap();
            conn.execute(
                "INSERT INTO authorization_receipts(
                    authorization_instance,action_id,attempt_id,phase,outcome,action_digest,
                    authority_epoch,provider_idempotency_key)
                 VALUES(
                    'auth-status','action-status','attempt-status','indeterminate','indeterminate',
                    'digest-status',1,'provider-status'
                 )",
                [],
            ).unwrap();
        }

        let reopened=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-status").unwrap();
        let conn=reopened.connection().unwrap();
        let (boundary,scope,operation_id):(String,String,String)=conn.query_row(
            "SELECT boundary_id,attempt_scope_digest,operation_id
             FROM authorization_status_checks
             WHERE authorization_instance='auth-status' AND attempt_id='attempt-status'",
            [],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
        ).unwrap();
        let receipt_operation_id: String = conn.query_row(
            "SELECT operation_id
             FROM authorization_receipts
             WHERE authorization_instance='auth-status' AND attempt_id='attempt-status' AND phase='indeterminate'",
            [],
            |row| row.get(0)
        ).unwrap();
        assert_eq!(receipt_operation_id,"op-status");
        assert_eq!(boundary,"boundary-authoritative");
        assert_eq!(
            scope,
            compute_attempt_scope_digest("boundary-authoritative","attempt-status").unwrap()
        );
        assert_eq!(operation_id,"op-status");

        conn.execute(
            "UPDATE authorization_status_checks
             SET operation_id='op-forged'
             WHERE authorization_instance='auth-status' AND attempt_id='attempt-status' AND phase='pre_entry'",
            [],
        ).unwrap();
        drop(conn);
        drop(reopened);
        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party(&path,"rp-status"),
            Err(AuthorizationStoreError::InvalidState(message))
                if message.contains("attempt operation mismatch")
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn startup_rejects_forged_persisted_attempt_scope() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-attempt-scope-startup-{}.db",std::process::id()
        ));
        {
            let store=SqliteAuthorizationStore::open(&path).unwrap();
            store.connection().unwrap().execute(
                "INSERT INTO authorization_leases(
                    authorization_instance,action_id,action_digest,support_digest,policy,
                    authority_epoch,remaining_executions,state,attempt_id,operation_id,
                    boundary_id,attempt_scope_digest
                 ) VALUES(
                    'startup-auth','startup-action','startup-digest','startup-support','startup-policy',
                    1,1,'prepared','startup-attempt','startup-operation','startup-boundary','sha256:forged-scope'
                 )",
                [],
            ).unwrap();
        }
        assert!(matches!(
            SqliteAuthorizationStore::open(&path),
            Err(AuthorizationStoreError::InvalidState(message))
                if message.contains("attempt scope digest mismatch")
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn cross_table_attempt_boundary_splice_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-cross-table-boundary-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-cross-table").unwrap();
        let effect=ActionEffectBinding::new(
            "target-cross","audience-cross","adapter-cross"
        );
        let action=EpistemicAction::new(
            "cross-table-action","effect",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"approval-cross-table".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-cross-table",&action,&effect,
            "boundary-A","operation-cross-table","native-cross-table"
        ).unwrap();

        let boundary_b_scope=
            compute_attempt_scope_digest("boundary-B",&record.attempt_id).unwrap();
        store.connection().unwrap().execute(
            "INSERT INTO authorization_receipts(
                authorization_instance,action_id,attempt_id,phase,outcome,action_digest,
                authority_epoch,provider_idempotency_key,boundary_id,attempt_scope_digest
             ) VALUES(?1,?2,?3,'indeterminate','indeterminate',?4,?5,?6,?7,?8)",
            params![
                record.authorization_instance,
                record.action_id,
                record.attempt_id,
                record.action_digest,
                witness.authority_epoch as i64,
                record.provider_idempotency_key,
                "boundary-B",
                boundary_b_scope,
            ],
        ).unwrap();

        let mut connection=store.connection().unwrap();
        let tx=connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        let err=store.validate_persisted_dispatch_record(&tx,&record).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        tx.rollback().unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn startup_rejects_cross_table_attempt_boundary_splice() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-cross-table-startup-{}.db",std::process::id()
        ));
        {
            let store=SqliteAuthorizationStore::open_with_relying_party(
                &path,"rp-cross-table-startup"
            ).unwrap();
            let effect=ActionEffectBinding::new(
                "target-cross-startup","audience-cross-startup","adapter-cross-startup"
            );
            let action=EpistemicAction::new(
                "cross-table-startup-action",
                "effect",
                super::super::ActionRisk::Critical,
            ).with_effect_binding(effect.clone());
            let digest=action.canonical_action_digest();
            let witness=ActionAuthorizationWitness {
                operation_id: None,
                action_id:action.id.clone(),
                authorization_instance:"approval-cross-table-startup".into(),
                action_digest:digest.clone(),
                frame:"frame@1".into(),
                support_digest:"sha256:support".into(),
                policy:"policy-v1".into(),
                decision:"execute".into(),
                issued_at:"2026-10-03T10:00:00Z".into(),
                expires_at:Some("2026-10-04T12:00:00Z".into()),
                authority_epoch:1,
            };
            store.register_lease(&AuthorizationLease::new_with_instance(
                witness.authorization_instance.clone(),
                action.id.clone(),
                digest,
                witness.support_digest.clone(),
                witness.policy.clone(),
                1,
                1,
            )).unwrap();
            let record=mark_dispatch_pending_bound_for_test(
                &store,
                &witness.authorization_instance,
                "attempt-cross-table-startup",
                &action,
                &effect,
                "boundary-A",
                "operation-cross-table-startup",
                "native-cross-table-startup",
            ).unwrap();

            let boundary_b_scope=
                compute_attempt_scope_digest("boundary-B",&record.attempt_id).unwrap();
            store.connection().unwrap().execute(
                "INSERT INTO authorization_receipts(
                    authorization_instance,action_id,attempt_id,phase,outcome,action_digest,
                    authority_epoch,provider_idempotency_key,boundary_id,attempt_scope_digest
                 ) VALUES(?1,?2,?3,'indeterminate','indeterminate',?4,?5,?6,?7,?8)",
                params![
                    record.authorization_instance,
                    record.action_id,
                    record.attempt_id,
                    record.action_digest,
                    witness.authority_epoch as i64,
                    record.provider_idempotency_key,
                    "boundary-B",
                    boundary_b_scope,
                ],
            ).unwrap();
        }

        assert!(matches!(
            SqliteAuthorizationStore::open_with_relying_party(
                &path,
                "rp-cross-table-startup",
            ),
            Err(AuthorizationStoreError::InvalidState(message))
                if message.contains("attempt boundary mismatch")
        ));
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn persisted_attempt_scope_tampering_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-attempt-scope-tamper-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-scope").unwrap();
        let effect=ActionEffectBinding::new("target-scope","prod","adapter-scope");
        let action=EpistemicAction::new("attempt-scope-action","effect",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(), authorization_instance:"approval-scope".into(),
            action_digest:digest.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "approval-scope",action.id.clone(),digest.clone(),"sha256:support","policy-v1",1,1
        )).unwrap();
        store.prepare_for_execution_bound_with_operation(
            &witness,&action,"frame@1","attempt-scope","boundary-scope","operation-scope"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,"approval-scope","attempt-scope",&action,&effect,"boundary-scope",
            "operation-scope","native-scope"
        ).unwrap();

        let mut connection=store.connection().unwrap();
        let tx=connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        tx.execute(
            "UPDATE authorization_dispatches
             SET attempt_scope_digest='sha256:tampered-scope'
             WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
            params![
                record.authorization_instance,
                record.attempt_id,
                record.boundary_id,
            ],
        ).unwrap();
        let err=store.validate_persisted_dispatch_record(&tx,&record).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding)
        ));
        tx.rollback().unwrap();
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
            operation_id: None,
            action_id:action_a.id.clone(), authorization_instance:"scope-approval-a".into(),
            action_digest:digest_a.clone(), frame:"frame@1".into(),
            support_digest:"sha256:support".into(), policy:"policy-v1".into(), decision:"execute".into(),
            issued_at:"2026-10-02T20:11:00Z".into(), expires_at:Some("2026-10-04T12:00:00Z".into()), authority_epoch:1,
        };
        let witness_b=ActionAuthorizationWitness {
            operation_id: None,
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
    fn crash_recovery_rejects_tampered_bound_dispatch_provenance() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-recovery-provenance-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open_with_relying_party(&path,"rp-recovery").unwrap();
        let effect=ActionEffectBinding::new("target-recovery","prod","adapter-recovery");
        let action=EpistemicAction::new(
            "recovery-provenance-action","effect",super::super::ActionRisk::Critical
        ).with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"recovery-provenance-approval".into(),
            action_digest:digest.clone(),
            frame:"frame@1".into(),
            support_digest:"sha256:support".into(),
            policy:"policy-v1".into(),
            decision:"execute".into(),
            issued_at:"2026-10-03T10:00:00Z".into(),
            expires_at:Some("2026-10-04T12:00:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            "recovery-provenance-approval",action.id.clone(),digest.clone(),
            "sha256:support","policy-v1",1,1
        )).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,"recovery-provenance-approval","attempt-recovery",
            &action,&effect,"boundary-recovery","operation-recovery","native-recovery"
        ).unwrap();

        {
            let mut connection=store.connection().unwrap();
            let tx=connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
            tx.execute(
                "UPDATE authorization_dispatches
                 SET provider_idempotency_key='forged-provider-key'
                 WHERE authorization_instance=?1 AND attempt_id=?2 AND boundary_id=?3",
                params![
                    record.authorization_instance,
                    record.attempt_id,
                    record.boundary_id,
                ],
            ).unwrap();
            tx.commit().unwrap();
        }

        let err=store.recover_incomplete_attempts_for_boundary("boundary-recovery").unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(AuthorizationConsumptionError::InvalidBinding)
        ));

        let connection=store.connection().unwrap();
        let state:String=connection.query_row(
            "SELECT state FROM authorization_leases
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(state,"dispatch_pending");
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
        let effect=ActionEffectBinding::new("target-cross-rp","prod","adapter-A");
        let action=EpistemicAction::new("rp-scoped-action","intervention",super::super::ActionRisk::Critical)
            .with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();

        for (store,instance,attempt,boundary,operation,native) in [
            (&store_a,"rp-approval-a","attempt-rp-a","boundary-A","operation-rp-a","native-rp-a"),
            (&store_b,"rp-approval-b","attempt-rp-b","boundary-B","operation-rp-b","native-rp-b"),
        ] {
            let witness=ActionAuthorizationWitness {
                operation_id: None,
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
    #[test]
    fn bound_preparation_rejects_contradictory_witness_operation_metadata() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-operation-witness-consistency-{}.db",
            std::process::id()
        ));
        let (store, action, witness) = fixture(&path);
        let mut contradictory = witness.clone();
        contradictory.operation_id = Some("operation-B".into());

        assert!(matches!(
            store.prepare_for_execution_bound_with_operation(
                &contradictory,
                &action,
                "frame@1",
                "attempt-operation",
                "boundary-A",
                "operation-A",
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));

        let connection = store.connection().unwrap();
        let state: String = connection
            .query_row(
                "SELECT state FROM authorization_leases WHERE authorization_instance=?1",
                params![witness.authorization_instance.as_str()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state, "ready");

        let operation: Option<String> = connection
            .query_row(
                "SELECT operation_id FROM authorization_leases WHERE authorization_instance=?1",
                params![witness.authorization_instance.as_str()],
                |row| row.get(0),
            )
            .unwrap();
        assert!(operation.is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bound_dispatch_rejects_operation_identity_substitution() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-operation-binding-{}.db",
            std::process::id()
        ));
        let (store, action, witness) = fixture(&path);
        let effect = ActionEffectBinding::new(
            "target-operation",
            "prod",
            "adapter-A",
        );
        let action = action.with_effect_binding(effect.clone());
        let witness = ActionAuthorizationWitness {
            operation_id: None,
            action_id: action.id.clone(),
            authorization_instance: witness.authorization_instance.clone(),
            action_digest: action.canonical_action_digest(),
            frame: witness.frame,
            support_digest: witness.support_digest,
            policy: witness.policy,
            decision: witness.decision,
            issued_at: witness.issued_at,
            expires_at: witness.expires_at,
            authority_epoch: witness.authority_epoch,
        };
        store.prepare_for_execution_bound_with_operation(
            &witness,
            &action,
            "frame@1",
            "attempt-operation",
            "boundary-A",
            "operation-A",
        ).unwrap();
        assert!(matches!(
            store.validate_bound_dispatch_preconditions(
                &witness.authorization_instance,
                "attempt-operation",
                &action,
                &effect,
                "boundary-A",
                "operation-B",
                "https://issuer.example",
                "native-auth-1",
                "authority:issuer.example",
            ),
            Err(AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            ))
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bound_receipt_replays_preserve_native_provider_identity() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-receipt-provider-key-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-receipt-provider-key","prod","adapter-receipt"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"receipt-provider-key".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T07:10:00Z".into(),
            expires_at:Some("2026-10-04T07:10:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),action.id.clone(),digest,
            witness.support_digest.clone(),witness.policy.clone(),1,1
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-receipt-provider-key","boundary-receipt"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-receipt-provider-key",
            &action,&effect,"boundary-receipt",
            "operation:receipt-provider-key","native-receipt-provider-key"
        ).unwrap();
        let evidence=verified_evidence(&record,ExecutionOutcome::Succeeded);
        let first=store.commit_bound_verified(
            &record,&evidence,&TestProviderVerifier
        ).unwrap();
        assert_eq!(first.provider_idempotency_key,record.provider_idempotency_key);

        let second=store.commit_bound_verified(
            &record,&evidence,&TestProviderVerifier
        ).unwrap();
        assert_eq!(second.provider_idempotency_key,record.provider_idempotency_key);
        assert_ne!(
            second.provider_idempotency_key,
            AuthorizationLease::new_with_instance(
                record.authorization_instance.clone(),
                record.action_id.clone(),
                record.action_digest.clone(),
                String::new(),
                String::new(),
                1,
                1,
            ).provider_idempotency_key()
        );
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn bound_receipt_identity_tampering_is_rejected_on_read() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-receipt-tamper-{}.db",std::process::id()
        ));
        let (store,action,witness)=fixture(&path);
        let effect=ActionEffectBinding::new(
            "target-receipt-tamper","prod","adapter-receipt-tamper"
        );
        let action=action.with_effect_binding(effect.clone());
        let digest=action.canonical_action_digest();
        let witness=ActionAuthorizationWitness {
            operation_id: None,
            action_id:action.id.clone(),
            authorization_instance:"receipt-tamper".into(),
            action_digest:digest.clone(),
            frame:witness.frame,
            support_digest:witness.support_digest,
            policy:witness.policy,
            decision:"execute".into(),
            issued_at:"2026-10-03T07:20:00Z".into(),
            expires_at:Some("2026-10-04T07:20:00Z".into()),
            authority_epoch:1,
        };
        store.register_lease(&AuthorizationLease::new_with_instance(
            witness.authorization_instance.clone(),
            action.id.clone(),
            digest,
            witness.support_digest.clone(),
            witness.policy.clone(),
            1,
            1,
        )).unwrap();
        store.prepare_for_execution_bound(
            &witness,&action,"frame@1","attempt-receipt-tamper","boundary-receipt-tamper"
        ).unwrap();
        let record=mark_dispatch_pending_bound_for_test(
            &store,&witness.authorization_instance,"attempt-receipt-tamper",
            &action,&effect,"boundary-receipt-tamper",
            "operation:receipt-tamper","native-receipt-tamper"
        ).unwrap();

        let tampered_scope=compute_attempt_scope_digest(
            &record.boundary_id,&record.attempt_id
        ).unwrap();
        store.connection().unwrap().execute(
            "INSERT INTO authorization_receipts(
                authorization_instance,action_id,attempt_id,phase,outcome,action_digest,
                authority_epoch,provider_idempotency_key,boundary_id,attempt_scope_digest
             ) VALUES(?1,?2,?3,'final','succeeded',?4,?5,?6,?7,?8)",
            params![
                record.authorization_instance,
                record.action_id,
                record.attempt_id,
                "forged-action-digest",
                witness.authority_epoch as i64,
                record.provider_idempotency_key,
                record.boundary_id,
                tampered_scope,
            ],
        ).unwrap();

        let mut connection=store.connection().unwrap();
        let tx=connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        let err=load_receipt(
            &tx,
            &record.authorization_instance,
            &record.attempt_id,
            "final",
        ).unwrap_err();
        assert!(matches!(
            err,
            AuthorizationStoreError::Consumption(
                AuthorizationConsumptionError::InvalidBinding
            )
        ));
        tx.rollback().unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn terminal_record_splicing_is_rejected_before_settlement() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-gis-record-splice-{}.db",
            std::process::id()
        ));
        let store = SqliteAuthorizationStore::open_with_relying_party(&path, "rp-test").unwrap();
        let mut connection = store.connection().unwrap();
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        tx.execute(
            "INSERT INTO authorization_dispatches
             (authorization_instance,attempt_id,operation_id,native_replay_identity,
              native_issuer,native_authority_namespace,native_authorization_id,
              action_id,action_digest,provider_idempotency_key,target_identity,audience,adapter,
              boundary_id,state,relying_party_id)
             VALUES ('auth','attempt','op-A','replay-A','issuer','ns','native-A',
                     'action','digest','provider-key-A','target-A','audience-A','adapter-A',
                     'boundary','invoked','rp-test')",
            [],
        ).unwrap();

        let record = DurableDispatchRecord {
            authorization_instance: "auth".into(),
            attempt_id: "attempt".into(),
            operation_id: "op-B".into(),
            native_replay_identity: "replay-A".into(),
            native_issuer: "issuer".into(),
            native_authority_namespace: "ns".into(),
            native_authorization_id: "native-A".into(),
            native_replay_derivation_digest: "derivation".into(),
            status_identifier: "status".into(),
            status_source_digest: "source".into(),
            status_observed_at: "2026-10-03T10:00:00Z".into(),
            status_valid_until: "2026-10-03T11:00:00Z".into(),
            status_evidence_digest: "status-digest".into(),
            action_id: "action".into(),
            action_digest: "digest".into(),
            provider_idempotency_key: "provider-key-A".into(),
            target_identity: "target-A".into(),
            audience: "audience-A".into(),
            adapter: "adapter-A".into(),
            adapter_revision: "test-adapter/v1".into(),
            adapter_implementation_digest: "sha256:test-adapter-implementation".into(),
            boundary_id: "boundary".into(),
            attempt_binding_digest: "sha256:forged-binding".into(),
        };

        let err = store.validate_persisted_dispatch_record(&tx, &record).unwrap_err();
        assert!(matches!(err, AuthorizationStoreError::Consumption(
            AuthorizationConsumptionError::InvalidBinding
        )));
        tx.rollback().unwrap();
        let _ = std::fs::remove_file(path);
    }

}
    #[test]
    fn persisted_status_tampering_is_rejected_by_attempt_binding() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-status-binding-tamper-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let record=DurableDispatchRecord {
            authorization_instance:"auth-status".into(),
            attempt_id:"attempt-status".into(),
            operation_id:"op-status".into(),
            native_replay_identity:"replay-status".into(),
            native_issuer:"issuer-status".into(),
            native_authority_namespace:"ns-status".into(),
            native_authorization_id:"native-status".into(),
            native_replay_derivation_digest:"derivation-status".into(),
            status_identifier:"status-A".into(),
            status_source_digest:"source-A".into(),
            status_observed_at:"2026-10-03T10:00:00Z".into(),
            status_valid_until:"2026-10-03T11:00:00Z".into(),
            status_evidence_digest:"evidence-A".into(),
            action_id:"action-status".into(),
            action_digest:"digest-status".into(),
            provider_idempotency_key:"provider-status".into(),
            target_identity:"target-status".into(),
            audience:"audience-status".into(),
            adapter:"adapter-status".into(),
            adapter_revision:"test-adapter/v1".into(),
            adapter_implementation_digest:"sha256:test-adapter-implementation".into(),
            boundary_id:"boundary-status".into(),
            attempt_binding_digest:"sha256:binding-status".into(),
        };
        let mut connection=store.connection().unwrap();
        let tx=connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        tx.execute(
            "INSERT INTO authorization_dispatches(
                authorization_instance,attempt_id,operation_id,native_replay_identity,
                native_issuer,native_authority_namespace,native_authorization_id,
                native_replay_derivation_digest,native_authority_pin_set_id,
                native_authority_pin_set_digest,relying_party_id,action_id,action_digest,
                provider_idempotency_key,target_identity,audience,adapter,boundary_id,
                attempt_binding_digest,status_identifier,status_source_digest,status_observed_at,
                status_valid_until,status_evidence_digest,validity_issued_at,validity_expires_at,
                validity_policy_digest,state
             ) VALUES(
                ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,
                ?19,?20,?21,?22,?23,?24,?25,?26,?27,?28
             )",
            params![
                record.authorization_instance,record.attempt_id,record.operation_id,
                record.native_replay_identity,record.native_issuer,record.native_authority_namespace,
                record.native_authorization_id,"derivation-status","pinset-status","digest-status",
                store.relying_party_id(),record.action_id,record.action_digest,record.provider_idempotency_key,
                record.target_identity,record.audience,record.adapter,record.boundary_id,
                record.attempt_binding_digest,record.status_identifier,record.status_source_digest,
                record.status_observed_at,record.status_valid_until,record.status_evidence_digest,
                "",Option::<String>::None,"", "dispatch_pending"
            ],
        ).unwrap();

        tx.execute(
            "UPDATE authorization_dispatches
             SET status_evidence_digest='tampered-evidence'
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();

        let err=store.validate_persisted_dispatch_record(&tx,&record).unwrap_err();
        assert!(matches!(err,AuthorizationStoreError::Consumption(
            AuthorizationConsumptionError::InvalidBinding
        )));
        tx.rollback().unwrap();
        let _=std::fs::remove_file(path);
    }

    #[test]
    fn persisted_attempt_binding_tampering_is_rejected() {
        let path=std::env::temp_dir().join(format!(
            "symthaea-gis-auth-attempt-binding-tamper-{}.db",std::process::id()
        ));
        let store=SqliteAuthorizationStore::open(&path).unwrap();
        let effect=ActionEffectBinding::new("target-A","audience-A","adapter-A");
        let record=DurableDispatchRecord {
            authorization_instance:"auth".into(),
            attempt_id:"attempt".into(),
            operation_id:"op-A".into(),
            native_replay_identity:"replay-A".into(),
            native_issuer:"issuer".into(),
            native_authority_namespace:"ns".into(),
            native_authorization_id:"native-A".into(),
            status_identifier:"status-A".into(),
            status_source_digest:"source-A".into(),
            status_observed_at:"2026-10-03T10:00:00Z".into(),
            status_valid_until:"2026-10-03T11:00:00Z".into(),
            status_evidence_digest:"status-digest-A".into(),
            action_id:"action-A".into(),
            action_digest:"digest-A".into(),
            provider_idempotency_key:"provider-key-A".into(),
            target_identity:effect.target_identity.clone(),
            audience:effect.audience.clone(),
            adapter:effect.adapter.clone(),
            adapter_revision:"test-adapter/v1".into(),
            adapter_implementation_digest:"sha256:test-adapter-implementation".into(),
            boundary_id:"boundary-A".into(),
            attempt_binding_digest:"sha256:uncomputed".into(),
        };

        let mut connection=store.connection().unwrap();
        let tx=connection.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        tx.execute(
            "INSERT INTO authorization_dispatches(
                authorization_instance,attempt_id,operation_id,native_replay_identity,
                native_issuer,native_authority_namespace,native_authorization_id,
                native_replay_derivation_digest,native_authority_pin_set_id,
                native_authority_pin_set_digest,relying_party_id,action_id,action_digest,
                provider_idempotency_key,target_identity,audience,adapter,boundary_id,
                attempt_binding_digest,state
             ) VALUES(
                ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20
             )",
            params![
                record.authorization_instance,record.attempt_id,record.operation_id,
                record.native_replay_identity,record.native_issuer,record.native_authority_namespace,
                record.native_authorization_id,"derivation-A","pinset-A","digest-A",
                store.relying_party_id(),record.action_id,record.action_digest,
                record.provider_idempotency_key,record.target_identity,record.audience,record.adapter,
                record.boundary_id,record.attempt_binding_digest,"dispatch_pending"
            ],
        ).unwrap();

        let digest=compute_attempt_binding_digest(
            &record.authorization_instance,&record.attempt_id,&record.operation_id,
            &record.native_replay_identity,&record.native_issuer,&record.native_authority_namespace,
            &record.native_authorization_id,"derivation-A","pinset-A","digest-A",
            store.relying_party_id(),&record.action_id,&record.action_digest,
            &record.provider_idempotency_key,&record.target_identity,&record.audience,&record.adapter,
            &record.status_identifier,&record.status_source_digest,&record.status_observed_at,
            &record.status_valid_until,&record.status_evidence_digest,"",None,""
        );

        tx.execute(
            "UPDATE authorization_dispatches
             SET attempt_binding_digest=?3
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id,digest],
        ).unwrap();

        tx.execute(
            "UPDATE authorization_dispatches SET target_identity='target-B'
             WHERE authorization_instance=?1 AND attempt_id=?2",
            params![record.authorization_instance,record.attempt_id],
        ).unwrap();

        let err=store.validate_persisted_dispatch_record(&tx,&DurableDispatchRecord {
            target_identity:"target-A".into(),
            attempt_binding_digest:digest,
            ..record
        }).unwrap_err();
        assert!(matches!(err, AuthorizationStoreError::Consumption(
            AuthorizationConsumptionError::InvalidBinding
        )));
        tx.rollback().unwrap();
        let _=std::fs::remove_file(path);
}