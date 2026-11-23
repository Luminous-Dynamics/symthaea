// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Small reference kernel for adversarial memory lifecycle qualification.
//!
//! This module is intentionally not a production memory store. Its security
//! properties depend on an externally authenticated IdentityAuthority and an
//! independently durable TrustedWitness. SHA-256 is used for commitments only;
//! it does not authenticate issuers or establish semantic truth.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sha2::{Digest, Sha256};

/// Permission profile returned by an external identity/authority verifier.
///
/// The caller must not construct this profile from untrusted record fields.
/// A production adapter should derive it from an authenticated issuer registry,
/// credential status and a policy version pinned to the decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorityProfile {
    pub allowed_scopes: BTreeSet<String>,
    pub allowed_purposes: BTreeSet<String>,
    pub may_assert: bool,
    pub may_derive: bool,
    pub may_revoke: bool,
    pub may_erase: bool,
}

impl AuthorityProfile {
    fn permits(&self, scope: &str, purpose: &str) -> bool {
        self.allowed_scopes.contains(scope) && self.allowed_purposes.contains(purpose)
    }
}

/// Trusted identity/authority lookup supplied by the embedding application.
///
/// This trait is a trust boundary, not a cryptographic implementation. A fake
/// implementation can forge profiles; production callers must bind it to
/// independently verified credentials and a versioned authorization policy.
pub trait IdentityAuthority {
    fn profile_for(&self, issuer_id: &str) -> Option<AuthorityProfile>;
}

/// The scope and intended decision purpose for which evidence is requested.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionNeed {
    pub scope: String,
    pub purpose: String,
}

impl DecisionNeed {
    pub fn new(scope: impl Into<String>, purpose: impl Into<String>) -> Self {
        Self {
            scope: scope.into(),
            purpose: purpose.into(),
        }
    }
}

/// Immutable evidence record. Parent IDs are required derivation dependencies.
///
/// A matching payload digest is only a byte-integrity check. It does not
/// establish that the statement is true, that its issuer is authentic, or that
/// the issuer had authority to make the statement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryRecord {
    pub id: String,
    pub payload: String,
    pub payload_sha256: String,
    pub issuer_id: String,
    pub scope: String,
    pub purpose: String,
    pub parents: Vec<String>,
}

impl MemoryRecord {
    pub fn new(
        id: impl Into<String>,
        payload: impl Into<String>,
        issuer_id: impl Into<String>,
        scope: impl Into<String>,
        purpose: impl Into<String>,
        mut parents: Vec<String>,
    ) -> Self {
        parents.sort();
        Self {
            id: id.into(),
            payload: payload.into(),
            payload_sha256: String::new(),
            issuer_id: issuer_id.into(),
            scope: scope.into(),
            purpose: purpose.into(),
            parents,
        }
        .with_computed_digest()
    }

    fn with_computed_digest(mut self) -> Self {
        self.payload_sha256 = sha256_hex(self.payload.as_bytes());
        self
    }
}

/// Cloneable working-store snapshot. It deliberately does not contain the
/// TrustedWitness, so restoring a snapshot cannot also roll back the witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySnapshot {
    generation: u64,
    records: BTreeMap<String, MemoryRecord>,
}

impl Default for MemorySnapshot {
    fn default() -> Self {
        Self {
            generation: 0,
            records: BTreeMap::new(),
        }
    }
}

impl MemorySnapshot {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn contains(&self, record_id: &str) -> bool {
        self.records.contains_key(record_id)
    }

    pub fn record_ids(&self) -> Vec<String> {
        self.records.keys().cloned().collect()
    }

    /// Canonical commitment over generation and all stored record fields.
    pub fn head_sha256(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.generation.to_be_bytes());
        for (key, record) in &self.records {
            append_field(&mut bytes, key);
            append_field(&mut bytes, &record.id);
            append_field(&mut bytes, &record.payload);
            append_field(&mut bytes, &record.payload_sha256);
            append_field(&mut bytes, &record.issuer_id);
            append_field(&mut bytes, &record.scope);
            append_field(&mut bytes, &record.purpose);
            bytes.extend_from_slice(&(record.parents.len() as u64).to_be_bytes());
            for parent in &record.parents {
                append_field(&mut bytes, parent);
            }
        }
        sha256_hex(&bytes)
    }
}

/// Externally persisted monotonic witness.
///
/// This value must live outside restorable memory snapshots. The crate models
/// the API contract only; it does not protect this object from rollback,
/// impersonation, deletion, or storage corruption.
#[derive(Debug)]
pub struct TrustedWitness {
    generation: u64,
    store_head_sha256: String,
    revoked_ids: BTreeSet<String>,
    erased_ids: BTreeSet<String>,
}

impl Default for TrustedWitness {
    fn default() -> Self {
        let empty = MemorySnapshot::default();
        Self {
            generation: 0,
            store_head_sha256: empty.head_sha256(),
            revoked_ids: BTreeSet::new(),
            erased_ids: BTreeSet::new(),
        }
    }
}

impl TrustedWitness {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn revoked_ids(&self) -> &BTreeSet<String> {
        &self.revoked_ids
    }

    pub fn erased_ids(&self) -> &BTreeSet<String> {
        &self.erased_ids
    }
}

/// Fail-closed reference-kernel errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LifecycleError {
    EmptyIdentity,
    DuplicateRecord,
    MissingRecord,
    MissingParent,
    ParentCycle,
    NonCanonicalParents,
    ParentUseMismatch,
    PayloadDigestMismatch,
    UnknownIssuer,
    ScopeDenied,
    PurposeDenied,
    IssuerCannotAssert,
    IssuerCannotDerive,
    RevokerNotAuthorized,
    EraserNotAuthorized,
    AlreadyInvalidated,
    StoreHeadMismatch,
    SnapshotAheadOfWitness,
    RollbackOrTamperDetected,
    GenerationOverflow,
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LifecycleError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidationReport {
    pub invalidated_ids: Vec<String>,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveryReport {
    pub removed_resurrected_ids: Vec<String>,
    pub removed_orphaned_ids: Vec<String>,
    pub generation: u64,
    pub recovered_head_sha256: String,
}

/// Commit a new record after validating its identity profile and every parent.
///
/// Parent evidence must already exist and be eligible for the same scope and
/// purpose. This prevents a derived record from laundering an inadmissible
/// parent into a newly trusted representation.
pub fn commit_record(
    snapshot: &mut MemorySnapshot,
    witness: &mut TrustedWitness,
    record: MemoryRecord,
    authority: &dyn IdentityAuthority,
) -> Result<(), LifecycleError> {
    verify_store_head(snapshot, witness)?;
    if record.id.is_empty() || record.issuer_id.is_empty() {
        return Err(LifecycleError::EmptyIdentity);
    }
    if snapshot.records.contains_key(&record.id) {
        return Err(LifecycleError::DuplicateRecord);
    }
    if witness.erased_ids.contains(&record.id) {
        return Err(LifecycleError::AlreadyInvalidated);
    }
    if witness.revoked_ids.contains(&record.id) {
        return Err(LifecycleError::AlreadyInvalidated);
    }
    validate_record_shape(&record)?;
    let need = DecisionNeed::new(record.scope.clone(), record.purpose.clone());
    let profile = authority
        .profile_for(&record.issuer_id)
        .ok_or(LifecycleError::UnknownIssuer)?;
    validate_profile(&profile, &need)?;
    if record.parents.is_empty() {
        if !profile.may_assert {
            return Err(LifecycleError::IssuerCannotAssert);
        }
    } else {
        if !profile.may_derive {
            return Err(LifecycleError::IssuerCannotDerive);
        }
        let mut visiting = BTreeSet::new();
        for parent_id in &record.parents {
            let parent = snapshot
                .records
                .get(parent_id)
                .ok_or(LifecycleError::MissingParent)?;
            if parent.scope != record.scope || parent.purpose != record.purpose {
                return Err(LifecycleError::ParentUseMismatch);
            }
            verify_record_inner(
                snapshot,
                witness,
                authority,
                &need,
                parent_id,
                &mut visiting,
            )?;
        }
    }
    snapshot.records.insert(record.id.clone(), record);
    advance_head(snapshot, witness)
}

/// Verify that a record is usable for the exact requested need and current
/// externally witnessed store state.
pub fn verify_for_use(
    snapshot: &MemorySnapshot,
    witness: &TrustedWitness,
    authority: &dyn IdentityAuthority,
    need: &DecisionNeed,
    record_id: &str,
) -> Result<(), LifecycleError> {
    verify_store_head(snapshot, witness)?;
    let mut visiting = BTreeSet::new();
    verify_record_inner(snapshot, witness, authority, need, record_id, &mut visiting)
}

/// Revoke one record. Descendants remain in the store for auditability but fail
/// current-at-use recursively whenever they depend on the revoked parent.
pub fn revoke_record(
    snapshot: &mut MemorySnapshot,
    witness: &mut TrustedWitness,
    authority: &dyn IdentityAuthority,
    revoker_id: &str,
    record_id: &str,
) -> Result<InvalidationReport, LifecycleError> {
    verify_store_head(snapshot, witness)?;
    let record = snapshot
        .records
        .get(record_id)
        .ok_or(LifecycleError::MissingRecord)?;
    let profile = authority
        .profile_for(revoker_id)
        .ok_or(LifecycleError::UnknownIssuer)?;
    let need = DecisionNeed::new(record.scope.clone(), record.purpose.clone());
    validate_profile(&profile, &need)?;
    if !profile.may_revoke {
        return Err(LifecycleError::RevokerNotAuthorized);
    }
    if witness.revoked_ids.contains(record_id) {
        return Ok(InvalidationReport {
            invalidated_ids: vec![record_id.to_owned()],
            generation: witness.generation,
        });
    }
    witness.revoked_ids.insert(record_id.to_owned());
    let descendants = descendant_closure(snapshot, record_id);
    advance_head(snapshot, witness)?;
    Ok(InvalidationReport {
        invalidated_ids: descendants.into_iter().collect(),
        generation: witness.generation,
    })
}

/// Erase a record and its complete derived-descendant closure. Tombstones are
/// written to the separate witness before the working snapshot is advanced.
pub fn erase_record(
    snapshot: &mut MemorySnapshot,
    witness: &mut TrustedWitness,
    authority: &dyn IdentityAuthority,
    eraser_id: &str,
    record_id: &str,
) -> Result<InvalidationReport, LifecycleError> {
    verify_store_head(snapshot, witness)?;
    if witness.erased_ids.contains(record_id) {
        return Ok(InvalidationReport {
            invalidated_ids: vec![record_id.to_owned()],
            generation: witness.generation,
        });
    }
    let record = snapshot
        .records
        .get(record_id)
        .ok_or(LifecycleError::MissingRecord)?;
    let profile = authority
        .profile_for(eraser_id)
        .ok_or(LifecycleError::UnknownIssuer)?;
    let need = DecisionNeed::new(record.scope.clone(), record.purpose.clone());
    validate_profile(&profile, &need)?;
    if !profile.may_erase {
        return Err(LifecycleError::EraserNotAuthorized);
    }
    let closure = descendant_closure(snapshot, record_id);
    for id in &closure {
        witness.erased_ids.insert(id.clone());
    }
    for id in &closure {
        snapshot.records.remove(id);
    }
    advance_head(snapshot, witness)?;
    Ok(InvalidationReport {
        invalidated_ids: closure.into_iter().collect(),
        generation: witness.generation,
    })
}

/// Reconcile a stale backup against an independently held witness.
///
/// Only a backup whose differences are fully explained by witnessed erasures
/// and whose reconciled canonical head matches the witness is accepted.
/// Missing newer writes, arbitrary byte changes, or unknown orphaning fail
/// closed. This models the T10 erasure-resurrection case; it is not durable
/// erasure without a real external witness implementation.
pub fn reconcile_snapshot(
    mut snapshot: MemorySnapshot,
    witness: &TrustedWitness,
) -> Result<(MemorySnapshot, RecoveryReport), LifecycleError> {
    if snapshot.generation > witness.generation {
        return Err(LifecycleError::SnapshotAheadOfWitness);
    }

    let removed_resurrected_ids: Vec<String> = witness
        .erased_ids
        .iter()
        .filter(|id| snapshot.records.contains_key(*id))
        .cloned()
        .collect();

    for id in &witness.erased_ids {
        snapshot.records.remove(id);
    }

    let mut removed_orphaned_ids = Vec::new();
    loop {
        let orphaned: Vec<String> = snapshot
            .records
            .iter()
            .filter_map(|(id, record)| {
                record
                    .parents
                    .iter()
                    .any(|parent| !snapshot.records.contains_key(parent))
                    .then(|| id.clone())
            })
            .collect();
        if orphaned.is_empty() {
            break;
        }
        for id in orphaned {
            snapshot.records.remove(&id);
            removed_orphaned_ids.push(id);
        }
    }

    snapshot.generation = witness.generation;
    let recovered_head_sha256 = snapshot.head_sha256();
    if recovered_head_sha256 != witness.store_head_sha256 {
        return Err(LifecycleError::RollbackOrTamperDetected);
    }

    let report = RecoveryReport {
        removed_resurrected_ids,
        removed_orphaned_ids,
        generation: witness.generation,
        recovered_head_sha256,
    };
    Ok((snapshot, report))
}

fn validate_record_shape(record: &MemoryRecord) -> Result<(), LifecycleError> {
    if record.id.is_empty() || record.issuer_id.is_empty() {
        return Err(LifecycleError::EmptyIdentity);
    }
    if record.payload_sha256 != sha256_hex(record.payload.as_bytes()) {
        return Err(LifecycleError::PayloadDigestMismatch);
    }
    if record.parents.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(LifecycleError::NonCanonicalParents);
    }
    if record.parents.iter().any(String::is_empty) {
        return Err(LifecycleError::NonCanonicalParents);
    }
    Ok(())
}

fn validate_profile(
    profile: &AuthorityProfile,
    need: &DecisionNeed,
) -> Result<(), LifecycleError> {
    if !profile.allowed_scopes.contains(&need.scope) {
        return Err(LifecycleError::ScopeDenied);
    }
    if !profile.allowed_purposes.contains(&need.purpose) {
        return Err(LifecycleError::PurposeDenied);
    }
    Ok(())
}

fn verify_record_inner(
    snapshot: &MemorySnapshot,
    witness: &TrustedWitness,
    authority: &dyn IdentityAuthority,
    need: &DecisionNeed,
    record_id: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<(), LifecycleError> {
    if witness.erased_ids.contains(record_id) {
        return Err(LifecycleError::AlreadyInvalidated);
    }
    if witness.revoked_ids.contains(record_id) {
        return Err(LifecycleError::AlreadyInvalidated);
    }
    if !visiting.insert(record_id.to_owned()) {
        return Err(LifecycleError::ParentCycle);
    }
    let record = snapshot
        .records
        .get(record_id)
        .ok_or(LifecycleError::MissingRecord)?;
    validate_record_shape(record)?;
    if record.scope != need.scope {
        return Err(LifecycleError::ScopeDenied);
    }
    if record.purpose != need.purpose {
        return Err(LifecycleError::PurposeDenied);
    }
    let profile = authority
        .profile_for(&record.issuer_id)
        .ok_or(LifecycleError::UnknownIssuer)?;
    validate_profile(&profile, need)?;
    if record.parents.is_empty() {
        if !profile.may_assert {
            return Err(LifecycleError::IssuerCannotAssert);
        }
    } else if !profile.may_derive {
        return Err(LifecycleError::IssuerCannotDerive);
    }
    for parent_id in &record.parents {
        let parent = snapshot
            .records
            .get(parent_id)
            .ok_or(LifecycleError::MissingParent)?;
        if parent.scope != record.scope || parent.purpose != record.purpose {
            return Err(LifecycleError::ParentUseMismatch);
        }
        verify_record_inner(snapshot, witness, authority, need, parent_id, visiting)?;
    }
    visiting.remove(record_id);
    Ok(())
}

fn descendant_closure(snapshot: &MemorySnapshot, root_id: &str) -> BTreeSet<String> {
    let mut closure = BTreeSet::from([root_id.to_owned()]);
    loop {
        let before = closure.len();
        for (id, record) in &snapshot.records {
            if record.parents.iter().any(|parent| closure.contains(parent)) {
                closure.insert(id.clone());
            }
        }
        if closure.len() == before {
            return closure;
        }
    }
}

fn verify_store_head(
    snapshot: &MemorySnapshot,
    witness: &TrustedWitness,
) -> Result<(), LifecycleError> {
    if snapshot.generation != witness.generation
        || snapshot.head_sha256() != witness.store_head_sha256
    {
        return Err(LifecycleError::StoreHeadMismatch);
    }
    Ok(())
}

fn advance_head(
    snapshot: &mut MemorySnapshot,
    witness: &mut TrustedWitness,
) -> Result<(), LifecycleError> {
    let generation = snapshot
        .generation
        .checked_add(1)
        .ok_or(LifecycleError::GenerationOverflow)?;
    snapshot.generation = generation;
    witness.generation = generation;
    witness.store_head_sha256 = snapshot.head_sha256();
    Ok(())
}

fn append_field(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FixtureAuthority {
        profiles: BTreeMap<String, AuthorityProfile>,
    }

    impl FixtureAuthority {
        fn allow(&mut self, issuer: &str, profile: AuthorityProfile) {
            self.profiles.insert(issuer.to_owned(), profile);
        }
    }

    impl IdentityAuthority for FixtureAuthority {
        fn profile_for(&self, issuer_id: &str) -> Option<AuthorityProfile> {
            self.profiles.get(issuer_id).cloned()
        }
    }

    fn full_profile() -> AuthorityProfile {
        AuthorityProfile {
            allowed_scopes: BTreeSet::from(["project-x".to_owned()]),
            allowed_purposes: BTreeSet::from(["planning".to_owned()]),
            may_assert: true,
            may_derive: true,
            may_revoke: true,
            may_erase: true,
        }
    }

    fn setup() -> (MemorySnapshot, TrustedWitness, FixtureAuthority) {
        let mut authority = FixtureAuthority::default();
        authority.allow("source-a", full_profile());
        authority.allow("source-b", full_profile());
        authority.allow("reviewer", full_profile());
        authority.allow("eraser", full_profile());
        let snapshot = MemorySnapshot::default();
        let witness = TrustedWitness::default();
        (snapshot, witness, authority)
    }

    fn record(id: &str, issuer: &str, parents: &[&str]) -> MemoryRecord {
        MemoryRecord::new(
            id,
            format!("inert fixture payload for {id}"),
            issuer,
            "project-x",
            "planning",
            parents.iter().map(|value| (*value).to_owned()).collect(),
        )
    }

    fn need() -> DecisionNeed {
        DecisionNeed::new("project-x", "planning")
    }

    #[test]
    fn clean_assertion_is_admitted_for_its_exact_need() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("e1", "source-a", &[]), &authority)
            .expect("valid fixture write");
        assert_eq!(
            verify_for_use(&store, &witness, &authority, &need(), "e1"),
            Ok(())
        );
    }

    #[test]
    fn correct_digest_does_not_authenticate_an_unknown_issuer() {
        let (mut store, mut witness, authority) = setup();
        let result = commit_record(
            &mut store,
            &mut witness,
            record("e1", "attacker-claimed-source", &[]),
            &authority,
        );
        assert_eq!(result, Err(LifecycleError::UnknownIssuer));
        assert!(store.is_empty());
    }

    #[test]
    fn issuer_scope_and_purpose_are_separate_hard_gates() {
        let (mut store, mut witness, mut authority) = setup();
        let mut restricted = full_profile();
        restricted.allowed_scopes = BTreeSet::from(["other-project".to_owned()]);
        authority.allow("source-a", restricted);
        assert_eq!(
            commit_record(
                &mut store,
                &mut witness,
                record("e1", "source-a", &[]),
                &authority
            ),
            Err(LifecycleError::ScopeDenied)
        );
    }

    #[test]
    fn revoking_one_parent_invalidates_multi_parent_descendants() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("a", "source-a", &[]), &authority)
            .expect("parent a");
        commit_record(&mut store, &mut witness, record("b", "source-b", &[]), &authority)
            .expect("parent b");
        commit_record(&mut store, &mut witness, record("c", "source-a", &["a", "b"]), &authority)
            .expect("derived c");
        let report = revoke_record(&mut store, &mut witness, &authority, "reviewer", "a")
            .expect("authorized revocation");
        assert_eq!(
            report.invalidated_ids,
            vec!["a".to_owned(), "c".to_owned()]
        );
        assert_eq!(
            verify_for_use(&store, &witness, &authority, &need(), "c"),
            Err(LifecycleError::AlreadyInvalidated)
        );
        assert_eq!(
            verify_for_use(&store, &witness, &authority, &need(), "b"),
            Ok(())
        );
    }

    #[test]
    fn derived_record_cannot_launder_a_revoked_parent() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("root", "source-a", &[]), &authority)
            .expect("root");
        revoke_record(&mut store, &mut witness, &authority, "reviewer", "root")
            .expect("revoke");
        assert_eq!(
            commit_record(
                &mut store,
                &mut witness,
                record("derived", "source-b", &["root"]),
                &authority
            ),
            Err(LifecycleError::AlreadyInvalidated)
        );
    }

    #[test]
    fn erase_removes_the_complete_derived_descendant_closure() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("root", "source-a", &[]), &authority)
            .expect("root");
        commit_record(
            &mut store,
            &mut witness,
            record("child", "source-b", &["root"]),
            &authority,
        )
        .expect("child");
        commit_record(
            &mut store,
            &mut witness,
            record("grandchild", "source-a", &["child"]),
            &authority,
        )
        .expect("grandchild");
        commit_record(&mut store, &mut witness, record("unrelated", "source-a", &[]), &authority)
            .expect("unrelated");
        let report =
            erase_record(&mut store, &mut witness, &authority, "eraser", "root")
                .expect("erase");
        assert_eq!(
            report.invalidated_ids,
            vec!["child".to_owned(), "grandchild".to_owned(), "root".to_owned()]
        );
        assert!(!store.contains("root"));
        assert!(!store.contains("child"));
        assert!(!store.contains("grandchild"));
        assert!(store.contains("unrelated"));
        assert!(witness.erased_ids().contains("grandchild"));
    }

    #[test]
    fn stale_backup_is_not_served_and_t10_reconciliation_blocks_resurrection() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("root", "source-a", &[]), &authority)
            .expect("root");
        commit_record(
            &mut store,
            &mut witness,
            record("derived", "source-b", &["root"]),
            &authority,
        )
        .expect("derived");
        let stale_backup = store.clone();

        erase_record(&mut store, &mut witness, &authority, "eraser", "root")
            .expect("erase");
        assert_eq!(
            verify_for_use(&stale_backup, &witness, &authority, &need(), "root"),
            Err(LifecycleError::StoreHeadMismatch)
        );

        let (recovered, report) =
            reconcile_snapshot(stale_backup, &witness).expect("witnessed erasure reconciliation");
        assert!(!recovered.contains("root"));
        assert!(!recovered.contains("derived"));
        assert!(report.removed_resurrected_ids.contains(&"root".to_owned()));
        assert!(report.removed_resurrected_ids.contains(&"derived".to_owned()));
        assert_eq!(
            verify_store_head(&recovered, &witness),
            Ok(())
        );
    }

    #[test]
    fn whole_store_rollback_without_a_tombstone_fails_closed() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("first", "source-a", &[]), &authority)
            .expect("first write");
        let stale_backup = store.clone();
        commit_record(&mut store, &mut witness, record("second", "source-b", &[]), &authority)
            .expect("second write");
        assert_eq!(
            reconcile_snapshot(stale_backup, &witness).map(|_| ()),
            Err(LifecycleError::RollbackOrTamperDetected)
        );
    }

    #[test]
    fn unauthorized_actor_cannot_revoke_or_erase() {
        let (mut store, mut witness, mut authority) = setup();
        let mut writer_only = full_profile();
        writer_only.may_revoke = false;
        writer_only.may_erase = false;
        authority.allow("source-a", writer_only);
        commit_record(&mut store, &mut witness, record("root", "source-a", &[]), &authority)
            .expect("root");
        assert_eq!(
            revoke_record(&mut store, &mut witness, &authority, "source-a", "root"),
            Err(LifecycleError::RevokerNotAuthorized)
        );
        assert_eq!(
            erase_record(&mut store, &mut witness, &authority, "source-a", "root"),
            Err(LifecycleError::EraserNotAuthorized)
        );
        assert!(store.contains("root"));
    }

    #[test]
    fn duplicate_or_unsorted_parent_ids_are_rejected() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("a", "source-a", &[]), &authority)
            .expect("a");
        commit_record(&mut store, &mut witness, record("b", "source-b", &[]), &authority)
            .expect("b");
        let mut malformed = record("c", "source-a", &["a", "b"]);
        malformed.parents = vec!["b".to_owned(), "a".to_owned()];
        assert_eq!(
            commit_record(&mut store, &mut witness, malformed, &authority),
            Err(LifecycleError::NonCanonicalParents)
        );
    }

    #[test]
    fn head_commitment_is_input_order_independent() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("z", "source-a", &[]), &authority)
            .expect("z");
        commit_record(&mut store, &mut witness, record("a", "source-b", &[]), &authority)
            .expect("a");
        let digest = store.head_sha256();
        assert_eq!(digest, store.clone().head_sha256());
        assert_ne!(digest, String::new());
    }

    #[test]
    fn erasure_is_idempotent_for_a_known_tombstone() {
        let (mut store, mut witness, authority) = setup();
        commit_record(&mut store, &mut witness, record("root", "source-a", &[]), &authority)
            .expect("root");
        erase_record(&mut store, &mut witness, &authority, "eraser", "root")
            .expect("first erase");
        let generation = witness.generation();
        let second = erase_record(&mut store, &mut witness, &authority, "eraser", "root")
            .expect("idempotent erase");
        assert_eq!(witness.generation(), generation);
        assert_eq!(second.invalidated_ids, vec!["root".to_owned()]);
    }
}
