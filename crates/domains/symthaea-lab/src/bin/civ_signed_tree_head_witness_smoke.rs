//! SYM-CIV-009: signed tree heads, witness lifecycle, and equivocation smoke.
//!
//! This is a protocol/state-machine smoke, not cryptography.
//!
//! FixtureAuthenticator uses a public deterministic digest to prove that the
//! exact canonical statement is being bound and checked. Anyone can forge it.
//! It MUST NOT be treated as a signature scheme, signer authentication, or
//! production verification. A real signature backend and interop/security review
//! are required before any production trust claim.
//!
//! The fixture separates:
//! signed statement authenticity at signing time
//! current key eligibility
//! current witness-quorum eligibility
//! Merkle consistency
//! witness agreement / equivocation
//!
//! Those are distinct checks and must not silently substitute for each other.

use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

type Hash = [u8; 32];

const HEAD_DOMAIN: &[u8] = b"mycelix-civ-signed-tree-head-v1\0";
const WITNESS_DOMAIN: &[u8] = b"mycelix-civ-witness-attestation-v1\0";
const HEAD_DIGEST_DOMAIN: &[u8] = b"mycelix-civ-tree-head-identity-v1\0";
const FIXTURE_AUTH_DOMAIN: &[u8] = b"TEST-ONLY-NOT-A-SIGNATURE-v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignatureAlgorithm {
    FixtureOnlyDeterministicDigest,
    MlDsa65,
}

impl SignatureAlgorithm {
    fn id(self) -> &'static [u8] {
        match self {
            Self::FixtureOnlyDeterministicDigest => b"fixture-only-deterministic-digest-v1",
            Self::MlDsa65 => b"ML-DSA-65-FIPS-204",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KeyRole {
    LogSigner,
    Witness,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KeyRecord {
    key_id: String,
    principal_id: String,
    authority_lineage: String,
    role: KeyRole,
    algorithm: SignatureAlgorithm,
    key_epoch: u64,
    valid_from_epoch: u64,
    valid_until_epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KeyLifecycleAction {
    Registered,
    Revoked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KeyLifecycleEvent {
    sequence: u64,
    key_id: String,
    action: KeyLifecycleAction,
    effective_epoch: u64,
    reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KeyRegistry {
    keys: BTreeMap<String, KeyRecord>,
    events: Vec<KeyLifecycleEvent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TreeHeadStatement {
    protocol_version: u16,
    log_id: String,
    tree_size: u64,
    root_hash: Hash,
    timestamp_epoch: u64,
    signature_algorithm: SignatureAlgorithm,
    key_id: String,
    key_epoch: u64,
    policy_version: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SignedTreeHead {
    statement: TreeHeadStatement,
    signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WitnessAttestation {
    witness_id: String,
    authority_lineage: String,
    witness_key_id: String,
    witness_key_epoch: u64,
    algorithm: SignatureAlgorithm,
    signed_at_epoch: u64,
    head: SignedTreeHead,
    signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EquivocationEvidence {
    log_id: String,
    tree_size: u64,
    first_head: SignedTreeHead,
    second_head: SignedTreeHead,
    first_observer: String,
    second_observer: String,
    detected_at_epoch: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct EquivocationEvidenceStore {
    records: Vec<EquivocationEvidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VerifyFailure {
    DuplicateKeyId,
    InvalidKeyValidity,
    NonMonotonicKeyEpoch,
    UnknownKey,
    WrongKeyRole,
    PrincipalMismatch,
    AuthorityLineageMismatch,
    AlgorithmMismatch,
    KeyEpochMismatch,
    KeyNotYetValidAtSigning,
    KeyExpiredAtSigning,
    KeyRevokedAtSigning,
    KeyRevokedForCurrentUse,
    KeyInactiveForCurrentUse,
    SignatureMismatch,
    UnsupportedProductionAlgorithm,
    UnsupportedProtocolVersion,
    WrongLog,
    FutureDatedHead,
    StaleHead,
    InvalidWitnessTime,
    HeadMismatch,
    InsufficientWitnesses,
    DuplicateWitness,
    SharedAuthorityLineage,
    NotEquivocation,
}

impl KeyRegistry {
    fn new() -> Self {
        Self {
            keys: BTreeMap::new(),
            events: Vec::new(),
        }
    }

    fn register(&mut self, key: KeyRecord) -> Result<(), VerifyFailure> {
        if self.keys.contains_key(&key.key_id) {
            return Err(VerifyFailure::DuplicateKeyId);
        }
        if key.valid_from_epoch > key.valid_until_epoch {
            return Err(VerifyFailure::InvalidKeyValidity);
        }
        if self.keys.values().any(|existing| {
            existing.role == key.role
                && existing.principal_id == key.principal_id
                && existing.key_epoch >= key.key_epoch
        }) {
            return Err(VerifyFailure::NonMonotonicKeyEpoch);
        }
        let next_seq = self.events.len() as u64 + 1;
        self.events.push(KeyLifecycleEvent {
            sequence: next_seq,
            key_id: key.key_id.clone(),
            action: KeyLifecycleAction::Registered,
            effective_epoch: key.valid_from_epoch,
            reason: "key-registration".to_owned(),
        });
        self.keys.insert(key.key_id.clone(), key);
        Ok(())
    }

    fn revoke(
        &mut self,
        key_id: &str,
        effective_epoch: u64,
        reason: &str,
    ) -> Result<(), VerifyFailure> {
        if !self.keys.contains_key(key_id) {
            return Err(VerifyFailure::UnknownKey);
        }
        if self.events.iter().any(|event| {
            event.key_id == key_id && event.action == KeyLifecycleAction::Revoked
        }) {
            return Err(VerifyFailure::KeyRevokedForCurrentUse);
        }

        let next_seq = self.events.len() as u64 + 1;
        self.events.push(KeyLifecycleEvent {
            sequence: next_seq,
            key_id: key_id.to_owned(),
            action: KeyLifecycleAction::Revoked,
            effective_epoch,
            reason: reason.to_owned(),
        });
        Ok(())
    }

    fn key(&self, key_id: &str) -> Result<&KeyRecord, VerifyFailure> {
        self.keys.get(key_id).ok_or(VerifyFailure::UnknownKey)
    }

    fn revocation_epoch(&self, key_id: &str) -> Option<u64> {
        self.events
            .iter()
            .filter(|event| {
                event.key_id == key_id && event.action == KeyLifecycleAction::Revoked
            })
            .map(|event| event.effective_epoch)
            .min()
    }

    fn verify_historical_key_time(
        &self,
        key: &KeyRecord,
        signed_at_epoch: u64,
    ) -> Result<(), VerifyFailure> {
        if signed_at_epoch < key.valid_from_epoch {
            return Err(VerifyFailure::KeyNotYetValidAtSigning);
        }
        if signed_at_epoch > key.valid_until_epoch {
            return Err(VerifyFailure::KeyExpiredAtSigning);
        }
        if self
            .revocation_epoch(&key.key_id)
            .is_some_and(|revoked_at| revoked_at <= signed_at_epoch)
        {
            return Err(VerifyFailure::KeyRevokedAtSigning);
        }
        Ok(())
    }

    fn require_current_key_eligibility(
        &self,
        key: &KeyRecord,
        now_epoch: u64,
    ) -> Result<(), VerifyFailure> {
        if self
            .revocation_epoch(&key.key_id)
            .is_some_and(|revoked_at| revoked_at <= now_epoch)
        {
            return Err(VerifyFailure::KeyRevokedForCurrentUse);
        }
        if now_epoch < key.valid_from_epoch || now_epoch > key.valid_until_epoch {
            return Err(VerifyFailure::KeyInactiveForCurrentUse);
        }
        Ok(())
    }
}

/// Protocol abstraction for a real signature verifier.
///
/// The only implementation below is deliberately forgeable and test-only. Do
/// not substitute it for ML-DSA, Ed25519, or any other actual signature scheme.
trait StatementAuthenticator {
    fn verify(
        &self,
        key_id: &str,
        algorithm: SignatureAlgorithm,
        message: &[u8],
        signature: &[u8],
    ) -> bool;
}

/// Public deterministic test double: demonstrates byte binding only; it has no
/// secret key and provides no signer authentication or unforgeability.
struct FixtureAuthenticator;

impl FixtureAuthenticator {
    fn forgeable_fixture_tag(
        key_id: &str,
        algorithm: SignatureAlgorithm,
        message: &[u8],
    ) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(FIXTURE_AUTH_DOMAIN);
        update_field(&mut hasher, key_id.as_bytes());
        update_field(&mut hasher, algorithm.id());
        update_field(&mut hasher, message);
        hasher.finalize().to_vec()
    }
}

impl StatementAuthenticator for FixtureAuthenticator {
    fn verify(
        &self,
        key_id: &str,
        algorithm: SignatureAlgorithm,
        message: &[u8],
        signature: &[u8],
    ) -> bool {
        let expected = Self::forgeable_fixture_tag(key_id, algorithm, message);
        signature == expected.as_slice()
    }
}

fn update_field(hasher: &mut Sha256, field: &[u8]) {
    hasher.update((field.len() as u64).to_be_bytes());
    hasher.update(field);
}

fn encode_field(output: &mut Vec<u8>, field: &[u8]) {
    output.extend_from_slice(&(field.len() as u64).to_be_bytes());
    output.extend_from_slice(field);
}

fn tree_head_signing_bytes(statement: &TreeHeadStatement) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(HEAD_DOMAIN);
    output.extend_from_slice(&statement.protocol_version.to_be_bytes());
    encode_field(&mut output, statement.log_id.as_bytes());
    output.extend_from_slice(&statement.tree_size.to_be_bytes());
    output.extend_from_slice(&statement.root_hash);
    output.extend_from_slice(&statement.timestamp_epoch.to_be_bytes());
    encode_field(&mut output, statement.signature_algorithm.id());
    encode_field(&mut output, statement.key_id.as_bytes());
    output.extend_from_slice(&statement.key_epoch.to_be_bytes());
    encode_field(&mut output, statement.policy_version.as_bytes());
    output
}

fn signed_tree_head_identity(head: &SignedTreeHead) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(HEAD_DIGEST_DOMAIN);
    update_field(&mut hasher, &tree_head_signing_bytes(&head.statement));
    update_field(&mut hasher, &head.signature);
    hasher.finalize().into()
}

fn witness_signing_bytes(attestation: &WitnessAttestation) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(WITNESS_DOMAIN);
    encode_field(&mut output, attestation.witness_id.as_bytes());
    encode_field(&mut output, attestation.authority_lineage.as_bytes());
    encode_field(&mut output, attestation.witness_key_id.as_bytes());
    output.extend_from_slice(&attestation.witness_key_epoch.to_be_bytes());
    encode_field(&mut output, attestation.algorithm.id());
    output.extend_from_slice(&attestation.signed_at_epoch.to_be_bytes());
    output.extend_from_slice(&signed_tree_head_identity(&attestation.head));
    output
}

fn test_sign_tree_head(statement: TreeHeadStatement) -> SignedTreeHead {
    let bytes = tree_head_signing_bytes(&statement);
    let signature = FixtureAuthenticator::forgeable_fixture_tag(
        &statement.key_id,
        statement.signature_algorithm,
        &bytes,
    );
    SignedTreeHead {
        statement,
        signature,
    }
}

fn test_sign_witness_attestation(
    witness_id: &str,
    authority_lineage: &str,
    witness_key_id: &str,
    witness_key_epoch: u64,
    algorithm: SignatureAlgorithm,
    signed_at_epoch: u64,
    head: SignedTreeHead,
) -> WitnessAttestation {
    let mut attestation = WitnessAttestation {
        witness_id: witness_id.to_owned(),
        authority_lineage: authority_lineage.to_owned(),
        witness_key_id: witness_key_id.to_owned(),
        witness_key_epoch,
        algorithm,
        signed_at_epoch,
        head,
        signature: Vec::new(),
    };
    attestation.signature = FixtureAuthenticator::forgeable_fixture_tag(
        &attestation.witness_key_id,
        attestation.algorithm,
        &witness_signing_bytes(&attestation),
    );
    attestation
}

fn verify_signed_head_at_signing_time(
    head: &SignedTreeHead,
    registry: &KeyRegistry,
    verifier: &impl StatementAuthenticator,
) -> Result<(), VerifyFailure> {
    let statement = &head.statement;
    if statement.protocol_version != 1 {
        return Err(VerifyFailure::UnsupportedProtocolVersion);
    }

    let key = registry.key(&statement.key_id)?;
    if key.role != KeyRole::LogSigner {
        return Err(VerifyFailure::WrongKeyRole);
    }
    if key.principal_id != statement.log_id {
        return Err(VerifyFailure::PrincipalMismatch);
    }
    if key.algorithm != statement.signature_algorithm {
        return Err(VerifyFailure::AlgorithmMismatch);
    }
    if key.key_epoch != statement.key_epoch {
        return Err(VerifyFailure::KeyEpochMismatch);
    }
    if statement.signature_algorithm != SignatureAlgorithm::FixtureOnlyDeterministicDigest {
        return Err(VerifyFailure::UnsupportedProductionAlgorithm);
    }

    registry.verify_historical_key_time(key, statement.timestamp_epoch)?;
    let bytes = tree_head_signing_bytes(statement);
    if !verifier.verify(
        &statement.key_id,
        statement.signature_algorithm,
        &bytes,
        &head.signature,
    ) {
        return Err(VerifyFailure::SignatureMismatch);
    }
    Ok(())
}

fn verify_signed_head_for_current_use(
    head: &SignedTreeHead,
    registry: &KeyRegistry,
    verifier: &impl StatementAuthenticator,
    now_epoch: u64,
    max_age_epochs: u64,
) -> Result<(), VerifyFailure> {
    verify_signed_head_at_signing_time(head, registry, verifier)?;

    if now_epoch < head.statement.timestamp_epoch {
        return Err(VerifyFailure::FutureDatedHead);
    }
    if now_epoch - head.statement.timestamp_epoch > max_age_epochs {
        return Err(VerifyFailure::StaleHead);
    }

    let key = registry.key(&head.statement.key_id)?;
    registry.require_current_key_eligibility(key, now_epoch)
}

fn verify_witness_at_signing_time(
    attestation: &WitnessAttestation,
    registry: &KeyRegistry,
    verifier: &impl StatementAuthenticator,
) -> Result<(), VerifyFailure> {
    let key = registry.key(&attestation.witness_key_id)?;
    if key.role != KeyRole::Witness {
        return Err(VerifyFailure::WrongKeyRole);
    }
    if key.principal_id != attestation.witness_id {
        return Err(VerifyFailure::PrincipalMismatch);
    }
    if key.authority_lineage != attestation.authority_lineage {
        return Err(VerifyFailure::AuthorityLineageMismatch);
    }
    if key.algorithm != attestation.algorithm {
        return Err(VerifyFailure::AlgorithmMismatch);
    }
    if key.key_epoch != attestation.witness_key_epoch {
        return Err(VerifyFailure::KeyEpochMismatch);
    }
    if attestation.algorithm != SignatureAlgorithm::FixtureOnlyDeterministicDigest {
        return Err(VerifyFailure::UnsupportedProductionAlgorithm);
    }
    if attestation.signed_at_epoch < attestation.head.statement.timestamp_epoch {
        return Err(VerifyFailure::InvalidWitnessTime);
    }

    registry.verify_historical_key_time(key, attestation.signed_at_epoch)?;
    let bytes = witness_signing_bytes(attestation);
    if !verifier.verify(
        &attestation.witness_key_id,
        attestation.algorithm,
        &bytes,
        &attestation.signature,
    ) {
        return Err(VerifyFailure::SignatureMismatch);
    }
    Ok(())
}

fn verify_witness_for_current_quorum(
    attestation: &WitnessAttestation,
    registry: &KeyRegistry,
    verifier: &impl StatementAuthenticator,
    now_epoch: u64,
    max_head_age_epochs: u64,
) -> Result<(), VerifyFailure> {
    verify_witness_at_signing_time(attestation, registry, verifier)?;
    verify_signed_head_for_current_use(
        &attestation.head,
        registry,
        verifier,
        now_epoch,
        max_head_age_epochs,
    )?;

    if now_epoch < attestation.signed_at_epoch {
        return Err(VerifyFailure::FutureDatedHead);
    }
    let witness_key = registry.key(&attestation.witness_key_id)?;
    registry.require_current_key_eligibility(witness_key, now_epoch)
}

fn verify_witness_quorum(
    attestations: &[WitnessAttestation],
    expected_head: &SignedTreeHead,
    minimum_witnesses: usize,
    registry: &KeyRegistry,
    verifier: &impl StatementAuthenticator,
    now_epoch: u64,
    max_head_age_epochs: u64,
) -> Result<(), VerifyFailure> {
    if attestations.len() < minimum_witnesses {
        return Err(VerifyFailure::InsufficientWitnesses);
    }

    let expected_head_id = signed_tree_head_identity(expected_head);
    let mut witness_ids = BTreeSet::new();
    let mut authority_lineages = BTreeSet::new();

    for attestation in attestations {
        if signed_tree_head_identity(&attestation.head) != expected_head_id {
            return Err(VerifyFailure::HeadMismatch);
        }
        verify_witness_for_current_quorum(
            attestation,
            registry,
            verifier,
            now_epoch,
            max_head_age_epochs,
        )?;
        if !witness_ids.insert(attestation.witness_id.as_str()) {
            return Err(VerifyFailure::DuplicateWitness);
        }
        if !authority_lineages.insert(attestation.authority_lineage.as_str()) {
            return Err(VerifyFailure::SharedAuthorityLineage);
        }
    }

    Ok(())
}

fn detect_equivocation(
    first: &SignedTreeHead,
    second: &SignedTreeHead,
    first_observer: &str,
    second_observer: &str,
    detected_at_epoch: u64,
    registry: &KeyRegistry,
    verifier: &impl StatementAuthenticator,
) -> Result<EquivocationEvidence, VerifyFailure> {
    verify_signed_head_at_signing_time(first, registry, verifier)?;
    verify_signed_head_at_signing_time(second, registry, verifier)?;

    if first.statement.log_id != second.statement.log_id {
        return Err(VerifyFailure::WrongLog);
    }
    if first.statement.tree_size != second.statement.tree_size
        || first.statement.root_hash == second.statement.root_hash
    {
        return Err(VerifyFailure::NotEquivocation);
    }

    Ok(EquivocationEvidence {
        log_id: first.statement.log_id.clone(),
        tree_size: first.statement.tree_size,
        first_head: first.clone(),
        second_head: second.clone(),
        first_observer: first_observer.to_owned(),
        second_observer: second_observer.to_owned(),
        detected_at_epoch,
    })
}

fn sample_key(
    key_id: &str,
    principal_id: &str,
    lineage: &str,
    role: KeyRole,
    key_epoch: u64,
    valid_from_epoch: u64,
    valid_until_epoch: u64,
) -> KeyRecord {
    KeyRecord {
        key_id: key_id.to_owned(),
        principal_id: principal_id.to_owned(),
        authority_lineage: lineage.to_owned(),
        role,
        algorithm: SignatureAlgorithm::FixtureOnlyDeterministicDigest,
        key_epoch,
        valid_from_epoch,
        valid_until_epoch,
    }
}

fn sample_head(
    root_hash: Hash,
    timestamp_epoch: u64,
    key_id: &str,
    key_epoch: u64,
) -> SignedTreeHead {
    test_sign_tree_head(TreeHeadStatement {
        protocol_version: 1,
        log_id: "civ-log-v1".to_owned(),
        tree_size: 12,
        root_hash,
        timestamp_epoch,
        signature_algorithm: SignatureAlgorithm::FixtureOnlyDeterministicDigest,
        key_id: key_id.to_owned(),
        key_epoch,
        policy_version: "witness-policy-v1".to_owned(),
    })
}

fn main() {
    let verifier = FixtureAuthenticator;
    let mut registry = KeyRegistry::new();

    registry
        .register(sample_key(
            "log-key-v1",
            "civ-log-v1",
            "lineage-log-operator",
            KeyRole::LogSigner,
            1,
            90,
            200,
        ))
        .expect("register log key");

    for (key_id, witness_id, lineage) in [
        ("witness-a-key-v1", "witness-a", "authority-lineage-a"),
        ("witness-b-key-v1", "witness-b", "authority-lineage-b"),
        ("witness-c-key-v1", "witness-c", "authority-lineage-c"),
    ] {
        registry
            .register(sample_key(
                key_id,
                witness_id,
                lineage,
                KeyRole::Witness,
                1,
                90,
                200,
            ))
            .expect("register witness key");
    }

    let root_a: Hash = Sha256::digest(b"merkle-root-a").into();
    let head_a = sample_head(root_a, 100, "log-key-v1", 1);
    assert_eq!(
        verify_signed_head_at_signing_time(&head_a, &registry, &verifier),
        Ok(())
    );
    assert_eq!(
        verify_signed_head_for_current_use(&head_a, &registry, &verifier, 110, 20),
        Ok(())
    );

    // The fixture tag is bound to the canonical statement, including log ID,
    // tree size, root, timestamp, algorithm, key ID, key epoch and policy.
    let mut modified = head_a.clone();
    modified.statement.root_hash[0] ^= 0x01;
    assert_eq!(
        verify_signed_head_at_signing_time(&modified, &registry, &verifier),
        Err(VerifyFailure::SignatureMismatch)
    );
    let mut wrong_log = head_a.clone();
    wrong_log.statement.log_id = "civ-log-other".to_owned();
    assert_eq!(
        verify_signed_head_at_signing_time(&wrong_log, &registry, &verifier),
        Err(VerifyFailure::PrincipalMismatch)
    );
    let mut wrong_algorithm = head_a.clone();
    wrong_algorithm.statement.signature_algorithm = SignatureAlgorithm::MlDsa65;
    assert_eq!(
        verify_signed_head_at_signing_time(&wrong_algorithm, &registry, &verifier),
        Err(VerifyFailure::AlgorithmMismatch)
    );
    let mut wrong_epoch = head_a.clone();
    wrong_epoch.statement.key_epoch = 2;
    assert_eq!(
        verify_signed_head_at_signing_time(&wrong_epoch, &registry, &verifier),
        Err(VerifyFailure::KeyEpochMismatch)
    );

    assert_eq!(
        verify_signed_head_for_current_use(&head_a, &registry, &verifier, 99, 20),
        Err(VerifyFailure::FutureDatedHead)
    );
    assert_eq!(
        verify_signed_head_for_current_use(&head_a, &registry, &verifier, 125, 30),
        Ok(())
    );
    assert_eq!(
        verify_signed_head_for_current_use(&head_a, &registry, &verifier, 121, 10),
        Err(VerifyFailure::StaleHead)
    );

    // A later key revocation preserves historical authenticity while removing
    // the key from current-use eligibility.
    registry
        .revoke("log-key-v1", 120, "operator-key-compromise")
        .expect("append log key revocation");
    assert_eq!(
        verify_signed_head_at_signing_time(&head_a, &registry, &verifier),
        Ok(())
    );
    assert_eq!(
        verify_signed_head_for_current_use(&head_a, &registry, &verifier, 121, 30),
        Err(VerifyFailure::KeyRevokedForCurrentUse)
    );
    assert_eq!(
        registry.revoke("log-key-v1", 125, "duplicate-revocation"),
        Err(VerifyFailure::KeyRevokedForCurrentUse)
    );

    // A statement signed at or after revocation is not historically authentic.
    let post_revoke = sample_head(
        Sha256::digest(b"post-revoke-root").into(),
        121,
        "log-key-v1",
        1,
    );
    assert_eq!(
        verify_signed_head_at_signing_time(&post_revoke, &registry, &verifier),
        Err(VerifyFailure::KeyRevokedAtSigning)
    );

    // Key IDs are never reused; rotation requires a fresh ID and epoch.
    assert_eq!(
        registry.register(sample_key(
            "log-key-invalid-window",
            "civ-log-invalid",
            "lineage-invalid",
            KeyRole::LogSigner,
            1,
            200,
            100,
        )),
        Err(VerifyFailure::InvalidKeyValidity)
    );
    assert_eq!(
        registry.register(sample_key(
            "log-key-epoch-replay",
            "civ-log-v1",
            "lineage-log-operator",
            KeyRole::LogSigner,
            1,
            121,
            300,
        )),
        Err(VerifyFailure::NonMonotonicKeyEpoch)
    );

    registry
        .register(sample_key(
            "log-key-v2",
            "civ-log-v1",
            "lineage-log-operator",
            KeyRole::LogSigner,
            2,
            121,
            300,
        ))
        .expect("register rotated log key");
    assert_eq!(
        registry.register(sample_key(
            "log-key-v1",
            "civ-log-v1",
            "lineage-log-operator",
            KeyRole::LogSigner,
            3,
            121,
            300,
        )),
        Err(VerifyFailure::DuplicateKeyId)
    );
    let head_v2 = sample_head(
        Sha256::digest(b"rotated-root").into(),
        130,
        "log-key-v2",
        2,
    );
    assert_eq!(
        verify_signed_head_for_current_use(&head_v2, &registry, &verifier, 135, 20),
        Ok(())
    );

    // Witnesses attest to the exact serialized signed tree head.
    let quorum_head = head_v2.clone();
    let witnesses = vec![
        test_sign_witness_attestation(
            "witness-a",
            "authority-lineage-a",
            "witness-a-key-v1",
            1,
            SignatureAlgorithm::FixtureOnlyDeterministicDigest,
            132,
            quorum_head.clone(),
        ),
        test_sign_witness_attestation(
            "witness-b",
            "authority-lineage-b",
            "witness-b-key-v1",
            1,
            SignatureAlgorithm::FixtureOnlyDeterministicDigest,
            132,
            quorum_head.clone(),
        ),
        test_sign_witness_attestation(
            "witness-c",
            "authority-lineage-c",
            "witness-c-key-v1",
            1,
            SignatureAlgorithm::FixtureOnlyDeterministicDigest,
            132,
            quorum_head.clone(),
        ),
    ];

    assert_eq!(
        verify_witness_quorum(&witnesses, &quorum_head, 3, &registry, &verifier, 135, 20),
        Ok(())
    );

    let duplicate_witness = vec![
        witnesses[0].clone(),
        witnesses[0].clone(),
        witnesses[2].clone(),
    ];
    assert_eq!(
        verify_witness_quorum(
            &duplicate_witness,
            &quorum_head,
            3,
            &registry,
            &verifier,
            135,
            20
        ),
        Err(VerifyFailure::DuplicateWitness)
    );

    // Two IDs cannot conceal a single underlying authority lineage. Real
    // registered keys must agree with the claimed lineage before quorum credit.
    let witness_b_same_lineage_key = sample_key(
        "witness-b-captured-key-v1",
        "witness-b-captured",
        "authority-lineage-a",
        KeyRole::Witness,
        1,
        90,
        200,
    );
    registry
        .register(witness_b_same_lineage_key)
        .expect("register separate identity with shared lineage");
    let captured = test_sign_witness_attestation(
        "witness-b-captured",
        "authority-lineage-a",
        "witness-b-captured-key-v1",
        1,
        SignatureAlgorithm::FixtureOnlyDeterministicDigest,
        132,
        quorum_head.clone(),
    );
    assert_eq!(
        verify_witness_quorum(
            &[witnesses[0].clone(), captured, witnesses[2].clone()],
            &quorum_head,
            3,
            &registry,
            &verifier,
            135,
            20
        ),
        Err(VerifyFailure::SharedAuthorityLineage)
    );

    // Witness key revocation has the same historical/current distinction.
    registry
        .revoke("witness-a-key-v1", 134, "witness-identity-compromise")
        .expect("append witness-key revocation");
    assert_eq!(
        verify_witness_at_signing_time(&witnesses[0], &registry, &verifier),
        Ok(())
    );
    assert_eq!(
        verify_witness_quorum(&witnesses, &quorum_head, 3, &registry, &verifier, 135, 20),
        Err(VerifyFailure::KeyRevokedForCurrentUse)
    );

    // Two individually fixture-authentic tree heads with the same log and size
    // but different roots produce retained equivocation evidence. No vote,
    // timestamp, or quorum automatically chooses a canonical root.
    let conflicting_head = sample_head(
        Sha256::digest(b"conflicting-root").into(),
        130,
        "log-key-v2",
        2,
    );
    let evidence = detect_equivocation(
        &quorum_head,
        &conflicting_head,
        "client-east",
        "client-west",
        140,
        &registry,
        &verifier,
    )
    .expect("same-size conflicting authenticated heads should form evidence");
    assert_eq!(evidence.log_id, "civ-log-v1");
    assert_eq!(evidence.tree_size, quorum_head.statement.tree_size);

    let mut evidence_store = EquivocationEvidenceStore::default();
    evidence_store.records.push(evidence.clone());
    assert_eq!(evidence_store.records.len(), 1);
    assert_eq!(evidence_store.records[0].first_head, quorum_head);
    assert_eq!(evidence_store.records[0].second_head, conflicting_head);

    // A good statement-authentication result does not establish Merkle
    // consistency; no consistency proof is accepted by this module.
    assert_eq!(
        verify_signed_head_at_signing_time(&head_v2, &registry, &verifier),
        Ok(())
    );

    println!("SYM-CIV-009 PASS: signed-head binding, key lifecycle, quorum and equivocation state-machine controls hold.");
    println!("Claim ceiling: forgeable test authenticator only; no cryptographic signature or production gossip claim.");
}
