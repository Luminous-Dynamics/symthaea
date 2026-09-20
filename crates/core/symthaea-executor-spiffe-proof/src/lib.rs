// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical, non-verifying SPIFFE X.509-SVID workload proof evidence.
//!
//! This crate freezes an ordinary evidence envelope and the exact bytes a
//! challenged workload signs. It deliberately does not parse X.509, validate a
//! SPIFFE ID, trust a certificate chain, evaluate revocation/currentness, or
//! verify the proof-of-possession signature.
//!
//! ```text
//! SpiffeWorkloadProofV1
//!     != valid X.509-SVID
//!     != trusted SPIFFE trust domain
//!     != current bundle / CRL state
//!     != verified proof of possession
//!     != verified Workload
//!     != same-subject relation
//!     != VerifiedExecutorBinding
//!     != authority
//! ```
//!
//! The proof carries neither private-key material nor verifier trust state. A
//! later concrete verifier must derive the supported signature algorithm from
//! the exact leaf certificate/public key and validate the chain against its own
//! independently configured/current SPIFFE trust state.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;
use symthaea_executor_identity::ExecutorIdentityChallenge;
use symthaea_interaction_core::Digest32;

pub const SPIFFE_WORKLOAD_PROOF_SCHEMA_VERSION: u16 = 1;

/// Symthaea envelope bounds, not SPIFFE specification limits.
pub const MAX_SPIFFE_ID_BYTES: usize = 2_048;
pub const MAX_X509_SVID_CHAIN_ENTRIES: usize = 8;
pub const MAX_X509_CERT_DER_BYTES: usize = 64 * 1_024;
pub const MAX_X509_CHAIN_DER_BYTES: usize = 256 * 1_024;
pub const MAX_PROOF_SIGNATURE_BYTES: usize = 16 * 1_024;

const CLAIM_DOMAIN: &[u8] = b"symthaea.executor.spiffe.workload-proof.claim.v1\0";
const CHAIN_DOMAIN: &[u8] = b"symthaea.executor.spiffe.workload-proof.chain.v1\0";
const ENVELOPE_DOMAIN: &[u8] = b"symthaea.executor.spiffe.workload-proof.envelope.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpiffeWorkloadProofProfileV1 {
    /// X.509-SVID leaf-key proof of possession over the exact canonical claim.
    X509SvidChallengePop,
}

impl SpiffeWorkloadProofProfileV1 {
    const fn code(self) -> u16 {
        match self {
            Self::X509SvidChallengePop => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpiffeWorkloadProofError {
    EmptySpiffeId,
    SpiffeIdTooLarge,
    SpiffeIdContainsControl,
    EmptyCertificateChain,
    TooManyCertificateChainEntries,
    EmptyCertificate { index: usize },
    CertificateTooLarge { index: usize },
    CertificateChainTooLarge,
    EmptyProofSignature,
    ProofSignatureTooLarge,
}

impl fmt::Display for SpiffeWorkloadProofError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySpiffeId => write!(formatter, "SPIFFE ID evidence is empty"),
            Self::SpiffeIdTooLarge => write!(formatter, "SPIFFE ID evidence exceeds the Symthaea profile bound"),
            Self::SpiffeIdContainsControl => write!(formatter, "SPIFFE ID evidence contains an ASCII control character"),
            Self::EmptyCertificateChain => write!(formatter, "X.509-SVID chain evidence is empty"),
            Self::TooManyCertificateChainEntries => write!(formatter, "X.509-SVID chain evidence exceeds the entry bound"),
            Self::EmptyCertificate { index } => write!(formatter, "X.509 certificate evidence at index {index} is empty"),
            Self::CertificateTooLarge { index } => write!(formatter, "X.509 certificate evidence at index {index} exceeds the per-certificate bound"),
            Self::CertificateChainTooLarge => write!(formatter, "X.509-SVID chain evidence exceeds the total DER bound"),
            Self::EmptyProofSignature => write!(formatter, "workload proof-of-possession signature is empty"),
            Self::ProofSignatureTooLarge => write!(formatter, "workload proof-of-possession signature exceeds the profile bound"),
        }
    }
}

impl Error for SpiffeWorkloadProofError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpiffeWorkloadClaimIdV1(Digest32);

impl SpiffeWorkloadClaimIdV1 {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpiffeX509ChainIdV1(Digest32);

impl SpiffeX509ChainIdV1 {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpiffeWorkloadProofIdV1(Digest32);

impl SpiffeWorkloadProofIdV1 {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

/// Ordinary evidence emitted by a challenged workload.
///
/// The first DER chain entry is positionally designated as the asserted leaf
/// because the SPIFFE Workload API represents X.509-SVID chains leaf-first. This
/// crate only commits that position; it does not parse or validate the entry as
/// a certificate. A later concrete verifier must establish all X.509/SPIFFE
/// semantics independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpiffeWorkloadProofV1 {
    profile: SpiffeWorkloadProofProfileV1,
    challenge: Digest32,
    spiffe_id: String,
    x509_svid_chain_der: Vec<Vec<u8>>,
    leaf_certificate_sha256: Digest32,
    proof_signature: Vec<u8>,
    claim_id: SpiffeWorkloadClaimIdV1,
    chain_id: SpiffeX509ChainIdV1,
    id: SpiffeWorkloadProofIdV1,
}

impl SpiffeWorkloadProofV1 {
    /// Construct ordinary proof evidence for the exact EXEC-ID challenge.
    ///
    /// Construction performs only deterministic envelope validation and
    /// canonicalization. In particular, it does not establish that `spiffe_id`
    /// is syntactically/canonically valid, that the DER bytes are certificates,
    /// or that `proof_signature` is cryptographically valid.
    pub fn new(
        challenge: &ExecutorIdentityChallenge,
        spiffe_id: String,
        x509_svid_chain_der: Vec<Vec<u8>>,
        proof_signature: Vec<u8>,
    ) -> Result<Self, SpiffeWorkloadProofError> {
        validate_spiffe_id_evidence(&spiffe_id)?;
        validate_chain_evidence(&x509_svid_chain_der)?;
        validate_signature_evidence(&proof_signature)?;

        let profile = SpiffeWorkloadProofProfileV1::X509SvidChallengePop;
        let challenge_digest = challenge.digest();
        let leaf_certificate_sha256 = sha256(&x509_svid_chain_der[0]);
        let claim_statement = build_claim_statement(
            profile,
            challenge_digest,
            &spiffe_id,
            leaf_certificate_sha256,
        );
        let claim_id = SpiffeWorkloadClaimIdV1(sha256(&claim_statement));
        let chain_id = SpiffeX509ChainIdV1(chain_digest(&x509_svid_chain_der));
        let id = SpiffeWorkloadProofIdV1(envelope_digest(
            claim_id,
            chain_id,
            &proof_signature,
        ));

        Ok(Self {
            profile,
            challenge: challenge_digest,
            spiffe_id,
            x509_svid_chain_der,
            leaf_certificate_sha256,
            proof_signature,
            claim_id,
            chain_id,
            id,
        })
    }

    pub const fn profile(&self) -> SpiffeWorkloadProofProfileV1 {
        self.profile
    }

    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge
    }

    /// Exact unverified SPIFFE-ID evidence text.
    ///
    /// Positive verification must parse this as a SPIFFE ID and require its
    /// canonical serialization to equal these exact bytes.
    pub fn spiffe_id_evidence(&self) -> &str {
        &self.spiffe_id
    }

    /// Exact ordered unverified DER chain evidence, asserted leaf first.
    pub fn x509_svid_chain_der(&self) -> &[Vec<u8>] {
        &self.x509_svid_chain_der
    }

    /// SHA-256 of `x509_svid_chain_der()[0]`, computed by this constructor.
    pub const fn leaf_certificate_sha256(&self) -> Digest32 {
        self.leaf_certificate_sha256
    }

    /// Unverified proof-of-possession signature bytes.
    ///
    /// The later concrete verifier derives the supported algorithm from the
    /// validated leaf public key; no caller-selected algorithm label exists.
    pub fn proof_signature(&self) -> &[u8] {
        &self.proof_signature
    }

    /// Exact bytes that the challenged workload must sign.
    pub fn signed_statement(&self) -> Vec<u8> {
        build_claim_statement(
            self.profile,
            self.challenge,
            &self.spiffe_id,
            self.leaf_certificate_sha256,
        )
    }

    pub const fn claim_id(&self) -> SpiffeWorkloadClaimIdV1 {
        self.claim_id
    }

    pub const fn chain_id(&self) -> SpiffeX509ChainIdV1 {
        self.chain_id
    }

    /// Audit identity of the whole ordinary evidence envelope.
    ///
    /// This digest is not a verification result and is never sufficient to
    /// construct a live Workload identity node.
    pub const fn id(&self) -> SpiffeWorkloadProofIdV1 {
        self.id
    }
}

fn validate_spiffe_id_evidence(spiffe_id: &str) -> Result<(), SpiffeWorkloadProofError> {
    if spiffe_id.is_empty() {
        return Err(SpiffeWorkloadProofError::EmptySpiffeId);
    }
    if spiffe_id.len() > MAX_SPIFFE_ID_BYTES {
        return Err(SpiffeWorkloadProofError::SpiffeIdTooLarge);
    }
    if spiffe_id.bytes().any(|byte| byte.is_ascii_control()) {
        return Err(SpiffeWorkloadProofError::SpiffeIdContainsControl);
    }
    Ok(())
}

fn validate_chain_evidence(chain: &[Vec<u8>]) -> Result<(), SpiffeWorkloadProofError> {
    if chain.is_empty() {
        return Err(SpiffeWorkloadProofError::EmptyCertificateChain);
    }
    if chain.len() > MAX_X509_SVID_CHAIN_ENTRIES {
        return Err(SpiffeWorkloadProofError::TooManyCertificateChainEntries);
    }

    let mut total = 0_usize;
    for (index, certificate) in chain.iter().enumerate() {
        if certificate.is_empty() {
            return Err(SpiffeWorkloadProofError::EmptyCertificate { index });
        }
        if certificate.len() > MAX_X509_CERT_DER_BYTES {
            return Err(SpiffeWorkloadProofError::CertificateTooLarge { index });
        }
        total = total.saturating_add(certificate.len());
        if total > MAX_X509_CHAIN_DER_BYTES {
            return Err(SpiffeWorkloadProofError::CertificateChainTooLarge);
        }
    }
    Ok(())
}

fn validate_signature_evidence(signature: &[u8]) -> Result<(), SpiffeWorkloadProofError> {
    if signature.is_empty() {
        return Err(SpiffeWorkloadProofError::EmptyProofSignature);
    }
    if signature.len() > MAX_PROOF_SIGNATURE_BYTES {
        return Err(SpiffeWorkloadProofError::ProofSignatureTooLarge);
    }
    Ok(())
}

fn build_claim_statement(
    profile: SpiffeWorkloadProofProfileV1,
    challenge: Digest32,
    spiffe_id: &str,
    leaf_certificate_sha256: Digest32,
) -> Vec<u8> {
    let mut statement = Vec::with_capacity(
        CLAIM_DOMAIN.len() + 2 + 2 + 32 + 4 + spiffe_id.len() + 32,
    );
    statement.extend_from_slice(CLAIM_DOMAIN);
    push_u16(&mut statement, SPIFFE_WORKLOAD_PROOF_SCHEMA_VERSION);
    push_u16(&mut statement, profile.code());
    statement.extend_from_slice(challenge.as_bytes());
    push_bytes_u32(&mut statement, spiffe_id.as_bytes());
    statement.extend_from_slice(leaf_certificate_sha256.as_bytes());
    statement
}

fn chain_digest(chain: &[Vec<u8>]) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(CHAIN_DOMAIN);
    hasher.update(SPIFFE_WORKLOAD_PROOF_SCHEMA_VERSION.to_be_bytes());
    hasher.update((chain.len() as u32).to_be_bytes());
    for certificate in chain {
        hasher.update((certificate.len() as u32).to_be_bytes());
        hasher.update(certificate);
    }
    finish_sha256(hasher)
}

fn envelope_digest(
    claim_id: SpiffeWorkloadClaimIdV1,
    chain_id: SpiffeX509ChainIdV1,
    signature: &[u8],
) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(ENVELOPE_DOMAIN);
    hasher.update(SPIFFE_WORKLOAD_PROOF_SCHEMA_VERSION.to_be_bytes());
    hasher.update(claim_id.digest().as_bytes());
    hasher.update(chain_id.digest().as_bytes());
    hasher.update((signature.len() as u32).to_be_bytes());
    hasher.update(signature);
    finish_sha256(hasher)
}

fn sha256(bytes: &[u8]) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    finish_sha256(hasher)
}

fn finish_sha256(hasher: Sha256) -> Digest32 {
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&digest);
    Digest32::new(bytes)
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_be_bytes());
}

fn push_bytes_u32(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u32).to_be_bytes());
    output.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_executor_identity::{
        ExecutorIdentityDimension, ExecutorIdentityDimensionSet, ExecutorIdentityProfile,
        ExecutorIdentityRequirement, ExecutorProfileId, ExecutorRuntimeIncarnationId,
    };
    use symthaea_interaction_core::{
        IdentityComponent, IdentityOrdering, NamespaceId, PrincipalRef,
    };

    const EXPECTED_CHALLENGE: &str =
        "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41";
    const EXPECTED_LEAF: &str =
        "3f6e6cc25161138d9067ffad9e9eaaf57bbb82fba3cf83c4ec37efa96cf318ba";
    const EXPECTED_STATEMENT: &str = concat!(
        "73796d74686165612e6578656375746f722e7370696666652e776f726b6c6f61642d70726f6f662e636c61696d2e763100",
        "00010000",
        "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41",
        "00000021",
        "7370696666653a2f2f6578616d706c652e6f72672f776f726b6c6f61642f617069",
        "3f6e6cc25161138d9067ffad9e9eaaf57bbb82fba3cf83c4ec37efa96cf318ba",
    );
    const EXPECTED_CLAIM: &str =
        "feb5411e3bba766511fde2c16307faa55093e7301019ed881e41e843ada55fd2";
    const EXPECTED_CHAIN: &str =
        "0d8997a95e42b40c07e938ae2bb648c6b8df1ca4b9beb4368c1f7c8ef00471b9";
    const EXPECTED_PROOF: &str =
        "309391c2040694ddb59bf8a85573f489fe1396b134612791a59eb7ffe6c24163";

    fn digest(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    fn principal() -> PrincipalRef {
        PrincipalRef::new(
            NamespaceId::new("intx/workload").expect("namespace"),
            "policy-pdp",
            IdentityOrdering::NamedSet,
            vec![IdentityComponent::new("name", "pdp-1").expect("component")],
        )
        .expect("principal")
    }

    fn requirement() -> ExecutorIdentityRequirement {
        ExecutorIdentityRequirement::new(
            ExecutorIdentityProfile::ConsequentialDigital,
            ExecutorIdentityDimensionSet::new(&[
                ExecutorIdentityDimension::SessionPeer,
                ExecutorIdentityDimension::Workload,
                ExecutorIdentityDimension::Software,
                ExecutorIdentityDimension::ExecutorProfile,
            ]),
        )
        .expect("requirement")
    }

    fn challenge(runtime: u8) -> ExecutorIdentityChallenge {
        ExecutorIdentityChallenge::new(
            [0xA3; 32],
            &principal(),
            ExecutorProfileId::new(digest(0xA1)).expect("profile"),
            ExecutorRuntimeIncarnationId::new(digest(runtime)).expect("runtime"),
            &requirement(),
        )
        .expect("challenge")
    }

    fn fixture() -> SpiffeWorkloadProofV1 {
        SpiffeWorkloadProofV1::new(
            &challenge(0xA2),
            "spiffe://example.org/workload/api".to_string(),
            vec![vec![0x30, 0x03, 0x01, 0x02, 0x03], vec![0x30, 0x02, 0x04, 0x05]],
            vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF],
        )
        .expect("proof evidence")
    }

    fn hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }

    #[test]
    fn frozen_canonical_vectors() {
        let challenge = challenge(0xA2);
        assert_eq!(challenge.digest().to_hex(), EXPECTED_CHALLENGE);
        let proof = fixture();
        assert_eq!(proof.leaf_certificate_sha256().to_hex(), EXPECTED_LEAF);
        assert_eq!(hex(&proof.signed_statement()), EXPECTED_STATEMENT);
        assert_eq!(proof.claim_id().digest().to_hex(), EXPECTED_CLAIM);
        assert_eq!(proof.chain_id().digest().to_hex(), EXPECTED_CHAIN);
        assert_eq!(proof.id().digest().to_hex(), EXPECTED_PROOF);
    }

    #[test]
    fn challenge_substitution_changes_signed_claim() {
        let baseline = fixture();
        let changed = SpiffeWorkloadProofV1::new(
            &challenge(0xA7),
            baseline.spiffe_id_evidence().to_string(),
            baseline.x509_svid_chain_der().to_vec(),
            baseline.proof_signature().to_vec(),
        )
        .unwrap();
        assert_ne!(baseline.challenge_digest(), changed.challenge_digest());
        assert_ne!(baseline.claim_id(), changed.claim_id());
        assert_ne!(baseline.id(), changed.id());
    }

    #[test]
    fn spiffe_id_substitution_changes_signed_claim() {
        let baseline = fixture();
        let changed = SpiffeWorkloadProofV1::new(
            &challenge(0xA2),
            "spiffe://example.org/workload/other".to_string(),
            baseline.x509_svid_chain_der().to_vec(),
            baseline.proof_signature().to_vec(),
        )
        .unwrap();
        assert_ne!(baseline.claim_id(), changed.claim_id());
    }

    #[test]
    fn leaf_substitution_changes_signed_claim() {
        let baseline = fixture();
        let mut chain = baseline.x509_svid_chain_der().to_vec();
        chain[0].push(0x99);
        let changed = SpiffeWorkloadProofV1::new(
            &challenge(0xA2),
            baseline.spiffe_id_evidence().to_string(),
            chain,
            baseline.proof_signature().to_vec(),
        )
        .unwrap();
        assert_ne!(baseline.leaf_certificate_sha256(), changed.leaf_certificate_sha256());
        assert_ne!(baseline.claim_id(), changed.claim_id());
    }

    #[test]
    fn intermediate_path_is_envelope_evidence_not_signed_claim() {
        let baseline = fixture();
        let mut chain = baseline.x509_svid_chain_der().to_vec();
        chain[1].push(0x88);
        let changed = SpiffeWorkloadProofV1::new(
            &challenge(0xA2),
            baseline.spiffe_id_evidence().to_string(),
            chain,
            baseline.proof_signature().to_vec(),
        )
        .unwrap();
        assert_eq!(baseline.claim_id(), changed.claim_id());
        assert_ne!(baseline.chain_id(), changed.chain_id());
        assert_ne!(baseline.id(), changed.id());
    }

    #[test]
    fn signature_substitution_changes_only_envelope_identity() {
        let baseline = fixture();
        let mut signature = baseline.proof_signature().to_vec();
        signature[0] ^= 0x01;
        let changed = SpiffeWorkloadProofV1::new(
            &challenge(0xA2),
            baseline.spiffe_id_evidence().to_string(),
            baseline.x509_svid_chain_der().to_vec(),
            signature,
        )
        .unwrap();
        assert_eq!(baseline.claim_id(), changed.claim_id());
        assert_eq!(baseline.chain_id(), changed.chain_id());
        assert_ne!(baseline.id(), changed.id());
    }

    #[test]
    fn evidence_bounds_fail_closed() {
        assert_eq!(
            SpiffeWorkloadProofV1::new(
                &challenge(0xA2),
                String::new(),
                vec![vec![1]],
                vec![1],
            ),
            Err(SpiffeWorkloadProofError::EmptySpiffeId)
        );
        assert_eq!(
            SpiffeWorkloadProofV1::new(
                &challenge(0xA2),
                "spiffe://example.org/a\n".to_string(),
                vec![vec![1]],
                vec![1],
            ),
            Err(SpiffeWorkloadProofError::SpiffeIdContainsControl)
        );
        assert_eq!(
            SpiffeWorkloadProofV1::new(
                &challenge(0xA2),
                "spiffe://example.org/a".to_string(),
                Vec::new(),
                vec![1],
            ),
            Err(SpiffeWorkloadProofError::EmptyCertificateChain)
        );
        assert_eq!(
            SpiffeWorkloadProofV1::new(
                &challenge(0xA2),
                "spiffe://example.org/a".to_string(),
                vec![Vec::new()],
                vec![1],
            ),
            Err(SpiffeWorkloadProofError::EmptyCertificate { index: 0 })
        );
        assert_eq!(
            SpiffeWorkloadProofV1::new(
                &challenge(0xA2),
                "spiffe://example.org/a".to_string(),
                vec![vec![1]],
                Vec::new(),
            ),
            Err(SpiffeWorkloadProofError::EmptyProofSignature)
        );
    }

    #[test]
    fn constructor_computes_leaf_commitment_instead_of_accepting_one() {
        let proof = fixture();
        assert_eq!(proof.leaf_certificate_sha256(), sha256(&proof.x509_svid_chain_der()[0]));
    }
}
