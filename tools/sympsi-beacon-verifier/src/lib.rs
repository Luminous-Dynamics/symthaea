#![forbid(unsafe_code)]

//! Offline verifier for the pinned drand Quicknet BLS-unchained profile.
//!
//! This proves only that a supplied signature is valid for the selected
//! Quicknet trust root and requested round. It does not establish freshness,
//! beacon publication time, response-before-target chronology, or the
//! independence of any timestamp witness.

use drandite_core::Timeline;
use drandite_core::any::{AnyVerifier, SchemeId};
use drandite_core::types::{ChainHash, GenesisSeed, Period, Round, UnixTimestamp};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const PROFILE_ID: &str = "drand-quicknet-bls-unchained-g1-rfc9380-v1";
pub const SCHEME_ID: &str = "bls-unchained-g1-rfc9380";
pub const BEACON_ID: &str = "quicknet";
pub const TRUST_ROOT_MANIFEST_SHA256: &str =
    "8a74f38b67081c9483f2a640fc4ac8d0275318d758a7c6f9cb7b1ffc374485c4";

const TRUST_ROOT_BYTES: &[u8] = include_bytes!("../fixtures/quicknet-trust-root-v1.json");

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustRootManifest {
    schema_version: String,
    profile_id: String,
    beacon_id: String,
    chain_hash: String,
    scheme_id: String,
    public_key: String,
    genesis_seed: String,
    genesis_unix_seconds: u64,
    period_seconds: u64,
    source_references: Vec<String>,
    source_retrieved_on: String,
}

fn load_trust_root() -> Result<(TrustRootManifest, [u8; 32], String), VerificationError> {
    let digest: [u8; 32] = Sha256::digest(TRUST_ROOT_BYTES).into();
    let digest_hex = encode_hex(&digest);
    if digest_hex != TRUST_ROOT_MANIFEST_SHA256 {
        return Err(VerificationError::TrustRoot(
            "embedded trust-root manifest digest differs from the pinned digest".into(),
        ));
    }
    let root: TrustRootManifest = serde_json::from_slice(TRUST_ROOT_BYTES).map_err(|error| {
        VerificationError::TrustRoot(format!("trust-root JSON failed: {error}"))
    })?;
    if root.schema_version != "sympsi.drand-trust-root.v0.1"
        || root.profile_id != PROFILE_ID
        || root.beacon_id != BEACON_ID
        || root.scheme_id != SCHEME_ID
        || root.genesis_unix_seconds == 0
        || root.period_seconds == 0
        || root.source_references.is_empty()
        || root.source_retrieved_on != "2026-11-15"
    {
        return Err(VerificationError::TrustRoot(
            "trust-root manifest fields do not match the compiled Quicknet profile".into(),
        ));
    }
    let chain_hash = decode_hex::<32>(&root.chain_hash)
        .ok_or_else(|| VerificationError::TrustRoot("invalid chain hash in trust root".into()))?;
    Ok((root, chain_hash, digest_hex))
}

/// Strict Quicknet v1 relay payload. Unknown fields, including an unexpected
/// previous-signature link, are rejected rather than silently ignored.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuicknetBeaconResponse {
    pub round: u64,
    pub randomness: String,
    pub signature: String,
}

/// Separate predicates prevent a valid BLS signature from being mistaken for
/// proof of freshness or response chronology.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerificationReceipt {
    pub receipt_schema_version: String,
    pub profile_id: String,
    pub chain_hash: String,
    pub trust_root_manifest_sha256: String,
    pub scheme_id: String,
    pub beacon_id: String,
    pub requested_round: u64,
    pub returned_round: u64,
    pub signature_sha256: String,
    pub verified_randomness: String,
    pub published_randomness_matches_signature: bool,
    pub signature_valid_for_pinned_profile_and_round: bool,
    pub freshness_verified: bool,
    pub response_chronology_verified: bool,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerificationError {
    EmptyOrInvalidRound,
    RoundMismatch { requested: u64, returned: u64 },
    InvalidHex(&'static str),
    InvalidSignature(String),
    RandomnessMismatch,
    TrustRoot(String),
}

impl fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOrInvalidRound => {
                write!(formatter, "requested round must be greater than zero")
            }
            Self::RoundMismatch {
                requested,
                returned,
            } => write!(
                formatter,
                "relay returned round {returned}, but round {requested} was requested"
            ),
            Self::InvalidHex(field) => {
                write!(formatter, "{field} is not valid fixed-length hexadecimal")
            }
            Self::InvalidSignature(detail) => write!(
                formatter,
                "Quicknet signature verification failed: {detail}"
            ),
            Self::RandomnessMismatch => write!(
                formatter,
                "relay randomness does not match the randomness derived from the verified signature"
            ),
            Self::TrustRoot(detail) => write!(
                formatter,
                "pinned Quicknet trust-root initialization failed: {detail}"
            ),
        }
    }
}

impl std::error::Error for VerificationError {}

/// Verify an externally supplied relay response against the pinned Quicknet
/// trust root. No network access or clock access occurs in this function.
pub fn verify_quicknet_response(
    requested_round: u64,
    response: &QuicknetBeaconResponse,
) -> Result<VerificationReceipt, VerificationError> {
    if requested_round == 0 || response.round == 0 {
        return Err(VerificationError::EmptyOrInvalidRound);
    }
    if response.round != requested_round {
        return Err(VerificationError::RoundMismatch {
            requested: requested_round,
            returned: response.round,
        });
    }

    let signature =
        decode_hex::<48>(&response.signature).ok_or(VerificationError::InvalidHex("signature"))?;
    let published_randomness = decode_hex::<32>(&response.randomness)
        .ok_or(VerificationError::InvalidHex("randomness"))?;

    let (trust_root, chain_hash, trust_root_manifest_sha256) = load_trust_root()?;
    let public_key = decode_hex::<96>(&trust_root.public_key)
        .ok_or_else(|| VerificationError::TrustRoot("invalid public key in trust root".into()))?;
    let genesis_seed = decode_hex::<32>(&trust_root.genesis_seed)
        .ok_or_else(|| VerificationError::TrustRoot("invalid genesis seed in trust root".into()))?;
    let period = Period::from_secs(trust_root.period_seconds)
        .ok_or_else(|| VerificationError::TrustRoot("invalid round period in trust root".into()))?;
    let timeline = Timeline::new(
        UnixTimestamp::from_secs(trust_root.genesis_unix_seconds),
        period,
    );
    let verifier = AnyVerifier::new(
        SchemeId::BlsUnchainedG1Rfc9380,
        &public_key,
        timeline,
        Some(GenesisSeed::new(genesis_seed)),
        &trust_root.beacon_id,
        ChainHash::new(chain_hash),
    )
    .map_err(|error| VerificationError::TrustRoot(error.to_string()))?;

    // Quicknet is unchained. The verifier checks the requested round and BLS
    // signature against the independently pinned profile above.
    let verified = verifier
        .verify(
            Round::new(requested_round).ok_or(VerificationError::EmptyOrInvalidRound)?,
            &signature,
            None,
        )
        .map_err(|error| VerificationError::InvalidSignature(error.to_string()))?;

    let verified_randomness = *verified.randomness.get();
    let digest: [u8; 32] = Sha256::digest(signature).into();
    if digest != verified_randomness {
        return Err(VerificationError::RandomnessMismatch);
    }
    if published_randomness != verified_randomness {
        return Err(VerificationError::RandomnessMismatch);
    }

    Ok(VerificationReceipt {
        receipt_schema_version: "sympsi.quicknet-verification.v0.1".into(),
        profile_id: PROFILE_ID.into(),
        chain_hash: trust_root.chain_hash,
        trust_root_manifest_sha256,
        scheme_id: trust_root.scheme_id,
        beacon_id: BEACON_ID.into(),
        requested_round,
        returned_round: response.round,
        signature_sha256: encode_hex(&digest),
        verified_randomness: encode_hex(&verified_randomness),
        published_randomness_matches_signature: true,
        signature_valid_for_pinned_profile_and_round: true,
        freshness_verified: false,
        response_chronology_verified: false,
        claim_ceiling: "PINNED_QUICKNET_SIGNATURE_AND_ROUND_ONLY_NOT_FRESHNESS_OR_CHRONOLOGY"
            .into(),
    })
}

pub fn decode_hex<const N: usize>(value: &str) -> Option<[u8; N]> {
    if value.len() != N * 2 || !value.is_ascii() {
        return None;
    }
    let mut output = [0_u8; N];
    let bytes = value.as_bytes();
    for index in 0..N {
        let high = hex_nibble(bytes[index * 2])?;
        let low = hex_nibble(bytes[index * 2 + 1])?;
        output[index] = (high << 4) | low;
    }
    Some(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

pub fn encode_hex(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_answer_response() -> QuicknetBeaconResponse {
        QuicknetBeaconResponse {
            round: 1000,
            randomness: "fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd".into(),
            signature: "b44679b9a59af2ec876b1a6b1ad52ea9b1615fc3982b19576350f93447cb1125e342b73a8dd2bacbe47e4b6b63ed5e39".into(),
        }
    }

    #[test]
    fn published_quicknet_round_1000_signature_verifies() {
        let receipt = verify_quicknet_response(1000, &known_answer_response()).unwrap();
        assert_eq!(receipt.profile_id, PROFILE_ID);
        let (root, _, root_digest) = load_trust_root().unwrap();
        assert_eq!(receipt.chain_hash, root.chain_hash);
        assert_eq!(receipt.trust_root_manifest_sha256, root_digest);
        assert_eq!(receipt.scheme_id, SCHEME_ID);
        assert_eq!(
            receipt.verified_randomness,
            "fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd"
        );
        assert!(receipt.signature_valid_for_pinned_profile_and_round);
        assert!(receipt.published_randomness_matches_signature);
        assert!(!receipt.freshness_verified);
        assert!(!receipt.response_chronology_verified);
    }

    #[test]
    fn mutated_signature_is_rejected() {
        let mut response = known_answer_response();
        response.signature.replace_range(0..2, "00");
        assert!(matches!(
            verify_quicknet_response(1000, &response),
            Err(VerificationError::InvalidSignature(_))
        ));
    }

    #[test]
    fn mismatched_returned_round_is_rejected_before_signature_verification() {
        let response = known_answer_response();
        assert_eq!(
            verify_quicknet_response(1001, &response),
            Err(VerificationError::RoundMismatch {
                requested: 1001,
                returned: 1000
            })
        );
    }

    #[test]
    fn mismatched_published_randomness_is_rejected() {
        let mut response = known_answer_response();
        response.randomness.replace_range(0..2, "00");
        assert_eq!(
            verify_quicknet_response(1000, &response),
            Err(VerificationError::RandomnessMismatch)
        );
    }

    #[test]
    fn invalid_signature_encoding_is_rejected() {
        let mut response = known_answer_response();
        response.signature.pop();
        assert_eq!(
            verify_quicknet_response(1000, &response),
            Err(VerificationError::InvalidHex("signature"))
        );
    }

    #[test]
    fn round_zero_is_rejected() {
        let mut response = known_answer_response();
        response.round = 0;
        assert_eq!(
            verify_quicknet_response(0, &response),
            Err(VerificationError::EmptyOrInvalidRound)
        );
    }

    #[test]
    fn strict_relay_response_rejects_unknown_fields() {
        let bytes =
            br#"{"round":1000,"randomness":"00","signature":"00","previous_signature":"00"}"#;
        assert!(serde_json::from_slice::<QuicknetBeaconResponse>(bytes).is_err());
    }

    #[test]
    fn trust_root_manifest_hash_and_profile_are_pinned() {
        let (root, chain_hash, digest) = load_trust_root().unwrap();
        assert_eq!(digest, TRUST_ROOT_MANIFEST_SHA256);
        assert_eq!(root.profile_id, PROFILE_ID);
        assert_eq!(root.beacon_id, BEACON_ID);
        assert_eq!(root.scheme_id, SCHEME_ID);
        assert_eq!(
            encode_hex(&chain_hash),
            "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
        );
    }

    #[test]
    fn trust_root_profile_is_not_inferred_from_chain_hash_alone() {
        // The receipt binds an explicit scheme ID in addition to the chain hash.
        let receipt = verify_quicknet_response(1000, &known_answer_response()).unwrap();
        assert_eq!(receipt.scheme_id, SCHEME_ID);
        assert_eq!(receipt.beacon_id, BEACON_ID);
        assert_eq!(
            receipt.trust_root_manifest_sha256,
            TRUST_ROOT_MANIFEST_SHA256
        );
    }
}
