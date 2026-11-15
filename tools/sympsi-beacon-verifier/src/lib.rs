#![forbid(unsafe_code)]

//! Offline verifier for the pinned drand Quicknet BLS-unchained profile.
//!
//! This proves only that a supplied signature is valid for the selected
//! Quicknet trust root and requested round. It does not establish freshness,
//! beacon publication time, response-before-target chronology, or the
//! independence of any timestamp witness.

use drandite_core::any::{AnyVerifier, SchemeId};
use drandite_core::types::{ChainHash, GenesisSeed, Period, Round, UnixTimestamp};
use drandite_core::Timeline;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const PROFILE_ID: &str = "drand-quicknet-bls-unchained-g1-rfc9380-v1";
pub const CHAIN_HASH_HEX: &str =
    "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971";
pub const SCHEME_ID: &str = "bls-unchained-g1-rfc9380";
pub const BEACON_ID: &str = "quicknet";
pub const GENESIS_UNIX_SECONDS: u64 = 1_692_803_367;
pub const ROUND_PERIOD_SECONDS: u64 = 3;

const PUBLIC_KEY_HEX: &str = concat!(
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c",
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb",
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
);
const GENESIS_SEED_HEX: &str =
    "f477d5c89f21a17c863a7f937c6a6d15859414d2be09cd448d4279af331c5d3e";

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
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationReceipt {
    pub receipt_schema_version: String,
    pub profile_id: String,
    pub chain_hash: String,
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
            Self::EmptyOrInvalidRound => write!(formatter, "requested round must be greater than zero"),
            Self::RoundMismatch { requested, returned } => write!(
                formatter,
                "relay returned round {returned}, but round {requested} was requested"
            ),
            Self::InvalidHex(field) => write!(formatter, "{field} is not valid fixed-length hexadecimal"),
            Self::InvalidSignature(detail) => write!(formatter, "Quicknet signature verification failed: {detail}"),
            Self::RandomnessMismatch => write!(
                formatter,
                "relay randomness does not match the randomness derived from the verified signature"
            ),
            Self::TrustRoot(detail) => write!(formatter, "pinned Quicknet trust-root initialization failed: {detail}"),
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

    let signature = decode_hex::<48>(&response.signature)
        .ok_or(VerificationError::InvalidHex("signature"))?;
    let published_randomness = decode_hex::<32>(&response.randomness)
        .ok_or(VerificationError::InvalidHex("randomness"))?;

    let public_key = decode_hex::<96>(PUBLIC_KEY_HEX)
        .expect("constant Quicknet public key must have 96 bytes");
    let genesis_seed = decode_hex::<32>(GENESIS_SEED_HEX)
        .expect("constant Quicknet genesis seed must have 32 bytes");
    let chain_hash = decode_hex::<32>(CHAIN_HASH_HEX)
        .expect("constant Quicknet chain hash must have 32 bytes");

    let timeline = Timeline::new(
        UnixTimestamp::from_secs(GENESIS_UNIX_SECONDS),
        Period::from_secs(ROUND_PERIOD_SECONDS)
            .expect("pinned three-second period is valid"),
    );
    let verifier = AnyVerifier::new(
        SchemeId::BlsUnchainedG1Rfc9380,
        &public_key,
        timeline,
        Some(GenesisSeed::new(genesis_seed)),
        BEACON_ID,
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
        chain_hash: CHAIN_HASH_HEX.into(),
        scheme_id: SCHEME_ID.into(),
        beacon_id: BEACON_ID.into(),
        requested_round,
        returned_round: response.round,
        signature_sha256: encode_hex(&digest),
        verified_randomness: encode_hex(&verified_randomness),
        published_randomness_matches_signature: true,
        signature_valid_for_pinned_profile_and_round: true,
        freshness_verified: false,
        response_chronology_verified: false,
        claim_ceiling: "PINNED_QUICKNET_SIGNATURE_AND_ROUND_ONLY_NOT_FRESHNESS_OR_CHRONOLOGY".into(),
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
        assert_eq!(receipt.chain_hash, CHAIN_HASH_HEX);
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
    fn trust_root_profile_is_not_inferred_from_chain_hash_alone() {
        // The receipt binds an explicit scheme ID in addition to the chain hash.
        let receipt = verify_quicknet_response(1000, &known_answer_response()).unwrap();
        assert_eq!(receipt.scheme_id, "bls-unchained-g1-rfc9380");
        assert_eq!(receipt.beacon_id, "quicknet");
    }
}
