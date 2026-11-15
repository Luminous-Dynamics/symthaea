#![forbid(unsafe_code)]

//! Offline verification of drand Quicknet signatures against a pinned trust
//! root. This crate performs no network I/O and reads no clock.
//!
//! It establishes only:
//! - the returned round matches the caller's requested round;
//! - the signature verifies for the explicitly selected Quicknet scheme and
//!   pinned chain metadata;
//! - the round's scheduled Unix timestamp under the pinned timeline.
//!
//! It does NOT establish that the beacon is fresh, that the signature was not
//! published early, that a response commitment existed before the beacon, or
//! that a relay/witness/clock is independent. Those are separate protocol
//! gates owned by the SYMPsi custody layer.

use drandite_core::Timeline;
use drandite_core::any::{AnyVerifier, SchemeId};
use drandite_core::types::{ChainHash, GenesisSeed, Period, Round, UnixTimestamp};

/// Out-of-band Quicknet trust root. Do not populate these values from the same
/// untrusted relay response whose signature is being verified.
pub mod quicknet {
    pub const BEACON_ID: &str = "quicknet";
    pub const SCHEME_ID: &str = "bls-unchained-g1-rfc9380";
    pub const CHAIN_HASH: [u8; 32] = [
        0x52, 0xdb, 0x9b, 0xa7, 0x0e, 0x0c, 0xc0, 0xf6, 0xea, 0xf7, 0x80, 0x3d, 0xd0, 0x74, 0x47,
        0xa1, 0xf5, 0x47, 0x77, 0x35, 0xfd, 0x3f, 0x66, 0x17, 0x92, 0xba, 0x94, 0x60, 0x0c, 0x84,
        0xe9, 0x71,
    ];
    pub const PUBLIC_KEY: [u8; 96] = [
        0x83, 0xcf, 0x0f, 0x28, 0x96, 0xad, 0xee, 0x7e, 0xb8, 0xb5, 0xf0, 0x1f, 0xca, 0xd3, 0x91,
        0x22, 0x12, 0xc4, 0x37, 0xe0, 0x07, 0x3e, 0x91, 0x1f, 0xb9, 0x00, 0x22, 0xd3, 0xe7, 0x60,
        0x18, 0x3c, 0x8c, 0x4b, 0x45, 0x0b, 0x6a, 0x0a, 0x6c, 0x3a, 0xc6, 0xa5, 0x77, 0x6a, 0x2d,
        0x10, 0x64, 0x51, 0x0d, 0x1f, 0xec, 0x75, 0x8c, 0x92, 0x1c, 0xc2, 0x2b, 0x0e, 0x17, 0xe6,
        0x3a, 0xaf, 0x4b, 0xcb, 0x5e, 0xd6, 0x63, 0x04, 0xde, 0x9c, 0xf8, 0x09, 0xbd, 0x27, 0x4c,
        0xa7, 0x3b, 0xab, 0x4a, 0xf5, 0xa6, 0xe9, 0xc7, 0x6a, 0x4b, 0xc0, 0x9e, 0x76, 0xea, 0xe8,
        0x99, 0x1e, 0xf5, 0xec, 0xe4, 0x5a,
    ];
    pub const GENESIS_SEED: [u8; 32] = [
        0xf4, 0x77, 0xd5, 0xc8, 0x9f, 0x21, 0xa1, 0x7c, 0x86, 0x3a, 0x7f, 0x93, 0x7c, 0x6a, 0x6d,
        0x15, 0x85, 0x94, 0x14, 0xd2, 0xbe, 0x09, 0xcd, 0x44, 0x8d, 0x42, 0x79, 0xaf, 0x33, 0x1c,
        0x5d, 0x3e,
    ];
    pub const GENESIS_UNIX_SECONDS: i64 = 1_692_803_367;
    pub const PERIOD_SECONDS: u32 = 3;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedQuicknetBeacon {
    round: u64,
    randomness: [u8; 32],
    scheduled_unix_seconds: i64,
    chain_hash: [u8; 32],
}

impl VerifiedQuicknetBeacon {
    pub fn round(&self) -> u64 {
        self.round
    }

    /// drand beacon randomness is public. Never use it as a secret, key, or
    /// cryptographic entropy source.
    pub fn randomness(&self) -> &[u8; 32] {
        &self.randomness
    }

    /// Time assigned by the pinned Quicknet schedule, not a witnessed
    /// observation time and not proof that the beacon was released on time.
    pub fn scheduled_unix_seconds(&self) -> i64 {
        self.scheduled_unix_seconds
    }

    pub fn chain_hash(&self) -> &[u8; 32] {
        &self.chain_hash
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationError {
    ZeroRound,
    RequestedReturnedRoundMismatch,
    RoundOutsidePinnedTimeline,
    PinnedTrustRootInvalid,
    SignatureInvalid,
}

pub fn verify_quicknet_beacon(
    requested_round: u64,
    returned_round: u64,
    signature: &[u8; 48],
) -> Result<VerifiedQuicknetBeacon, VerificationError> {
    if requested_round == 0 || returned_round == 0 {
        return Err(VerificationError::ZeroRound);
    }
    if requested_round != returned_round {
        return Err(VerificationError::RequestedReturnedRoundMismatch);
    }

    let round = Round::new(requested_round).ok_or(VerificationError::ZeroRound)?;
    let timeline = pinned_timeline();
    let scheduled_at = timeline
        .checked_time_of(round)
        .ok_or(VerificationError::RoundOutsidePinnedTimeline)?;

    let verifier = pinned_verifier()?;
    let verified = verifier
        .verify(round, signature, None)
        .map_err(|_| VerificationError::SignatureInvalid)?;

    let randomness: [u8; 32] = verified.randomness.get();
    Ok(VerifiedQuicknetBeacon {
        round: requested_round,
        randomness,
        scheduled_unix_seconds: scheduled_at.as_secs(),
        chain_hash: quicknet::CHAIN_HASH,
    })
}

fn pinned_timeline() -> Timeline {
    Timeline::new(
        UnixTimestamp::from_secs(quicknet::GENESIS_UNIX_SECONDS),
        Period::from_secs(quicknet::PERIOD_SECONDS).expect("pinned Quicknet period is nonzero"),
    )
}

fn pinned_verifier() -> Result<AnyVerifier, VerificationError> {
    AnyVerifier::new(
        SchemeId::BlsUnchainedG1Rfc9380,
        &quicknet::PUBLIC_KEY,
        pinned_timeline(),
        Some(GenesisSeed::new(quicknet::GENESIS_SEED)),
        quicknet::BEACON_ID,
        ChainHash::new(quicknet::CHAIN_HASH),
    )
    .map_err(|_| VerificationError::PinnedTrustRootInvalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;

    const ROUND_1000_SIGNATURE: [u8; 48] = hex!(
        "b44679b9a59af2ec876b1a6b1ad52ea9b1615fc3982b19576350f93447cb1125         e342b73a8dd2bacbe47e4b6b63ed5e39"
    );
    const ROUND_1000_RANDOMNESS: [u8; 32] =
        hex!("fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd");

    #[test]
    fn published_quicknet_round_1000_signature_verifies() {
        let verified = verify_quicknet_beacon(1000, 1000, &ROUND_1000_SIGNATURE).unwrap();
        assert_eq!(verified.round(), 1000);
        assert_eq!(verified.randomness(), &ROUND_1000_RANDOMNESS);
        assert_eq!(verified.chain_hash(), &quicknet::CHAIN_HASH);
        assert_eq!(verified.scheduled_unix_seconds(), 1_692_806_364);
    }

    #[test]
    fn modified_signature_is_rejected() {
        let mut signature = ROUND_1000_SIGNATURE;
        signature[0] ^= 0x01;
        assert_eq!(
            verify_quicknet_beacon(1000, 1000, &signature),
            Err(VerificationError::SignatureInvalid)
        );
    }

    #[test]
    fn requested_and_returned_round_must_match() {
        assert_eq!(
            verify_quicknet_beacon(1001, 1000, &ROUND_1000_SIGNATURE),
            Err(VerificationError::RequestedReturnedRoundMismatch)
        );
    }

    #[test]
    fn zero_round_is_rejected_before_verification() {
        assert_eq!(
            verify_quicknet_beacon(0, 0, &ROUND_1000_SIGNATURE),
            Err(VerificationError::ZeroRound)
        );
    }

    #[test]
    fn wrong_public_key_cannot_reuse_pinned_chain_hash() {
        let mut wrong_key = quicknet::PUBLIC_KEY;
        wrong_key[0] ^= 0x01;
        let result = AnyVerifier::new(
            SchemeId::BlsUnchainedG1Rfc9380,
            &wrong_key,
            pinned_timeline(),
            Some(GenesisSeed::new(quicknet::GENESIS_SEED)),
            quicknet::BEACON_ID,
            ChainHash::new(quicknet::CHAIN_HASH),
        );
        assert!(result.is_err());
    }

    #[test]
    fn wrong_beacon_id_cannot_reuse_pinned_chain_hash() {
        let result = AnyVerifier::new(
            SchemeId::BlsUnchainedG1Rfc9380,
            &quicknet::PUBLIC_KEY,
            pinned_timeline(),
            Some(GenesisSeed::new(quicknet::GENESIS_SEED)),
            "not-quicknet",
            ChainHash::new(quicknet::CHAIN_HASH),
        );
        assert!(result.is_err());
    }

    #[test]
    fn freshness_and_commitment_chronology_are_not_implied_by_verification() {
        // A historical round still verifies. The verifier intentionally has
        // no clock and makes no claim about when this signature was released.
        let verified = verify_quicknet_beacon(1000, 1000, &ROUND_1000_SIGNATURE).unwrap();
        assert_eq!(verified.scheduled_unix_seconds(), 1_692_806_364);
    }
}
