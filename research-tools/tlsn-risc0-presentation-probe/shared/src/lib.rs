#![no_std]
extern crate alloc;

use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

pub const PROBE_CONTRACT_VERSION_V1: u32 = 1;
pub const MAX_PRESENTATION_BYTES_V1: usize = 8 * 1024 * 1024;
pub const MAX_TRUST_ROOTS_V1: usize = 16;
pub const MAX_TRUST_ROOT_DER_BYTES_V1: usize = 64 * 1024;
pub const MAX_TRUST_ROOT_PROFILE_BYTES_V1: usize = 256 * 1024;
pub const MAX_AUTHENTICATED_RANGES_V1: usize = 100_000;

pub const TLS_VERSION_1_2_V1: u8 = 12;
pub const TLS_VERSION_1_3_V1: u8 = 13;

pub const NOTARY_KEY_K256_V1: u8 = 1;
pub const NOTARY_KEY_P256_V1: u8 = 2;

/// Verifier-authorized request to validate one TLSNotary Presentation.
///
/// The relying verifier must compare every committed journal field with its
/// own request. Input syntax accepted by the guest is not itself a trust root.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationVerifyInputV1 {
    pub contract_version: u32,
    pub serialized_presentation: Vec<u8>,
    pub expected_notary_key_algorithm_id: u8,
    pub expected_notary_key_bytes: Vec<u8>,
    pub expected_server_dns_name: String,
    pub expected_tls_version_code: u8,
    pub expected_received_transcript_bytes: u32,
    pub trust_root_der: Vec<Vec<u8>>,
    pub expected_trust_root_profile_digest: [u8; 32],
}

/// One authenticated received-direction interval, half-open.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticatedRangeV1 {
    pub start: u32,
    pub end: u32,
}

/// Public journal emitted by the isolated Presentation-verification guest.
///
/// Transcript commitment records are intentionally absent: Beta.1 does not
/// expose the authenticated commitment list through PresentationOutput's
/// public API. See Symthaea #12373.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationVerifyJournalV1 {
    pub contract_version: u32,
    pub attestation_id: [u8; 16],
    pub attestation_header_digest: [u8; 32],
    pub notary_key_algorithm_id: u8,
    pub notary_key_fingerprint: [u8; 32],
    pub server_dns_name: String,
    pub tls_version_code: u8,
    pub connection_started_unix_seconds: u64,
    pub sent_transcript_bytes: u32,
    pub received_transcript_bytes: u32,
    pub trust_root_profile_digest: [u8; 32],
    pub authenticated_received_ranges: Vec<AuthenticatedRangeV1>,
}
