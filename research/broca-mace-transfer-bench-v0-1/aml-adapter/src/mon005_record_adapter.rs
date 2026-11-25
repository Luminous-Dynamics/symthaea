// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Source-matched adapter for the MON-005 SFEB v1 outer evidence record.
//!
//! The MACE digest is nested in the record payload. MON-005 hashes the outer
//! record preimage with its own domain, then RFC 9162 hashes the 32-byte
//! record digest as a Merkle leaf. This module prepares and re-verifies that
//! exact boundary; it does not append to a log or authenticate a checkpoint.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::mace_evaluation_anchor_payload::{
    parse_and_verify_mace_evaluation_anchor_payload_v1,
    MaceEvaluationAnchorPayloadRejectV1, MaceEvaluationAnchorPayloadV1,
    VerifiedMaceEvaluationAnchorPayloadV1,
};

pub const MON005_EVIDENCE_RECORD_SCHEMA_V1: &str = "sfeb-evidence-record-v1";
pub const MON005_MACE_ANCHOR_PAYLOAD_SCHEMA_V1: &str =
    "mace-mon005-evaluation-anchor-v1";
pub const MON005_MACE_ANCHOR_KIND_V1: &str = "mace-evaluation-anchor";
pub const MON005_RECORD_DOMAIN_V1: &[u8] = b"SFEB-RECORD-V1\0";
pub const MON005_GENESIS_DOMAIN_V1: &[u8] = b"SFEB-GENESIS-V1\0";
pub const MON005_MAX_RECORD_BYTES_V1: usize = 1_048_576;
pub const MON005_MAX_SAFE_SEQUENCE_V1: u64 = 9_007_199_254_740_991;

/// The MACE-specific value inside MON-005's generic `payload` field.
/// The complete inner payload is retained so a consumer can retrieve and
/// re-verify it; its digest alone would not be sufficient for that purpose.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Mon005MaceAnchorPayloadV1 {
    pub kind: String,
    pub schema: String,
    pub mace_payload: MaceEvaluationAnchorPayloadV1,
    pub mace_payload_sha256: String,
}

/// MON-005 v1 evidence-record body. The recordHash property is deliberately
/// excluded from this body and is computed over the canonical body bytes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Mon005EvidenceRecordV1 {
    pub schema: String,
    pub tenant_id: String,
    pub sequence: u64,
    pub previous_record_hash: String,
    pub payload: Mon005MaceAnchorPayloadV1,
    pub record_hash: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Mon005EvidenceRecordPreimageV1<'a> {
    schema: &'static str,
    tenant_id: &'a str,
    sequence: u64,
    previous_record_hash: &'a str,
    payload: &'a Mon005MaceAnchorPayloadV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mon005MaceRecordRejectV1 {
    InvalidTenantId,
    InvalidSequence,
    InvalidDigest,
    WrongSchema,
    WrongTenant,
    WrongSequence,
    WrongPreviousRecordHash,
    InvalidRecordJson,
    RecordTooLarge,
    CanonicalizationFailed,
    RecordHashMismatch,
    WrongMacePayloadKind,
    WrongMacePayloadSchema,
    MacePayloadRejected(MaceEvaluationAnchorPayloadRejectV1),
}

/// Prepared bytes and hashes to submit through the MON-005 append API.
/// This is not proof of append, inclusion, quorum, or external anchoring.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedMon005MaceAnchorRecordV1 {
    record: Mon005EvidenceRecordV1,
    canonical_body: Vec<u8>,
    canonical_record: Vec<u8>,
    merkle_leaf_sha256: String,
}

impl PreparedMon005MaceAnchorRecordV1 {
    pub fn record(&self) -> &Mon005EvidenceRecordV1 {
        &self.record
    }

    pub fn canonical_body(&self) -> &[u8] {
        &self.canonical_body
    }

    /// Canonical full record bytes, including the recordHash result field.
    pub fn canonical_record(&self) -> &[u8] {
        &self.canonical_record
    }

    pub fn record_sha256(&self) -> &str {
        &self.record.record_hash
    }

    /// RFC 9162 leaf: SHA-256(0x00 || bytes(recordHash)).
    pub fn merkle_leaf_sha256(&self) -> &str {
        &self.merkle_leaf_sha256
    }

    pub fn mace_payload_sha256(&self) -> &str {
        &self.record.payload.mace_payload_sha256
    }
}

/// Consumer-verified outer-record bytes and nested MACE payload. This proves
/// the record digest and expected append-position bindings only. A caller must
/// still verify an authenticated MON-005 checkpoint, witness roster/quorum,
/// inclusion proof, and consistency from a separately pinned prior checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedMon005MaceAnchorRecordV1 {
    record: Mon005EvidenceRecordV1,
    mace_payload: VerifiedMaceEvaluationAnchorPayloadV1,
    merkle_leaf_sha256: String,
}

impl VerifiedMon005MaceAnchorRecordV1 {
    pub fn record(&self) -> &Mon005EvidenceRecordV1 {
        &self.record
    }

    pub fn mace_payload(&self) -> &VerifiedMaceEvaluationAnchorPayloadV1 {
        &self.mace_payload
    }

    pub fn record_sha256(&self) -> &str {
        &self.record.record_hash
    }

    pub fn merkle_leaf_sha256(&self) -> &str {
        &self.merkle_leaf_sha256
    }
}

/// MON-005's tenant-specific genesis predecessor:
/// SHA-256("SFEB-GENESIS-V1\0" || tenant_id UTF-8).
pub fn mon005_genesis_record_hash_v1(
    tenant_id: &str,
) -> Result<String, Mon005MaceRecordRejectV1> {
    validate_tenant_id(tenant_id)?;
    let mut hasher = Sha256::new();
    hasher.update(MON005_GENESIS_DOMAIN_V1);
    hasher.update(tenant_id.as_bytes());
    Ok(hex_digest(&hasher.finalize()))
}

/// Prepare the precise MON-005 v1 record from a verified MACE capability and
/// an append cursor obtained by the caller. For sequence 1, the predecessor
/// must equal MON-005's tenant-specific genesis digest. For later sequences,
/// the predecessor must be a lowercase SHA-256 digest obtained from a
/// previously trusted MON-005 head; this function cannot authenticate it.
pub fn prepare_mon005_mace_anchor_record_v1(
    tenant_id: &str,
    sequence: u64,
    previous_record_hash: &str,
    verified_mace_payload: &VerifiedMaceEvaluationAnchorPayloadV1,
) -> Result<PreparedMon005MaceAnchorRecordV1, Mon005MaceRecordRejectV1> {
    validate_tenant_id(tenant_id)?;
    validate_sequence(sequence)?;
    validate_sha256(previous_record_hash)?;

    if sequence == 1 && previous_record_hash != mon005_genesis_record_hash_v1(tenant_id)? {
        return Err(Mon005MaceRecordRejectV1::WrongPreviousRecordHash);
    }

    // Re-parse the capability's content against its digest at this boundary.
    let mace_payload = verified_mace_payload.payload().clone();
    let mace_payload_sha256 = verified_mace_payload.payload_sha256().to_owned();
    let mace_wire = serde_json::to_vec(&mace_payload)
        .map_err(|_| Mon005MaceRecordRejectV1::CanonicalizationFailed)?;
    parse_and_verify_mace_evaluation_anchor_payload_v1(
        &mace_wire,
        &mace_payload_sha256,
    )
    .map_err(Mon005MaceRecordRejectV1::MacePayloadRejected)?;

    let payload = Mon005MaceAnchorPayloadV1 {
        kind: MON005_MACE_ANCHOR_KIND_V1.to_owned(),
        schema: MON005_MACE_ANCHOR_PAYLOAD_SCHEMA_V1.to_owned(),
        mace_payload,
        mace_payload_sha256,
    };
    let preimage = Mon005EvidenceRecordPreimageV1 {
        schema: MON005_EVIDENCE_RECORD_SCHEMA_V1,
        tenant_id,
        sequence,
        previous_record_hash,
        payload: &payload,
    };
    let canonical_body = serde_json_canonicalizer::to_vec(&preimage)
        .map_err(|_| Mon005MaceRecordRejectV1::CanonicalizationFailed)?;
    if canonical_body.len() > MON005_MAX_RECORD_BYTES_V1 {
        return Err(Mon005MaceRecordRejectV1::RecordTooLarge);
    }
    let record_hash = record_digest_v1(&canonical_body);
    let record = Mon005EvidenceRecordV1 {
        schema: MON005_EVIDENCE_RECORD_SCHEMA_V1.to_owned(),
        tenant_id: tenant_id.to_owned(),
        sequence,
        previous_record_hash: previous_record_hash.to_owned(),
        payload,
        record_hash,
    };
    let canonical_record = serde_json_canonicalizer::to_vec(&record)
        .map_err(|_| Mon005MaceRecordRejectV1::CanonicalizationFailed)?;
    if canonical_record.len() > MON005_MAX_RECORD_BYTES_V1 {
        return Err(Mon005MaceRecordRejectV1::RecordTooLarge);
    }
    let merkle_leaf_sha256 = merkle_leaf_digest_v1(&record.record_hash)?;

    Ok(PreparedMon005MaceAnchorRecordV1 {
        record,
        canonical_body,
        canonical_record,
        merkle_leaf_sha256,
    })
}

/// Verify a retrieved MON-005 outer record against consumer-pinned tenant,
/// sequence, predecessor and MACE payload digest. This function verifies
/// record content and its Merkle leaf value; it does not verify a proof path
/// or authenticate the checkpoint containing that leaf.
pub fn parse_and_verify_mon005_mace_anchor_record_v1(
    wire: &[u8],
    expected_tenant_id: &str,
    expected_sequence: u64,
    expected_previous_record_hash: &str,
    expected_mace_payload_sha256: &str,
) -> Result<VerifiedMon005MaceAnchorRecordV1, Mon005MaceRecordRejectV1> {
    if wire.len() > MON005_MAX_RECORD_BYTES_V1 {
        return Err(Mon005MaceRecordRejectV1::RecordTooLarge);
    }
    validate_tenant_id(expected_tenant_id)?;
    validate_sequence(expected_sequence)?;
    validate_sha256(expected_previous_record_hash)?;
    validate_sha256(expected_mace_payload_sha256)?;

    let record: Mon005EvidenceRecordV1 = serde_json::from_slice(wire)
        .map_err(|_| Mon005MaceRecordRejectV1::InvalidRecordJson)?;
    if record.schema != MON005_EVIDENCE_RECORD_SCHEMA_V1 {
        return Err(Mon005MaceRecordRejectV1::WrongSchema);
    }
    if record.tenant_id != expected_tenant_id {
        return Err(Mon005MaceRecordRejectV1::WrongTenant);
    }
    if record.sequence != expected_sequence {
        return Err(Mon005MaceRecordRejectV1::WrongSequence);
    }
    if record.previous_record_hash != expected_previous_record_hash {
        return Err(Mon005MaceRecordRejectV1::WrongPreviousRecordHash);
    }
    if record.sequence == 1
        && record.previous_record_hash != mon005_genesis_record_hash_v1(expected_tenant_id)?
    {
        return Err(Mon005MaceRecordRejectV1::WrongPreviousRecordHash);
    }
    if record.payload.kind != MON005_MACE_ANCHOR_KIND_V1 {
        return Err(Mon005MaceRecordRejectV1::WrongMacePayloadKind);
    }
    if record.payload.schema != MON005_MACE_ANCHOR_PAYLOAD_SCHEMA_V1 {
        return Err(Mon005MaceRecordRejectV1::WrongMacePayloadSchema);
    }
    if record.payload.mace_payload_sha256 != expected_mace_payload_sha256 {
        return Err(Mon005MaceRecordRejectV1::MacePayloadRejected(
            MaceEvaluationAnchorPayloadRejectV1::PayloadDigestMismatch,
        ));
    }
    validate_sha256(&record.record_hash)?;

    let preimage = Mon005EvidenceRecordPreimageV1 {
        schema: MON005_EVIDENCE_RECORD_SCHEMA_V1,
        tenant_id: &record.tenant_id,
        sequence: record.sequence,
        previous_record_hash: &record.previous_record_hash,
        payload: &record.payload,
    };
    let canonical_body = serde_json_canonicalizer::to_vec(&preimage)
        .map_err(|_| Mon005MaceRecordRejectV1::CanonicalizationFailed)?;
    if canonical_body.len() > MON005_MAX_RECORD_BYTES_V1 {
        return Err(Mon005MaceRecordRejectV1::RecordTooLarge);
    }
    if record_digest_v1(&canonical_body) != record.record_hash {
        return Err(Mon005MaceRecordRejectV1::RecordHashMismatch);
    }

    let mace_wire = serde_json::to_vec(&record.payload.mace_payload)
        .map_err(|_| Mon005MaceRecordRejectV1::CanonicalizationFailed)?;
    let mace_payload = parse_and_verify_mace_evaluation_anchor_payload_v1(
        &mace_wire,
        expected_mace_payload_sha256,
    )
    .map_err(Mon005MaceRecordRejectV1::MacePayloadRejected)?;
    let merkle_leaf_sha256 = merkle_leaf_digest_v1(&record.record_hash)?;

    Ok(VerifiedMon005MaceAnchorRecordV1 {
        record,
        mace_payload,
        merkle_leaf_sha256,
    })
}

/// Compute MON-005's record digest. The recordHash output field is not part
/// of the input: only schema, tenantId, sequence, previousRecordHash and
/// payload are canonicalized and domain-separated.
fn record_digest_v1(canonical_body: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MON005_RECORD_DOMAIN_V1);
    hasher.update(canonical_body);
    hex_digest(&hasher.finalize())
}

/// Compute RFC 9162 SHA-256(0x00 || recordHash_bytes).
fn merkle_leaf_digest_v1(
    record_hash: &str,
) -> Result<String, Mon005MaceRecordRejectV1> {
    let record_bytes = decode_sha256(record_hash)?;
    let mut hasher = Sha256::new();
    hasher.update([0x00]);
    hasher.update(record_bytes);
    Ok(hex_digest(&hasher.finalize()))
}

fn validate_tenant_id(value: &str) -> Result<(), Mon005MaceRecordRejectV1> {
    if value.is_empty()
        || value.len() > 256
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(Mon005MaceRecordRejectV1::InvalidTenantId);
    }
    Ok(())
}

fn validate_sequence(value: u64) -> Result<(), Mon005MaceRecordRejectV1> {
    if value == 0 || value > MON005_MAX_SAFE_SEQUENCE_V1 {
        return Err(Mon005MaceRecordRejectV1::InvalidSequence);
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), Mon005MaceRecordRejectV1> {
    decode_sha256(value).map(|_| ())
}

fn decode_sha256(value: &str) -> Result<[u8; 32], Mon005MaceRecordRejectV1> {
    if value.len() != 64 {
        return Err(Mon005MaceRecordRejectV1::InvalidDigest);
    }
    let mut output = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or(Mon005MaceRecordRejectV1::InvalidDigest)?;
        let low = hex_nibble(pair[1]).ok_or(Mon005MaceRecordRejectV1::InvalidDigest)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rfc9162_proofs::verify_rfc9162_inclusion_v1;
    use crate::sealed_evaluation_manifest::test_verified_sealed_evaluation_fixture_v1;

    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const DIGEST_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DIGEST_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const DIGEST_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const DIGEST_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const DIGEST_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn verified_mace_payload() -> VerifiedMaceEvaluationAnchorPayloadV1 {
        let (manifest, verified_manifest, verified_witness) =
            test_verified_sealed_evaluation_fixture_v1();
        let expected = crate::mace_evaluation_anchor_payload::MaceEvaluationAnchorExpectedBindingV1 {
            evaluator_profile_sha256: DIGEST_B,
            verifier_profile_sha256: DIGEST_C,
            access_control_profile_sha256: DIGEST_D,
            sealed_store_profile_sha256: DIGEST_E,
            candidate_access_profile_sha256: DIGEST_1,
            reveal_protocol_sha256: DIGEST_2,
            trace_seal_protocol_sha256: DIGEST_F,
        };
        crate::mace_evaluation_anchor_payload::build_mace_evaluation_anchor_payload_v1(
            &manifest,
            &verified_manifest,
            &verified_witness,
            &expected,
        )
        .unwrap()
    }

    fn prepared_fixture() -> PreparedMon005MaceAnchorRecordV1 {
        let previous = mon005_genesis_record_hash_v1("mace-test").unwrap();
        prepare_mon005_mace_anchor_record_v1(
            "mace-test",
            1,
            &previous,
            &verified_mace_payload(),
        )
        .unwrap()
    }

    #[test]
    fn matches_mon005_v1_record_and_rfc9162_leaf_reference_vectors() {
        let prepared = prepared_fixture();
        assert_eq!(
            prepared.mace_payload_sha256(),
            "117579a8803ecedd6b5eee4efbc73c4fe66ce9088c7687a5b8785ac4815d1154"
        );
        assert_eq!(
            mon005_genesis_record_hash_v1("mace-test").unwrap(),
            "646ed2e98580c76d575fc3b8fdd20bcc754a7828a6588fc2f64d1ebd62739228"
        );
        assert_eq!(
            prepared.record_sha256(),
            "075a9638686c559c547e1c1cbeb77671c833eeef5e52cf9daa36ad967d72d26c"
        );
        assert_eq!(
            prepared.merkle_leaf_sha256(),
            "46a32360e04e669cadb1b917781405c751d5cbbfedd70c2cdb1e37b6e5e1ca8c"
        );
        assert_eq!(prepared.record().schema, MON005_EVIDENCE_RECORD_SCHEMA_V1);
        assert_eq!(prepared.record().payload.kind, MON005_MACE_ANCHOR_KIND_V1);
    }

    #[test]
    fn consumer_reverifies_record_and_embedded_mace_payload() {
        let prepared = prepared_fixture();
        let verified = parse_and_verify_mon005_mace_anchor_record_v1(
            prepared.canonical_record(),
            "mace-test",
            1,
            &mon005_genesis_record_hash_v1("mace-test").unwrap(),
            "117579a8803ecedd6b5eee4efbc73c4fe66ce9088c7687a5b8785ac4815d1154",
        )
        .unwrap();
        assert_eq!(verified.record_sha256(), prepared.record_sha256());
        assert_eq!(verified.merkle_leaf_sha256(), prepared.merkle_leaf_sha256());
        assert_eq!(
            verified.mace_payload().payload().suite_id,
            "mace-t10-integrity"
        );
    }

    #[test]
    fn leaf_is_accepted_by_the_existing_rfc9162_verifier() {
        let prepared = prepared_fixture();
        let leaf = decode_sha256(prepared.merkle_leaf_sha256()).unwrap();
        verify_rfc9162_inclusion_v1(&leaf, 0, 1, &[], &leaf).unwrap();
    }

    #[test]
    fn wrong_tenant_sequence_or_predecessor_is_rejected() {
        let prepared = prepared_fixture();
        let previous = mon005_genesis_record_hash_v1("mace-test").unwrap();
        assert_eq!(
            parse_and_verify_mon005_mace_anchor_record_v1(
                prepared.canonical_record(), "other-tenant", 1, &previous,
                prepared.mace_payload_sha256(),
            ),
            Err(Mon005MaceRecordRejectV1::WrongTenant)
        );
        assert_eq!(
            parse_and_verify_mon005_mace_anchor_record_v1(
                prepared.canonical_record(), "mace-test", 2, &previous,
                prepared.mace_payload_sha256(),
            ),
            Err(Mon005MaceRecordRejectV1::WrongSequence)
        );
        assert_eq!(
            parse_and_verify_mon005_mace_anchor_record_v1(
                prepared.canonical_record(), "mace-test", 1, &"0".repeat(64),
                prepared.mace_payload_sha256(),
            ),
            Err(Mon005MaceRecordRejectV1::WrongPreviousRecordHash)
        );
    }

    #[test]
    fn first_record_must_use_tenant_specific_genesis_hash() {
        assert_eq!(
            prepare_mon005_mace_anchor_record_v1(
                "mace-test", 1, &"0".repeat(64), &verified_mace_payload(),
            ),
            Err(Mon005MaceRecordRejectV1::WrongPreviousRecordHash)
        );
    }

    #[test]
    fn record_digest_or_nested_payload_mutation_is_rejected() {
        let prepared = prepared_fixture();
        let mut changed = prepared.record().clone();
        changed.payload.mace_payload.suite_id = "mutated-suite".to_owned();
        let changed_wire = serde_json::to_vec(&changed).unwrap();
        assert_eq!(
            parse_and_verify_mon005_mace_anchor_record_v1(
                &changed_wire,
                "mace-test",
                1,
                &mon005_genesis_record_hash_v1("mace-test").unwrap(),
                prepared.mace_payload_sha256(),
            ),
            Err(Mon005MaceRecordRejectV1::RecordHashMismatch)
        );
    }

    #[test]
    fn wrong_mace_digest_is_rejected_even_when_outer_record_is_valid() {
        let prepared = prepared_fixture();
        assert_eq!(
            parse_and_verify_mon005_mace_anchor_record_v1(
                prepared.canonical_record(),
                "mace-test",
                1,
                &mon005_genesis_record_hash_v1("mace-test").unwrap(),
                &"0".repeat(64),
            ),
            Err(Mon005MaceRecordRejectV1::MacePayloadRejected(
                MaceEvaluationAnchorPayloadRejectV1::PayloadDigestMismatch
            ))
        );
    }

    #[test]
    fn duplicate_and_unknown_outer_fields_are_rejected() {
        let prepared = prepared_fixture();
        let wire = std::str::from_utf8(prepared.canonical_record()).unwrap();
        let duplicate = wire.replacen(
            "\"sequence\":1",
            "\"sequence\":1,\"sequence\":1",
            1,
        );
        assert_eq!(
            parse_and_verify_mon005_mace_anchor_record_v1(
                duplicate.as_bytes(),
                "mace-test",
                1,
                &mon005_genesis_record_hash_v1("mace-test").unwrap(),
                prepared.mace_payload_sha256(),
            ),
            Err(Mon005MaceRecordRejectV1::InvalidRecordJson)
        );
        let unknown = wire.replacen(
            "\"schema\":\"sfeb-evidence-record-v1\"",
            "\"unexpected\":true,\"schema\":\"sfeb-evidence-record-v1\"",
            1,
        );
        assert_eq!(
            parse_and_verify_mon005_mace_anchor_record_v1(
                unknown.as_bytes(),
                "mace-test",
                1,
                &mon005_genesis_record_hash_v1("mace-test").unwrap(),
                prepared.mace_payload_sha256(),
            ),
            Err(Mon005MaceRecordRejectV1::InvalidRecordJson)
        );
    }
}
