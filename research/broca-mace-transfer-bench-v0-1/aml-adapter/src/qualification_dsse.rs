// SPDX-License-Identifier: AGPL-3.0-or-later

//! DSSE / in-toto transport for independently issued MACE qualifications.
//!
//! The signed bytes are the DSSE PAE of an in-toto Statement v1. The
//! qualification predicate is application-specific and versioned. DSSE's
//! keyid is only a lookup hint; MACE also requires it to equal the key ID
//! inside the signed predicate and then checks it against the pinned profile.

use std::collections::BTreeMap;
use std::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::de::{Error as DeError, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

use crate::evidence_qualification_gate::{
    EvidenceQualificationBindingV1,
    EvidenceQualificationReceiptV1,
    EvidenceQualificationRejectV1,
    EvidenceQualificationSignatureVerifierV1,
    VerifiedEvidenceQualificationV1,
};
use crate::qualification_trust::PinnedEd25519QualificationVerifierV1;

pub const DSSE_PAYLOAD_TYPE_V1: &str = "application/vnd.in-toto+json";
pub const IN_TOTO_STATEMENT_TYPE_V1: &str = "https://in-toto.io/Statement/v1";
pub const MACE_QUALIFICATION_PREDICATE_TYPE_V1: &str =
    "https://luminousdynamics.io/attestations/mace/evidence-qualification/v1";

const MAX_DSSE_ENVELOPE_BYTES_V1: usize = 12 * 1024 * 1024;
const MAX_DSSE_PAYLOAD_BYTES_V1: usize = 8 * 1024 * 1024;

/// Construct the exact DSSE pre-authentication encoding (PAE).
///
/// Lengths are UTF-8 byte lengths, not character counts. The PAE authenticates
/// both the opaque payload type and the exact serialized payload bytes.
pub fn dsse_pae_v1(payload_type: &str, payload: &[u8]) -> Vec<u8> {
    let type_bytes = payload_type.as_bytes();
    let prefix = format!(
        "DSSEv1 {} {} {} ",
        type_bytes.len(),
        payload_type,
        payload.len()
    );
    let mut result = Vec::with_capacity(prefix.len() + payload.len());
    result.extend_from_slice(prefix.as_bytes());
    result.extend_from_slice(payload);
    result
}

/// Serialize a MACE qualification as an in-toto Statement v1 payload.
///
/// The candidate-set commitment is the immutable Statement subject. The
/// qualification receipt is the typed predicate. Its inline signature field
/// is deliberately removed: DSSE carries the signature in the outer envelope.
pub fn in_toto_qualification_payload_v1(
    receipt: &EvidenceQualificationReceiptV1,
) -> Result<Vec<u8>, EvidenceQualificationRejectV1> {
    let mut predicate = receipt.to_json_value();
    predicate
        .as_object_mut()
        .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?
        .remove("signature")
        .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;

    let subject_name = format!(
        "urn:luminous:mace:candidate-set:v1:{}",
        receipt.candidate_set_sha256
    );
    let statement = serde_json::json!({
        "_type": IN_TOTO_STATEMENT_TYPE_V1,
        "subject": [{
            "name": subject_name,
            "digest": {
                "sha256": receipt.candidate_set_sha256
            }
        }],
        "predicateType": MACE_QUALIFICATION_PREDICATE_TYPE_V1,
        "predicate": predicate
    });
    serde_json::to_vec(&statement)
        .map_err(|_| EvidenceQualificationRejectV1::DsseInvalidEnvelope)
}

/// Encode the single-signature DSSE JSON envelope used by this MACE v1
/// profile. DSSE itself permits multiple signatures; MACE v1 intentionally
/// rejects them until a threshold/role policy is specified.
pub fn encode_dsse_envelope_v1(
    payload: &[u8],
    key_id: &str,
    signature: &[u8],
) -> Result<String, EvidenceQualificationRejectV1> {
    if payload.len() > MAX_DSSE_PAYLOAD_BYTES_V1 || signature.len() != 64
        || key_id.trim().is_empty()
    {
        return Err(EvidenceQualificationRejectV1::DsseInvalidEnvelope);
    }
    let envelope = serde_json::json!({
        "payloadType": DSSE_PAYLOAD_TYPE_V1,
        "payload": STANDARD.encode(payload),
        "signatures": [{
            "keyid": key_id,
            "sig": STANDARD.encode(signature)
        }]
    });
    serde_json::to_string(&envelope)
        .map_err(|_| EvidenceQualificationRejectV1::DsseInvalidEnvelope)
}

/// A verifier prepared from one exact DSSE envelope and one currently pinned
/// trust profile. Reconstruct this verifier from the envelope and fresh trust
/// profile for every use-time check; do not retain a stale verifier across
/// key/profile rotation.
pub struct DsseQualificationSignatureVerifierV1<'a> {
    pinned: &'a PinnedEd25519QualificationVerifierV1,
    receipt: EvidenceQualificationReceiptV1,
    key_id: String,
    signature: Vec<u8>,
    pae: Vec<u8>,
}

impl<'a> DsseQualificationSignatureVerifierV1<'a> {
    pub fn from_envelope_json(
        envelope_json: &str,
        pinned: &'a PinnedEd25519QualificationVerifierV1,
    ) -> Result<Self, EvidenceQualificationRejectV1> {
        if envelope_json.len() > MAX_DSSE_ENVELOPE_BYTES_V1 {
            return Err(EvidenceQualificationRejectV1::DssePayloadTooLarge);
        }
        let envelope_value = parse_strict_json(envelope_json.as_bytes())?;
        let envelope = envelope_value
            .as_object()
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if envelope.len() != 3 {
            return Err(EvidenceQualificationRejectV1::DsseInvalidEnvelope);
        }

        let payload_type = envelope
            .get("payloadType")
            .and_then(Value::as_str)
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if payload_type != DSSE_PAYLOAD_TYPE_V1 {
            return Err(EvidenceQualificationRejectV1::DssePayloadTypeMismatch);
        }

        let payload_b64 = envelope
            .get("payload")
            .and_then(Value::as_str)
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        let payload = decode_canonical_base64(payload_b64)?;
        if payload.len() > MAX_DSSE_PAYLOAD_BYTES_V1 {
            return Err(EvidenceQualificationRejectV1::DssePayloadTooLarge);
        }

        let signatures = envelope
            .get("signatures")
            .and_then(Value::as_array)
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if signatures.len() != 1 {
            return Err(EvidenceQualificationRejectV1::DsseSignatureCountUnsupported);
        }
        let signature_object = signatures[0]
            .as_object()
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if signature_object.len() != 2 {
            return Err(EvidenceQualificationRejectV1::DsseInvalidEnvelope);
        }
        let key_id = signature_object
            .get("keyid")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty() && value.len() <= 256)
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?
            .to_owned();
        let signature_b64 = signature_object
            .get("sig")
            .and_then(Value::as_str)
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        let signature = decode_canonical_base64(signature_b64)?;
        if signature.len() != 64 {
            return Err(EvidenceQualificationRejectV1::InvalidSignatureEncoding);
        }

        let statement_value = parse_strict_json(&payload)?;
        let statement = statement_value
            .as_object()
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if statement.len() != 4
            || statement.get("_type").and_then(Value::as_str)
                != Some(IN_TOTO_STATEMENT_TYPE_V1)
            || statement.get("predicateType").and_then(Value::as_str)
                != Some(MACE_QUALIFICATION_PREDICATE_TYPE_V1)
        {
            return Err(EvidenceQualificationRejectV1::DssePayloadTypeMismatch);
        }

        let predicate = statement
            .get("predicate")
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        let predicate_object = predicate
            .as_object()
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if predicate_object.len() != 19 {
            return Err(EvidenceQualificationRejectV1::DsseInvalidEnvelope);
        }

        // Reconstruct the strict MACE receipt from the signed predicate and
        // the outer DSSE signature. The envelope's keyid must also be present
        // inside the signed predicate, so it is not trusted as an authority.
        let mut receipt_value = predicate.clone();
        let receipt_object = receipt_value
            .as_object_mut()
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        receipt_object.insert(
            "signature".to_owned(),
            Value::Array(signature.iter().map(|byte| Value::from(*byte)).collect()),
        );
        let receipt = EvidenceQualificationReceiptV1::from_json_value(&receipt_value)?;
        if receipt.key_id != key_id {
            return Err(EvidenceQualificationRejectV1::DssePayloadBindingMismatch);
        }
        if receipt.verifier_profile_sha256 != pinned.profile_sha256() {
            return Err(EvidenceQualificationRejectV1::TrustProfileDigestMismatch);
        }

        let subjects = statement
            .get("subject")
            .and_then(Value::as_array)
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if subjects.len() != 1 {
            return Err(EvidenceQualificationRejectV1::DssePayloadBindingMismatch);
        }
        let subject = subjects[0]
            .as_object()
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if subject.len() != 2 {
            return Err(EvidenceQualificationRejectV1::DsseInvalidEnvelope);
        }
        let expected_subject_name = format!(
            "urn:luminous:mace:candidate-set:v1:{}",
            receipt.candidate_set_sha256
        );
        if subject.get("name").and_then(Value::as_str) != Some(expected_subject_name.as_str()) {
            return Err(EvidenceQualificationRejectV1::DssePayloadBindingMismatch);
        }
        let digest = subject
            .get("digest")
            .and_then(Value::as_object)
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
        if digest.len() != 1
            || digest.get("sha256").and_then(Value::as_str)
                != Some(receipt.candidate_set_sha256.as_str())
        {
            return Err(EvidenceQualificationRejectV1::DssePayloadBindingMismatch);
        }

        // The predicate must be exactly the unsigned typed MACE receipt. This
        // catches unexpected fields and prevents alternate receipt semantics.
        let mut expected_predicate = receipt.to_json_value();
        expected_predicate
            .as_object_mut()
            .ok_or(EvidenceQualificationRejectV1::DsseInvalidEnvelope)?
            .remove("signature");
        if expected_predicate != *predicate {
            return Err(EvidenceQualificationRejectV1::DssePayloadBindingMismatch);
        }

        let pae = dsse_pae_v1(payload_type, &payload);
        Ok(Self {
            pinned,
            receipt,
            key_id,
            signature,
            pae,
        })
    }

    pub fn receipt(&self) -> &EvidenceQualificationReceiptV1 {
        &self.receipt
    }

    /// Authenticate the exact in-toto statement against the pinned key profile.
    /// keyid is only a key-selection hint; the signed predicate independently
    /// binds the same key ID and the trust profile checks its authority state.
    pub fn verify_at(
        &self,
        issued_at_unix: u64,
        now_unix: u64,
    ) -> Result<(), EvidenceQualificationRejectV1> {
        self.pinned.verify_message(
            &self.receipt.verifier_profile_sha256,
            &self.key_id,
            &self.pae,
            &self.signature,
            issued_at_unix,
            now_unix,
        )
    }
}

impl EvidenceQualificationSignatureVerifierV1 for DsseQualificationSignatureVerifierV1<'_> {
    fn verify_signature(
        &self,
        verifier_profile_sha256: &str,
        key_id: &str,
        payload_sha256: &str,
        signature: &[u8],
        issued_at_unix: u64,
        now_unix: u64,
    ) -> Result<(), EvidenceQualificationRejectV1> {
        if verifier_profile_sha256 != self.receipt.verifier_profile_sha256
            || key_id != self.key_id
            || payload_sha256 != self.receipt.signing_payload_sha256()
            || signature != self.signature.as_slice()
        {
            return Err(EvidenceQualificationRejectV1::DssePayloadBindingMismatch);
        }
        self.verify_at(issued_at_unix, now_unix)
    }
}

/// Parse and verify a DSSE in-toto qualification envelope against the exact
/// DecisionNeed binding. For later consumption, reconstruct a fresh
/// DsseQualificationSignatureVerifierV1 from the same envelope and current
/// pinned profile, then pass it to admissible_evidence_ids_for_use or the
/// qualified-candidate projection function.
pub fn verify_in_toto_qualification_envelope_v1(
    envelope_json: &str,
    binding: &EvidenceQualificationBindingV1<'_>,
    pinned: &PinnedEd25519QualificationVerifierV1,
) -> Result<VerifiedEvidenceQualificationV1, EvidenceQualificationRejectV1> {
    let verifier =
        DsseQualificationSignatureVerifierV1::from_envelope_json(envelope_json, pinned)?;
    VerifiedEvidenceQualificationV1::verify_and_bind(verifier.receipt(), binding, &verifier)
}

fn decode_canonical_base64(value: &str) -> Result<Vec<u8>, EvidenceQualificationRejectV1> {
    let decoded = STANDARD
        .decode(value)
        .map_err(|_| EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
    if STANDARD.encode(&decoded) != value {
        return Err(EvidenceQualificationRejectV1::DsseInvalidEnvelope);
    }
    Ok(decoded)
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, EvidenceQualificationRejectV1> {
    if bytes.len() > MAX_DSSE_PAYLOAD_BYTES_V1 {
        return Err(EvidenceQualificationRejectV1::DssePayloadTooLarge);
    }
    let parsed: StrictJsonValue = serde_json::from_slice(bytes)
        .map_err(|_| EvidenceQualificationRejectV1::DsseInvalidEnvelope)?;
    Ok(parsed.into_value())
}

/// serde_json::Value normally resolves duplicate object keys by keeping the
/// last value. Attestation payloads must reject duplicates instead, since
/// different implementations may interpret them differently.
enum StrictJsonValue {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<StrictJsonValue>),
    Object(BTreeMap<String, StrictJsonValue>),
}

impl StrictJsonValue {
    fn into_value(self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(value),
            Self::Number(value) => Value::Number(value),
            Self::String(value) => Value::String(value),
            Self::Array(values) => {
                Value::Array(values.into_iter().map(Self::into_value).collect())
            }
            Self::Object(values) => {
                let mut object = Map::new();
                for (key, value) in values {
                    object.insert(key, value.into_value());
                }
                Value::Object(object)
            }
        }
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value with no duplicate object keys")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictJsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        Number::from_f64(value)
            .map(StrictJsonValue::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictJsonValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictJsonValue::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJsonValue>()? {
            values.push(value);
        }
        Ok(StrictJsonValue::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, StrictJsonValue>()? {
            if values.insert(key, value).is_some() {
                return Err(A::Error::custom("duplicate JSON object key"));
            }
        }
        Ok(StrictJsonValue::Object(values))
    }
}

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    use crate::evidence_qualification_gate::{
        EvidenceDispositionV1,
        EvidenceQualificationEntryV1,
    };
    use crate::qualification_trust::{
        QualificationKeyStatusV1,
        QualificationTrustKeyV1,
        QualificationTrustProfileV1,
        qualification_key_id_v1,
    };
    use sha2::{Digest, Sha256};

    fn candidate_ids_digest(ids: &[String]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:candidate-evidence-ids:v1\\0");
        hasher.update((ids.len() as u64).to_be_bytes());
        for id in ids {
            hasher.update((id.len() as u64).to_be_bytes());
            hasher.update(id.as_bytes());
        }
        let bytes = hasher.finalize();
        let mut output = String::with_capacity(64);
        for byte in bytes {
            output.push_str(&format!("{byte:02x}"));
        }
        output
    }

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[73; 32])
    }

    fn receipt(key_id: &str, profile_sha256: &str) -> EvidenceQualificationReceiptV1 {
        EvidenceQualificationReceiptV1 {
            run_id: "run-dsse".to_owned(),
            query_invocation_id: "invocation-dsse".to_owned(),
            domain_id: "web".to_owned(),
            subject_scope_sha256: "11".repeat(32),
            environment_revision_sha256: "22".repeat(32),
            tool_profile_sha256: "33".repeat(32),
            query_sha256: "44".repeat(32),
            candidate_set_sha256: "55".repeat(32),
            candidate_evidence_ids_sha256: candidate_ids_digest(&vec!["evidence-a".to_owned()]),
            decision_need_sha256: "77".repeat(32),
            purpose_sha256: "88".repeat(32),
            qualification_policy_sha256: "99".repeat(32),
            lifecycle_snapshot_sha256: "aa".repeat(32),
            verifier_profile_sha256: profile_sha256.to_owned(),
            key_id: key_id.to_owned(),
            issued_at_unix: 100,
            expires_at_unix: 300,
            entries: vec![EvidenceQualificationEntryV1 {
                evidence_id: "evidence-a".to_owned(),
                disposition: EvidenceDispositionV1::Admissible,
                basis_receipt_sha256: "bb".repeat(32),
            }],
            signature: Vec::new(),
        }
    }

    fn binding<'a>(
        receipt: &'a EvidenceQualificationReceiptV1,
        ids: &'a [String],
    ) -> EvidenceQualificationBindingV1<'a> {
        EvidenceQualificationBindingV1 {
            run_id: &receipt.run_id,
            query_invocation_id: &receipt.query_invocation_id,
            domain_id: &receipt.domain_id,
            subject_scope_sha256: &receipt.subject_scope_sha256,
            environment_revision_sha256: &receipt.environment_revision_sha256,
            tool_profile_sha256: &receipt.tool_profile_sha256,
            query_sha256: &receipt.query_sha256,
            candidate_set_sha256: &receipt.candidate_set_sha256,
            candidate_evidence_ids: ids,
            decision_need_sha256: &receipt.decision_need_sha256,
            purpose_sha256: &receipt.purpose_sha256,
            qualification_policy_sha256: &receipt.qualification_policy_sha256,
            lifecycle_snapshot_sha256: &receipt.lifecycle_snapshot_sha256,
            verifier_profile_sha256: &receipt.verifier_profile_sha256,
            now_unix: 150,
        }
    }

    fn pinned(
        signing_key: &SigningKey,
        status: QualificationKeyStatusV1,
        profile_version: u64,
    ) -> PinnedEd25519QualificationVerifierV1 {
        let public_key = signing_key.verifying_key().to_bytes();
        let key = QualificationTrustKeyV1 {
            key_id: qualification_key_id_v1(&public_key),
            public_key,
            status,
            not_before_unix: 1,
            not_after_unix: 500,
        };
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-independent-qualifier".to_owned(),
            profile_version,
            keys: vec![key],
        };
        let digest = profile.profile_sha256().unwrap();
        PinnedEd25519QualificationVerifierV1::from_pinned_profile(&profile, &digest, profile_version)
            .unwrap()
    }

    fn envelope(
        receipt: &EvidenceQualificationReceiptV1,
        signing_key: &SigningKey,
    ) -> String {
        let payload = in_toto_qualification_payload_v1(receipt).unwrap();
        let pae = dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, &payload);
        let signature = signing_key.sign(&pae).to_bytes();
        encode_dsse_envelope_v1(
            &payload,
            &receipt.key_id,
            &signature,
        ).unwrap()
    }

    #[test]
    fn pae_uses_byte_lengths_and_binds_payload_type() {
        assert_eq!(
            dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, b"{}"),
            b"DSSEv1 28 application/vnd.in-toto+json 2 {}".to_vec()
        );
        assert_ne!(
            dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, b"{}"),
            dsse_pae_v1("application/json", b"{}")
        );
    }

    #[test]
    fn in_toto_dsse_receipt_verifies_and_revalidates_at_use() {
        let key = signing_key();
        let public_key = key.verifying_key().to_bytes();
        let key_id = qualification_key_id_v1(&public_key);
        let profile_key = QualificationTrustKeyV1 {
            key_id: key_id.clone(),
            public_key,
            status: QualificationKeyStatusV1::Active,
            not_before_unix: 1,
            not_after_unix: 500,
        };
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-independent-qualifier".to_owned(),
            profile_version: 1,
            keys: vec![profile_key],
        };
        let profile_digest = profile.profile_sha256().unwrap();
        let pinned = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile,
            &profile_digest,
            1,
        ).unwrap();

        let receipt = receipt(&key_id, &profile_digest);
        let ids = vec!["evidence-a".to_owned()];
        let envelope_json = envelope(&receipt, &key);
        let binding = binding(&receipt, &ids);
        let verified = verify_in_toto_qualification_envelope_v1(
            &envelope_json,
            &binding,
            &pinned,
        ).expect("DSSE/in-toto signature and exact binding verify");

        let verifier = DsseQualificationSignatureVerifierV1::from_envelope_json(
            &envelope_json,
            &pinned,
        ).unwrap();
        assert_eq!(
            verified.admissible_evidence_ids_for_use(&binding, &verifier).unwrap(),
            &["evidence-a".to_owned()]
        );

        let wrong_purpose_digest = "ff".repeat(32);
        let mut wrong_purpose = binding;
        wrong_purpose.purpose_sha256 = &wrong_purpose_digest;
        assert_eq!(
            verified.admissible_evidence_ids_for_use(&wrong_purpose, &verifier),
            Err(EvidenceQualificationRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn envelope_rejects_wrong_payload_type_multiple_signatures_and_duplicate_keys() {
        let key = signing_key();
        let public_key = key.verifying_key().to_bytes();
        let key_id = qualification_key_id_v1(&public_key);
        let pinned = pinned(&key, QualificationKeyStatusV1::Active, 1);
        let receipt = receipt(&key_id, pinned.profile_sha256());
        let payload = in_toto_qualification_payload_v1(&receipt).unwrap();
        let pae = dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, &payload);
        let signature = key.sign(&pae).to_bytes();
        let valid = envelope(&receipt, &key);

        let mut wrong_type: Value = serde_json::from_str(&valid).unwrap();
        wrong_type["payloadType"] = Value::String("application/json".to_owned());
        let wrong_type_json = serde_json::to_string(&wrong_type).unwrap();
        assert_eq!(
            DsseQualificationSignatureVerifierV1::from_envelope_json(&wrong_type_json, &pinned)
                .err(),
            Some(EvidenceQualificationRejectV1::DssePayloadTypeMismatch)
        );

        let mut multiple: Value = serde_json::from_str(&valid).unwrap();
        let first_signature = multiple["signatures"][0].clone();
        multiple["signatures"] = Value::Array(vec![first_signature.clone(), first_signature]);
        let multiple_json = serde_json::to_string(&multiple).unwrap();
        assert_eq!(
            DsseQualificationSignatureVerifierV1::from_envelope_json(&multiple_json, &pinned)
                .err(),
            Some(EvidenceQualificationRejectV1::DsseSignatureCountUnsupported)
        );

        let duplicate = format!(
            r#"{{"payloadType":"{}","payloadType":"{}","payload":"{}","signatures":[{{"keyid":"{}","sig":"{}"}}]}}"#,
            DSSE_PAYLOAD_TYPE_V1,
            DSSE_PAYLOAD_TYPE_V1,
            STANDARD.encode(&payload),
            key_id,
            STANDARD.encode(signature)
        );
        assert_eq!(
            DsseQualificationSignatureVerifierV1::from_envelope_json(&duplicate, &pinned)
                .err(),
            Some(EvidenceQualificationRejectV1::DsseInvalidEnvelope)
        );
    }

    #[test]
    fn dsse_signature_cannot_be_replayed_after_key_revocation_or_profile_rotation() {
        let key = signing_key();
        let public_key = key.verifying_key().to_bytes();
        let key_id = qualification_key_id_v1(&public_key);
        let active_profile = pinned(&key, QualificationKeyStatusV1::Active, 1);
        let revoked_profile = pinned(&key, QualificationKeyStatusV1::Revoked, 2);
        let receipt = receipt(&key_id, active_profile.profile_sha256());
        let envelope_json = envelope(&receipt, &key);
        let ids = vec!["evidence-a".to_owned()];
        let binding = binding(&receipt, &ids);
        let verified = verify_in_toto_qualification_envelope_v1(
            &envelope_json,
            &binding,
            &active_profile,
        ).unwrap();

        let current_verifier =
            DsseQualificationSignatureVerifierV1::from_envelope_json(
                &envelope_json,
                &revoked_profile,
            );
        assert_eq!(
            current_verifier.err(),
            Some(EvidenceQualificationRejectV1::TrustProfileDigestMismatch)
        );

        // If a caller incorrectly keeps the old pinned profile, verification
        // succeeds by design; freshness of the out-of-band profile pin is an
        // external relying-party obligation documented by the API.
        let stale_verifier = DsseQualificationSignatureVerifierV1::from_envelope_json(
            &envelope_json,
            &active_profile,
        ).unwrap();
        assert!(verified.admissible_evidence_ids_for_use(&binding, &stale_verifier).is_ok());
    }
}
