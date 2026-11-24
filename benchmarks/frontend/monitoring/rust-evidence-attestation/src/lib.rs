//! Strict source-receipt verification for the SFEB evidence contract.
//!
//! DSSE authenticates exact payload bytes and payload type. A separate trust
//! registry authorizes keys for issuer, tenant and source. Strict JSON/JCS
//! rejects ambiguous input before a content digest is computed.
//!
//! This crate does not provide key provisioning, durable revocation, HSM
//! custody, source-truth determination, or a persistent operation ledger.

use std::collections::{BTreeSet, HashSet};

use base64::{engine::general_purpose::{STANDARD as BASE64, STANDARD_NO_PAD as BASE64_NO_PAD, URL_SAFE as BASE64_URL, URL_SAFE_NO_PAD as BASE64_URL_NO_PAD}, Engine as _};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{
    de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor},
    Deserialize as SerdeDeserialize, Serialize,
};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const SOURCE_RECEIPT_PAYLOAD_TYPE: &str =
    "application/vnd.luminous.sfeb.source-receipt.v1+json";
pub const SOURCE_RECEIPT_SCHEMA: &str = "sfeb.source-receipt.v1";
pub const MAX_ENVELOPE_BYTES: usize = 1_048_576;
pub const MAX_PAYLOAD_BYTES: usize = 262_144;
pub const MAX_SIGNATURES: usize = 8;

#[derive(Debug, Error)]
pub enum AttestationError {
    #[error("strict JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("JCS canonicalization failed: {0}")]
    Canonicalization(String),
    #[error("invalid DSSE envelope: {0}")]
    InvalidEnvelope(&'static str),
    #[error("invalid base64 in {0}")]
    InvalidBase64(&'static str),
    #[error("unsupported DSSE payload type")]
    PayloadType,
    #[error("invalid signature threshold")]
    Threshold,
    #[error("signature did not match a trusted Ed25519 key")]
    InvalidSignature,
    #[error("signature matched multiple trust identities")]
    AmbiguousKey,
    #[error("signing key outside validity interval")]
    KeyOutsideValidity,
    #[error("signing key revoked")]
    KeyRevoked,
    #[error("signing key not authorized for tenant")]
    TenantScopeDenied,
    #[error("signing key not authorized for source")]
    SourceScopeDenied,
    #[error("duplicate key id")]
    DuplicateKeyId,
    #[error("duplicate public key under multiple trust identities")]
    DuplicatePublicKey,
    #[error("signature threshold not met by distinct trusted keys")]
    InsufficientSigners,
    #[error("invalid source receipt: {0}")]
    InvalidReceipt(&'static str),
    #[error("receipt tenant differs from requested tenant")]
    ReceiptTenantMismatch,
    #[error("receipt source differs from requested source")]
    ReceiptSourceMismatch,
    #[error("receipt issuer differs from signing key issuer")]
    IssuerMismatch,
    #[error("receipt is future-dated")]
    FutureReceipt,
    #[error("receipt expired")]
    ExpiredReceipt,
}

/// A serde value wrapper that rejects duplicate map keys before they can be
/// collapsed by serde_json::Value's normal last-value-wins map handling.
struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = StrictValue;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an I-JSON compatible value with no duplicate object names")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let as_f64 = value as f64;
        if (as_f64 as i128) != (value as i128) {
            return Err(E::custom("integer is not exactly representable as IEEE-754 binary64"));
        }
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let as_f64 = value as f64;
        if (as_f64 as u128) != (value as u128) {
            return Err(E::custom("integer is not exactly representable as IEEE-754 binary64"));
        }
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = Number::from_f64(value)
            .ok_or_else(|| E::custom("non-finite number is not valid I-JSON"))?;
        Ok(StrictValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<StrictValue>()? {
            values.push(value.0);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON property name: {key}")));
            }
            let value = map.next_value::<StrictValue>()?;
            values.insert(key, value.0);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}

/// Parse untrusted JSON while rejecting duplicate keys at every object depth.
pub fn parse_strict_json(bytes: &[u8]) -> Result<Value, AttestationError> {
    Ok(serde_json::from_slice::<StrictValue>(bytes)?.0)
}

/// Produce RFC 8785 canonical bytes after strict duplicate-key parsing.
pub fn canonicalize_jcs(bytes: &[u8]) -> Result<Vec<u8>, AttestationError> {
    let value = parse_strict_json(bytes)?;
    serde_json_canonicalizer::to_vec(&value)
        .map_err(|error| AttestationError::Canonicalization(error.to_string()))
}

/// SHA-256 over RFC 8785 canonical bytes.
pub fn sha256_jcs(bytes: &[u8]) -> Result<[u8; 32], AttestationError> {
    let canonical = canonicalize_jcs(bytes)?;
    Ok(Sha256::digest(canonical).into())
}

/// DSSE v1 pre-authentication encoding. Lengths are UTF-8 byte lengths.
pub fn pae(payload_type: &str, body: &[u8]) -> Vec<u8> {
    let payload_type = payload_type.as_bytes();
    let mut out = Vec::with_capacity(32 + payload_type.len() + body.len());
    out.extend_from_slice(b"DSSEv1 ");
    out.extend_from_slice(payload_type.len().to_string().as_bytes());
    out.push(b' ');
    out.extend_from_slice(payload_type);
    out.push(b' ');
    out.extend_from_slice(body.len().to_string().as_bytes());
    out.push(b' ');
    out.extend_from_slice(body);
    out
}

#[derive(Debug, Clone, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct DsseEnvelope {
    #[serde(rename = "payloadType")]
    pub payload_type: String,
    pub payload: String,
    pub signatures: Vec<DsseSignature>,
}

#[derive(Debug, Clone, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct DsseSignature {
    /// Optional, unauthenticated lookup hint. Never used as an authorization decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyid: Option<String>,
    pub sig: String,
}

#[derive(Debug, Clone, SerdeDeserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceReceiptClaims {
    pub schema: String,
    pub tenant_id: String,
    pub source: String,
    pub event_id: String,
    pub sequence: u64,
    pub issued_at: i64,
    pub occurred_at: i64,
    pub expires_at: i64,
    pub resource_id: String,
    pub action_type: String,
    pub issuer_id: String,
}

#[derive(Debug, Clone)]
pub struct TrustedKey {
    pub key_id: String,
    pub verifying_key: VerifyingKey,
    pub issuer_id: String,
    pub tenants: BTreeSet<String>,
    pub sources: BTreeSet<String>,
    pub not_before: i64,
    pub not_after: i64,
    pub revoked_at: Option<i64>,
}

#[derive(Debug, Default, Clone)]
pub struct TrustStore {
    keys: Vec<TrustedKey>,
}

impl TrustStore {
    pub fn register(&mut self, key: TrustedKey) -> Result<(), AttestationError> {
        if self.keys.iter().any(|existing| existing.key_id == key.key_id) {
            return Err(AttestationError::DuplicateKeyId);
        }
        if self.keys.iter().any(|existing| {
            existing.verifying_key.to_bytes() == key.verifying_key.to_bytes()
        }) {
            return Err(AttestationError::DuplicatePublicKey);
        }
        if key.key_id.is_empty()
            || key.issuer_id.is_empty()
            || key.tenants.is_empty()
            || key.sources.is_empty()
            || key.not_after <= key.not_before
        {
            return Err(AttestationError::InvalidEnvelope("invalid trust record"));
        }
        self.keys.push(key);
        Ok(())
    }

    pub fn revoke(&mut self, key_id: &str, at: i64) -> Result<(), AttestationError> {
        let key = self
            .keys
            .iter_mut()
            .find(|key| key.key_id == key_id)
            .ok_or(AttestationError::InvalidEnvelope("unknown trust key"))?;
        key.revoked_at = Some(key.revoked_at.map_or(at, |prior| prior.min(at)));
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VerifiedReceipt {
    pub claims: SourceReceiptClaims,
    pub payload_sha256: [u8; 32],
    pub signer_key_ids: Vec<String>,
}

/// Verify a DSSE receipt against every configured trust key.
///
/// keyid is intentionally ignored for trust selection: DSSE defines it as an
/// unauthenticated hint, so a missing or spoofed value cannot exclude a key.
/// Distinct trusted public keys, not repeated signature entries, satisfy the
/// requested threshold.
pub fn verify_source_receipt(
    envelope_bytes: &[u8],
    trust_store: &TrustStore,
    expected_tenant_id: &str,
    expected_source: &str,
    now: i64,
    threshold: usize,
) -> Result<VerifiedReceipt, AttestationError> {
    if envelope_bytes.len() > MAX_ENVELOPE_BYTES {
        return Err(AttestationError::InvalidEnvelope("envelope too large"));
    }
    let strict_envelope = parse_strict_json(envelope_bytes)?;
    let envelope: DsseEnvelope = serde_json::from_value(strict_envelope)?;
    if envelope.payload_type != SOURCE_RECEIPT_PAYLOAD_TYPE {
        return Err(AttestationError::PayloadType);
    }
    if envelope.signatures.is_empty() || envelope.signatures.len() > MAX_SIGNATURES {
        return Err(AttestationError::InvalidEnvelope("signature count outside profile bounds"));
    }
    if threshold == 0 || threshold > envelope.signatures.len() {
        return Err(AttestationError::Threshold);
    }

    let payload = decode_canonical_base64(&envelope.payload, "payload")?;
    if payload.len() > MAX_PAYLOAD_BYTES {
        return Err(AttestationError::InvalidEnvelope("payload too large"));
    }
    let preimage = pae(&envelope.payload_type, &payload);
    let mut verified = BTreeSet::new();
    let mut verified_records = Vec::new();

    for entry in &envelope.signatures {
        // Accessing keyid is deliberately unnecessary: it is not authenticated.
        let signature_bytes = decode_canonical_base64(&entry.sig, "signature")?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|_| AttestationError::InvalidBase64("Ed25519 signature"))?;
        let matching: Vec<&TrustedKey> = trust_store
            .keys
            .iter()
            .filter(|record| record.verifying_key.verify(&preimage, &signature).is_ok())
            .collect();

        if matching.is_empty() {
            return Err(AttestationError::InvalidSignature);
        }
        if matching.len() != 1 {
            return Err(AttestationError::AmbiguousKey);
        }
        let record = matching[0];
        if verified.contains(&record.key_id) {
            continue;
        }
        if now < record.not_before || now >= record.not_after {
            return Err(AttestationError::KeyOutsideValidity);
        }
        if record.revoked_at.is_some_and(|revoked_at| now >= revoked_at) {
            return Err(AttestationError::KeyRevoked);
        }
        if !record.tenants.contains(expected_tenant_id) {
            return Err(AttestationError::TenantScopeDenied);
        }
        if !record.sources.contains(expected_source) {
            return Err(AttestationError::SourceScopeDenied);
        }
        verified.insert(record.key_id.clone());
        verified_records.push(record);
    }

    if verified.len() < threshold {
        return Err(AttestationError::InsufficientSigners);
    }

    // Parse only after verifying the signature over the exact payload bytes.
    // The strict parser rejects duplicate JSON names and ambiguous encodings.
    let claims_value = parse_strict_json(&payload)?;
    let claims: SourceReceiptClaims = serde_json::from_value(claims_value)?;
    if claims.schema != SOURCE_RECEIPT_SCHEMA {
        return Err(AttestationError::InvalidReceipt("unknown receipt schema"));
    }
    if claims.tenant_id != expected_tenant_id {
        return Err(AttestationError::ReceiptTenantMismatch);
    }
    if claims.source != expected_source {
        return Err(AttestationError::ReceiptSourceMismatch);
    }
    if claims.event_id.is_empty()
        || claims.resource_id.is_empty()
        || claims.action_type.is_empty()
        || claims.issuer_id.is_empty()
        || claims.sequence == 0
        || claims.expires_at <= claims.issued_at
    {
        return Err(AttestationError::InvalidReceipt("required field or validity interval invalid"));
    }
    if claims.issued_at > now || claims.occurred_at > now {
        return Err(AttestationError::FutureReceipt);
    }
    if claims.expires_at <= now {
        return Err(AttestationError::ExpiredReceipt);
    }
    if verified_records.iter().any(|key| key.issuer_id != claims.issuer_id) {
        return Err(AttestationError::IssuerMismatch);
    }

    let payload_sha256 = Sha256::digest(&payload).into();
    Ok(VerifiedReceipt {
        claims,
        payload_sha256,
        signer_key_ids: verified.into_iter().collect(),
    })
}

fn decode_canonical_base64(value: &str, field: &'static str) -> Result<Vec<u8>, AttestationError> {
    // DSSE requires verifiers to accept standard and URL-safe base64. Each
    // accepted spelling must be canonical for its selected alphabet/padding.
    for engine in [&BASE64, &BASE64_NO_PAD, &BASE64_URL, &BASE64_URL_NO_PAD] {
        if let Ok(bytes) = engine.decode(value) {
            if engine.encode(&bytes) == value {
                return Ok(bytes);
            }
        }
    }
    Err(AttestationError::InvalidBase64(field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use serde_json::json;

    const NOW: i64 = 1_800_000_000;

    fn signing_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn trusted(key_id: &str, signer: &SigningKey) -> TrustedKey {
        TrustedKey {
            key_id: key_id.into(),
            verifying_key: signer.verifying_key(),
            issuer_id: "capacity-prod".into(),
            tenants: ["tenant-A".into()].into_iter().collect(),
            sources: ["urn:example:capacity-service".into()].into_iter().collect(),
            not_before: NOW - 1_000,
            not_after: NOW + 1_000,
            revoked_at: None,
        }
    }

    fn claims() -> Value {
        json!({
            "schema": SOURCE_RECEIPT_SCHEMA,
            "tenantId": "tenant-A",
            "source": "urn:example:capacity-service",
            "eventId": "evt-42",
            "sequence": 42,
            "issuedAt": NOW - 10,
            "occurredAt": NOW - 30,
            "expiresAt": NOW + 300,
            "resourceId": "slot-17",
            "actionType": "reserve",
            "issuerId": "capacity-prod"
        })
    }

    fn envelope(payload: &[u8], signers: &[(&SigningKey, Option<&str>)]) -> Vec<u8> {
        let preimage = pae(SOURCE_RECEIPT_PAYLOAD_TYPE, payload);
        let signatures = signers
            .iter()
            .map(|(signer, keyid)| DsseSignature {
                keyid: keyid.map(str::to_owned),
                sig: BASE64.encode(signer.sign(&preimage).to_bytes()),
            })
            .collect();
        serde_json::to_vec(&DsseEnvelope {
            payload_type: SOURCE_RECEIPT_PAYLOAD_TYPE.into(),
            payload: BASE64.encode(payload),
            signatures,
        })
        .unwrap()
    }

    fn fixture() -> (SigningKey, TrustStore, Vec<u8>) {
        let signer = signing_key(7);
        let mut store = TrustStore::default();
        store.register(trusted("key-prod-1", &signer)).unwrap();
        let payload = serde_json::to_vec(&claims()).unwrap();
        let env = envelope(&payload, &[(&signer, Some("key-prod-1"))]);
        (signer, store, env)
    }

    #[test]
    fn rfc8785_appendix_example_is_canonicalized() {
        let input = br#"{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001],"string":"\u20ac$\u000f\u000aA'\u0042\u0022\u005c\\\"\/","literals":[null,true,false]}"#;
        let expected = br#"{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}"#;
        assert_eq!(canonicalize_jcs(input).unwrap(), expected);
    }

    #[test]
    fn duplicate_keys_are_rejected_including_escaped_equivalents() {
        assert!(parse_strict_json(br#"{"a":1,"a":2}"#).is_err());
        assert!(parse_strict_json(br#"{"a":1,"\u0061":2}"#).is_err());
        assert!(parse_strict_json(br#"{"nested":{"x":1,"x":2}}"#).is_err());
    }

    #[test]
    fn utf16_property_order_is_used_for_non_bmp_keys() {
        let canonical = canonicalize_jcs("{\"\":2,\"𐀀\":1}".as_bytes()).unwrap();
        assert_eq!(canonical, "{\"𐀀\":1,\"\":2}".as_bytes());
    }

    #[test]
    fn canonicalization_is_key_order_invariant_but_does_not_normalize_unicode() {
        let a = canonicalize_jcs(br#"{"b":2,"a":{"d":4,"c":3}}"#).unwrap();
        let b = canonicalize_jcs(br#"{"a":{"c":3,"d":4},"b":2}"#).unwrap();
        assert_eq!(a, b);
        let precomposed = canonicalize_jcs("{\"x\":\"é\"}".as_bytes()).unwrap();
        let decomposed = canonicalize_jcs("{\"x\":\"é\"}".as_bytes()).unwrap();
        assert_ne!(precomposed, decomposed);
    }

    #[test]
    fn pae_uses_utf8_byte_lengths() {
        let encoded = pae("application/example+json;profile=✓", "payload-✓".as_bytes());
        let type_len = "application/example+json;profile=✓".len();
        let body_len = "payload-✓".len();
        assert!(encoded.starts_with(format!("DSSEv1 {type_len} application/example+json;profile=✓ {body_len} ").as_bytes()));
    }

    #[test]
    fn valid_signature_verifies_and_unknown_keyid_does_not_block_it() {
        let (_signer, store, env) = fixture();
        let mut envelope_value: Value = serde_json::from_slice(&env).unwrap();
        envelope_value["signatures"][0]["keyid"] = json!("untrusted-hint");
        let env_with_unknown_hint = serde_json::to_vec(&envelope_value).unwrap();
        let verified = verify_source_receipt(
            &env_with_unknown_hint, &store, "tenant-A", "urn:example:capacity-service", NOW, 1,
        ).unwrap();
        assert_eq!(verified.claims.event_id, "evt-42");
        assert_eq!(verified.signer_key_ids, vec!["key-prod-1"]);
    }

    #[test]
    fn standard_and_url_safe_base64_encodings_are_accepted() {
        let (signer, store, _env) = fixture();
        let payload = serde_json::to_vec(&claims()).unwrap();
        let preimage = pae(SOURCE_RECEIPT_PAYLOAD_TYPE, &payload);
        let signature = signer.sign(&preimage);
        let env = DsseEnvelope {
            payload_type: SOURCE_RECEIPT_PAYLOAD_TYPE.into(),
            payload: BASE64_URL_NO_PAD.encode(&payload),
            signatures: vec![DsseSignature {
                keyid: None,
                sig: BASE64_URL_NO_PAD.encode(signature.to_bytes()),
            }],
        };
        let bytes = serde_json::to_vec(&env).unwrap();
        assert!(verify_source_receipt(
            &bytes, &store, "tenant-A", "urn:example:capacity-service", NOW, 1
        ).is_ok());
    }

    #[test]
    fn missing_keyid_is_allowed() {
        let (signer, store, _env) = fixture();
        let payload = serde_json::to_vec(&claims()).unwrap();
        let env = envelope(&payload, &[(&signer, None)]);
        assert!(verify_source_receipt(&env, &store, "tenant-A", "urn:example:capacity-service", NOW, 1).is_ok());
    }

    #[test]
    fn attacker_cannot_relabel_signature_with_trusted_keyid() {
        let (_signer, store, _env) = fixture();
        let attacker = signing_key(9);
        let payload = serde_json::to_vec(&claims()).unwrap();
        let env = envelope(&payload, &[(&attacker, Some("key-prod-1"))]);
        assert!(matches!(
            verify_source_receipt(&env, &store, "tenant-A", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::InvalidSignature)
        ));
    }

    #[test]
    fn exact_payload_bytes_are_signed_not_canonicalized_implicitly() {
        let (signer, store, env) = fixture();
        let mut envelope_value: Value = serde_json::from_slice(&env).unwrap();
        let reordered = br#"{"tenantId":"tenant-A","schema":"sfeb.source-receipt.v1","source":"urn:example:capacity-service","eventId":"evt-42","sequence":42,"issuedAt":1799999990,"occurredAt":1799999970,"expiresAt":1800000300,"resourceId":"slot-17","actionType":"reserve","issuerId":"capacity-prod"}"#;
        envelope_value["payload"] = json!(BASE64.encode(reordered));
        let changed = serde_json::to_vec(&envelope_value).unwrap();
        let _ = signer;
        assert!(matches!(
            verify_source_receipt(&changed, &store, "tenant-A", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::InvalidSignature)
        ));
    }

    #[test]
    fn duplicated_signature_does_not_satisfy_two_key_threshold() {
        let (signer, store, _env) = fixture();
        let payload = serde_json::to_vec(&claims()).unwrap();
        let env = envelope(&payload, &[(&signer, None), (&signer, Some("alias"))]);
        assert!(matches!(
            verify_source_receipt(&env, &store, "tenant-A", "urn:example:capacity-service", NOW, 2),
            Err(AttestationError::InsufficientSigners)
        ));
    }

    #[test]
    fn two_distinct_keys_satisfy_two_of_two_threshold() {
        let signer_a = signing_key(7);
        let signer_b = signing_key(8);
        let mut store = TrustStore::default();
        store.register(trusted("key-a", &signer_a)).unwrap();
        store.register(trusted("key-b", &signer_b)).unwrap();
        let payload = serde_json::to_vec(&claims()).unwrap();
        let env = envelope(&payload, &[(&signer_a, Some("wrong-hint")), (&signer_b, Some("key-b"))]);
        let verified = verify_source_receipt(&env, &store, "tenant-A", "urn:example:capacity-service", NOW, 2).unwrap();
        assert_eq!(verified.signer_key_ids.len(), 2);
    }

    #[test]
    fn envelope_duplicate_keys_are_rejected_before_deserialization() {
        let (_signer, _store, env) = fixture();
        let value: Value = serde_json::from_slice(&env).unwrap();
        let payload = value["payload"].as_str().unwrap();
        let signatures = serde_json::to_string(&value["signatures"]).unwrap();
        let duplicate = format!(
            "{{\"payloadType\":\"{SOURCE_RECEIPT_PAYLOAD_TYPE}\",\"payload\":\"{payload}\",\"payload\":\"{payload}\",\"signatures\":{signatures}}}"
        );
        assert!(parse_strict_json(duplicate.as_bytes()).is_err());
    }

    #[test]
    fn signed_payload_duplicate_keys_are_rejected_after_signature_check() {
        let (signer, store, _env) = fixture();
        let payload = br#"{"schema":"sfeb.source-receipt.v1","tenantId":"tenant-A","tenantId":"tenant-B"}"#;
        let env = envelope(payload, &[(&signer, None)]);
        assert!(matches!(
            verify_source_receipt(&env, &store, "tenant-A", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::Json(_))
        ));
    }

    #[test]
    fn revoked_or_expired_keys_are_rejected() {
        let (_signer, mut store, env) = fixture();
        store.revoke("key-prod-1", NOW - 1).unwrap();
        assert!(matches!(
            verify_source_receipt(&env, &store, "tenant-A", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::KeyRevoked)
        ));
        let (_signer, mut store, env) = fixture();
        store.keys[0].not_after = NOW;
        assert!(matches!(
            verify_source_receipt(&env, &store, "tenant-A", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::KeyOutsideValidity)
        ));
    }

    #[test]
    fn signing_key_scope_and_issuer_are_enforced() {
        let (_signer, store, env) = fixture();
        assert!(matches!(
            verify_source_receipt(&env, &store, "tenant-B", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::TenantScopeDenied)
        ));
        assert!(matches!(
            verify_source_receipt(&env, &store, "tenant-A", "urn:other:source", NOW, 1),
            Err(AttestationError::SourceScopeDenied)
        ));
    }

    #[test]
    fn receipt_expiry_and_issuer_binding_are_enforced() {
        let (signer, store, _env) = fixture();
        let expired = json!({
            "schema":SOURCE_RECEIPT_SCHEMA,"tenantId":"tenant-A","source":"urn:example:capacity-service",
            "eventId":"expired","sequence":1,"issuedAt":NOW-10,"occurredAt":NOW-10,
            "expiresAt":NOW,"resourceId":"slot-17","actionType":"reserve","issuerId":"capacity-prod"
        });
        let expired_env = envelope(&serde_json::to_vec(&expired).unwrap(), &[(&signer, None)]);
        assert!(matches!(
            verify_source_receipt(&expired_env, &store, "tenant-A", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::ExpiredReceipt)
        ));
        let wrong_issuer = json!({
            "schema":SOURCE_RECEIPT_SCHEMA,"tenantId":"tenant-A","source":"urn:example:capacity-service",
            "eventId":"issuer","sequence":1,"issuedAt":NOW-10,"occurredAt":NOW-10,
            "expiresAt":NOW+100,"resourceId":"slot-17","actionType":"reserve","issuerId":"attacker"
        });
        let wrong_env = envelope(&serde_json::to_vec(&wrong_issuer).unwrap(), &[(&signer, None)]);
        assert!(matches!(
            verify_source_receipt(&wrong_env, &store, "tenant-A", "urn:example:capacity-service", NOW, 1),
            Err(AttestationError::IssuerMismatch)
        ));
    }
}
