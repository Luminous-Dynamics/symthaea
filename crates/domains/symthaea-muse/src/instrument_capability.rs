// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-cell instrument capability decisions.
//!
//! Instrument identity, a synthesis preset, a GM export program, a sample
//! filename, or a decoded WAV is not a playable-capability certificate.
//! This module makes the decision boundary explicit: a capability is supported
//! only by a unique, exact-match evidence cell with a well-formed receipt,
//! source-manifest digest, renderer-build digest and independent-verifier
//! digest. It never inherits support across pitch, velocity, technique,
//! articulation, setup, realization path or renderer build.
//!
//! This is a decision/schema layer. It checks receipt shape and exact identity;
//! it does not itself hash external asset bytes or replace the source inventory,
//! rights registry, render probe or independent qualification runner.

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::instrument_ontology::instrument_descriptors;

/// Exact pitch domain for an instrument capability request.
///
/// `PitchedMidi` is for a scientific pitch represented in MIDI note numbers.
/// `PercussionKey` is a drum-map key, not a pitched-frequency claim.
/// `Unpitched` is for non-pitched gestures such as cymbal crashes or noise
/// instruments; the stable token must identify the requested gesture family.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum CapabilityPitch {
    PitchedMidi(u8),
    PercussionKey(u8),
    Unpitched(String),
}

/// The actual synthesis/asset path being qualified. A sample-backed path
/// requires an immutable asset digest; a procedural path does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityRealization {
    SampleAsset,
    ProceduralSynthesis,
    Hybrid,
}

/// An exact instrument-performance request. Equality is intentionally exact:
/// evidence for one request cannot silently qualify a neighbouring pitch,
/// velocity, technique, articulation, setup or renderer build.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityRequest {
    /// Stable ID from `instrument_ontology::instrument_descriptors`.
    pub instrument_id: String,
    /// Stable setup identity, e.g. "default", "french-bow", or a named reed.
    pub setup_id: String,
    /// Stable technique identity, e.g. "arco", "pizzicato", "flutter_tongue".
    pub technique_id: String,
    /// Stable articulation identity, e.g. "sustain", "staccato", "tremolo".
    pub articulation_id: String,
    pub pitch: CapabilityPitch,
    /// MIDI-style velocity/dynamic value, inclusive range 0..=127.
    pub velocity: u8,
    pub realization: CapabilityRealization,
    /// Stable renderer identity plus exact build digest.
    pub renderer_id: String,
    pub renderer_build_sha256: String,
}

/// Evidence state recorded for one exact request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceOutcome {
    Qualified,
    Unsupported,
    NotEvaluated,
}

/// Provenance for a capability evaluation.
///
/// The SHA-256 strings are commitments to the corresponding immutable
/// artifacts/receipts. This module validates their syntax and the renderer
/// build binding; the upstream verifier remains responsible for hashing the
/// actual bytes and validating the contents those digests identify.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityProof {
    /// Canonical qualification receipt, including the exact request/outcome.
    pub receipt_sha256: String,
    /// Source/model/rights manifest digest for the evidence inputs.
    pub source_manifest_sha256: String,
    /// Independent verifier implementation/build digest. Must match the
    /// build pinned for `verifier_key_id` in the application trust policy.
    pub independent_verifier_sha256: String,
    /// Must equal the renderer build requested by the capability cell.
    pub renderer_build_sha256: String,
    /// Required for sample-backed or hybrid realization; optional for a
    /// purely procedural synthesis model.
    pub asset_sha256: Option<String>,
    /// Identifier of the trusted verifier key that signed this exact claim.
    pub verifier_key_id: String,
    /// Ed25519 signature over `capability_attestation_message`, hex-encoded.
    pub attestation_signature_hex: String,
}

/// One verifier public key pinned by application configuration. The key is
/// bound to one verifier build digest so a signature cannot claim evidence
/// produced by a different, unqualified executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedCapabilityVerifier {
    pub key_id: String,
    pub public_key: [u8; 32],
    pub verifier_build_sha256: String,
}

/// Trust roots for capability qualification. Construct this from keys pinned
/// by the application/release policy, never from the capability evidence
/// bundle being checked. An empty store cannot promote any evidence to
/// Supported or Unsupported.
#[derive(Debug, Clone, Default)]
pub struct CapabilityTrustStore {
    verifiers: BTreeMap<String, TrustedCapabilityVerifier>,
}

impl CapabilityTrustStore {
    pub fn new(
        verifiers: impl IntoIterator<Item = TrustedCapabilityVerifier>,
    ) -> Result<Self, String> {
        let mut store = Self::default();
        for verifier in verifiers {
            if !nonempty(&verifier.key_id) {
                return Err("trusted verifier key_id must be non-empty".into());
            }
            if !valid_sha256(&verifier.verifier_build_sha256) {
                return Err(format!(
                    "trusted verifier {} has an invalid build SHA-256",
                    verifier.key_id
                ));
            }
            VerifyingKey::from_bytes(&verifier.public_key)
                .map_err(|_| format!("trusted verifier {} has an invalid Ed25519 public key", verifier.key_id))?;
            if store.verifiers.insert(verifier.key_id.clone(), verifier).is_some() {
                return Err("duplicate trusted verifier key_id".into());
            }
        }
        Ok(store)
    }

    fn verifier(&self, key_id: &str) -> Option<&TrustedCapabilityVerifier> {
        self.verifiers.get(key_id)
    }
}

/// One immutable cell in the instrument × setup × technique × articulation ×
/// pitch × velocity × realization × renderer capability matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceCell {
    pub request: CapabilityRequest,
    pub outcome: CapabilityEvidenceOutcome,
    /// Required for Unsupported and recommended for NotEvaluated.
    pub reason: Option<String>,
    /// Qualified and Unsupported outcomes both require auditable proof.
    pub proof: Option<CapabilityProof>,
}

#[derive(Serialize)]
struct CapabilityAttestationPayload<'a> {
    domain: &'static str,
    schema_version: u16,
    request: &'a CapabilityRequest,
    outcome: CapabilityEvidenceOutcome,
    reason: Option<&'a str>,
    receipt_sha256: &'a str,
    source_manifest_sha256: &'a str,
    independent_verifier_sha256: &'a str,
    renderer_build_sha256: &'a str,
    asset_sha256: Option<&'a str>,
    verifier_key_id: &'a str,
}

/// Canonical message signed by the independent capability verifier. It binds
/// every decision-relevant field, including the exact request and whether the
/// verifier concluded Qualified or Unsupported. Keep struct field order and
/// schema version stable; any change requires a schema-version increment.
pub fn capability_attestation_message(
    cell: &CapabilityEvidenceCell,
) -> Result<Vec<u8>, String> {
    let Some(proof) = &cell.proof else {
        return Err("capability cell has no proof to attest".into());
    };
    if cell.outcome == CapabilityEvidenceOutcome::NotEvaluated {
        return Err("NotEvaluated cells cannot carry a capability attestation".into());
    }
    let payload = CapabilityAttestationPayload {
        domain: "luminous-dynamics/muse-instrument-capability",
        schema_version: 1,
        request: &cell.request,
        outcome: cell.outcome,
        reason: cell.reason.as_deref(),
        receipt_sha256: &proof.receipt_sha256,
        source_manifest_sha256: &proof.source_manifest_sha256,
        independent_verifier_sha256: &proof.independent_verifier_sha256,
        renderer_build_sha256: &proof.renderer_build_sha256,
        asset_sha256: proof.asset_sha256.as_deref(),
        verifier_key_id: &proof.verifier_key_id,
    };
    serde_json::to_vec(&payload)
        .map_err(|error| format!("failed to serialize capability attestation: {error}"))
}

/// Decision returned to the composer/arranger. Unknown is not Supported;
/// Unsupported is not the same as invalid evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum CapabilityDecision {
    Supported { receipt_sha256: String },
    Unsupported { reason: String, receipt_sha256: String },
    Unknown { reason: String },
    InvalidEvidence { reason: String },
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_request(request: &CapabilityRequest) -> Result<(), String> {
    if !instrument_descriptors()
        .iter()
        .any(|descriptor| descriptor.id == request.instrument_id.as_str())
    {
        return Err(format!("unknown instrument stable ID: {}", request.instrument_id));
    }
    for (field, value) in [
        ("setup_id", request.setup_id.as_str()),
        ("technique_id", request.technique_id.as_str()),
        ("articulation_id", request.articulation_id.as_str()),
        ("renderer_id", request.renderer_id.as_str()),
    ] {
        if !nonempty(value) {
            return Err(format!("{field} must be non-empty"));
        }
    }
    if request.velocity > 127 {
        return Err(format!(
            "velocity {} is outside the supported 0..=127 representation",
            request.velocity
        ));
    }
    if let CapabilityPitch::Unpitched(token) = &request.pitch {
        if !nonempty(token) {
            return Err("unpitched capability token must be non-empty".into());
        }
    }
    if !valid_sha256(&request.renderer_build_sha256) {
        return Err("renderer_build_sha256 must be a 64-character hexadecimal SHA-256 digest".into());
    }
    Ok(())
}

fn decode_signature_hex(value: &str) -> Result<[u8; 64], String> {
    if value.len() != 128 {
        return Err("attestation_signature_hex must encode exactly 64 bytes".into());
    }
    let mut signature = [0u8; 64];
    for (index, byte) in signature.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|_| "attestation_signature_hex contains non-hexadecimal data".to_string())?;
    }
    Ok(signature)
}

fn validate_proof(
    cell: &CapabilityEvidenceCell,
    trust_store: &CapabilityTrustStore,
) -> Result<(), String> {
    let request = &cell.request;
    let proof = cell.proof.as_ref().ok_or_else(|| "capability cell is missing its proof".to_string())?;
    for (field, value) in [
        ("receipt_sha256", proof.receipt_sha256.as_str()),
        ("source_manifest_sha256", proof.source_manifest_sha256.as_str()),
        (
            "independent_verifier_sha256",
            proof.independent_verifier_sha256.as_str(),
        ),
        ("renderer_build_sha256", proof.renderer_build_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(format!("{field} must be a 64-character hexadecimal SHA-256 digest"));
        }
    }
    if proof.renderer_build_sha256 != request.renderer_build_sha256 {
        return Err("proof renderer build does not match the exact requested renderer build".into());
    }
    if !nonempty(&proof.verifier_key_id) {
        return Err("verifier_key_id must be non-empty".into());
    }
    if let Some(asset_sha256) = &proof.asset_sha256 {
        if !valid_sha256(asset_sha256) {
            return Err("asset_sha256 must be a 64-character hexadecimal SHA-256 digest".into());
        }
    }
    if matches!(
        request.realization,
        CapabilityRealization::SampleAsset | CapabilityRealization::Hybrid
    ) && proof.asset_sha256.is_none()
    {
        return Err("sample-backed and hybrid capability cells require an immutable asset SHA-256".into());
    }

    let trusted = trust_store.verifier(&proof.verifier_key_id).ok_or_else(|| {
        format!("verifier key {} is not pinned by the application trust policy", proof.verifier_key_id)
    })?;
    if proof.independent_verifier_sha256 != trusted.verifier_build_sha256 {
        return Err("independent verifier build does not match the build pinned for this trusted key".into());
    }
    let public_key = VerifyingKey::from_bytes(&trusted.public_key)
        .map_err(|_| "pinned verifier public key is invalid".to_string())?;
    let signature_bytes = decode_signature_hex(&proof.attestation_signature_hex)?;
    let signature = Signature::from_bytes(&signature_bytes);
    let message = capability_attestation_message(cell)?;
    public_key
        .verify_strict(&message, &signature)
        .map_err(|_| "capability attestation signature is invalid for this exact cell".to_string())?;
    Ok(())
}

/// Resolve a request against immutable evidence cells.
///
/// - No exact cell: `Unknown`.
/// - One exact cell with a valid attestation from a verifier key pinned in
///   the supplied application trust store: `Supported` or `Unsupported`.
/// - Duplicate exact cells or malformed/conflicting receipts:
///   `InvalidEvidence`.
///
/// No wildcard matching, register interpolation, velocity interpolation,
/// instrument-family inheritance, or renderer fallback is performed here.
pub fn resolve_capability(
    request: &CapabilityRequest,
    cells: &[CapabilityEvidenceCell],
    trust_store: &CapabilityTrustStore,
) -> CapabilityDecision {
    if let Err(reason) = validate_request(request) {
        return CapabilityDecision::InvalidEvidence { reason };
    }

    let mut exact = cells.iter().filter(|cell| &cell.request == request);
    let Some(cell) = exact.next() else {
        return CapabilityDecision::Unknown {
            reason: "no evidence cell matches the complete instrument/setup/technique/articulation/pitch/velocity/realization/renderer-build request".into(),
        };
    };
    if exact.next().is_some() {
        return CapabilityDecision::InvalidEvidence {
            reason: "multiple evidence cells claim the same exact capability request".into(),
        };
    }

    match cell.outcome {
        CapabilityEvidenceOutcome::NotEvaluated => {
            if cell.proof.is_some() {
                return CapabilityDecision::InvalidEvidence {
                    reason: "NotEvaluated capability cell must not carry a qualification proof".into(),
                };
            }
            CapabilityDecision::Unknown {
                reason: cell
                    .reason
                    .clone()
                    .filter(|reason| nonempty(reason))
                    .unwrap_or_else(|| "capability cell has not been evaluated".into()),
            }
        }
        CapabilityEvidenceOutcome::Qualified => {
            if let Some(reason) = &cell.reason {
                if nonempty(reason) {
                    return CapabilityDecision::InvalidEvidence {
                        reason: "Qualified capability cell cannot simultaneously declare an unsupported/abstention reason".into(),
                    };
                }
            }
            let Some(proof) = &cell.proof else {
                return CapabilityDecision::InvalidEvidence {
                    reason: "Qualified capability cell is missing its proof receipt".into(),
                };
            };
            match validate_proof(cell, trust_store) {
                Ok(()) => CapabilityDecision::Supported {
                    receipt_sha256: proof.receipt_sha256.clone(),
                },
                Err(reason) => CapabilityDecision::InvalidEvidence { reason },
            }
        }
        CapabilityEvidenceOutcome::Unsupported => {
            let Some(reason) = cell.reason.clone().filter(|reason| nonempty(reason)) else {
                return CapabilityDecision::InvalidEvidence {
                    reason: "Unsupported capability cell requires a non-empty reason".into(),
                };
            };
            let Some(proof) = &cell.proof else {
                return CapabilityDecision::InvalidEvidence {
                    reason: "Unsupported capability cell requires a verifier-backed receipt".into(),
                };
            };
            match validate_proof(cell, trust_store) {
                Ok(()) => CapabilityDecision::Unsupported {
                    reason,
                    receipt_sha256: proof.receipt_sha256.clone(),
                },
                Err(reason) => CapabilityDecision::InvalidEvidence { reason },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    fn signing_key() -> SigningKey {
        // Deterministic test-only seed. Never use this key outside fixtures.
        SigningKey::from_bytes(&[
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60,
            0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c, 0xc4,
            0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19,
            0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae, 0x7f, 0x60,
        ])
    }

    fn encode_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut value = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            value.push(HEX[(byte >> 4) as usize] as char);
            value.push(HEX[(byte & 0x0f) as usize] as char);
        }
        value
    }

    fn trust_store(key: &SigningKey) -> CapabilityTrustStore {
        CapabilityTrustStore::new([TrustedCapabilityVerifier {
            key_id: "muse-qualification-test-v1".into(),
            public_key: key.verifying_key().to_bytes(),
            verifier_build_sha256: digest('d'),
        }])
        .expect("test verifier trust root must be valid")
    }

    fn request() -> CapabilityRequest {
        CapabilityRequest {
            instrument_id: "violin".into(),
            setup_id: "default".into(),
            technique_id: "arco".into(),
            articulation_id: "sustain".into(),
            pitch: CapabilityPitch::PitchedMidi(67),
            velocity: 80,
            realization: CapabilityRealization::SampleAsset,
            renderer_id: "muse-fluid-sampler".into(),
            renderer_build_sha256: digest('a'),
        }
    }

    fn proof(
        request: &CapabilityRequest,
        outcome: CapabilityEvidenceOutcome,
        reason: Option<&str>,
        signing_key: &SigningKey,
    ) -> CapabilityProof {
        let mut proof = CapabilityProof {
            receipt_sha256: digest('b'),
            source_manifest_sha256: digest('c'),
            independent_verifier_sha256: digest('d'),
            renderer_build_sha256: request.renderer_build_sha256.clone(),
            asset_sha256: Some(digest('e')),
            verifier_key_id: "muse-qualification-test-v1".into(),
            attestation_signature_hex: String::new(),
        };
        let cell = CapabilityEvidenceCell {
            request: request.clone(),
            outcome,
            reason: reason.map(str::to_owned),
            proof: Some(proof.clone()),
        };
        let message = capability_attestation_message(&cell)
            .expect("well-formed cell must produce the signing message");
        proof.attestation_signature_hex = encode_hex(&signing_key.sign(&message).to_bytes());
        proof
    }

    fn cell(
        request: &CapabilityRequest,
        outcome: CapabilityEvidenceOutcome,
        reason: Option<&str>,
        signing_key: &SigningKey,
    ) -> CapabilityEvidenceCell {
        CapabilityEvidenceCell {
            request: request.clone(),
            outcome,
            reason: reason.map(str::to_owned),
            proof: Some(proof(request, outcome, reason, signing_key)),
        }
    }

    #[test]
    fn exact_cell_with_pinned_verifier_attestation_is_supported() {
        let request = request();
        let key = signing_key();
        let trust = trust_store(&key);
        let qualified = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);
        assert_eq!(
            resolve_capability(&request, &[qualified], &trust),
            CapabilityDecision::Supported {
                receipt_sha256: digest('b')
            }
        );
    }

    #[test]
    fn evidence_does_not_transfer_across_any_exact_request_dimension() {
        let original = request();
        let key = signing_key();
        let trust = trust_store(&key);
        let qualified = cell(&original, CapabilityEvidenceOutcome::Qualified, None, &key);
        let mut variants = Vec::new();

        let mut changed = original.clone();
        changed.pitch = CapabilityPitch::PitchedMidi(68);
        variants.push(changed);

        let mut changed = original.clone();
        changed.velocity = 81;
        variants.push(changed);

        let mut changed = original.clone();
        changed.technique_id = "pizzicato".into();
        variants.push(changed);

        let mut changed = original.clone();
        changed.articulation_id = "staccato".into();
        variants.push(changed);

        let mut changed = original.clone();
        changed.setup_id = "scordatura".into();
        variants.push(changed);

        let mut changed = original.clone();
        changed.realization = CapabilityRealization::ProceduralSynthesis;
        variants.push(changed);

        let mut changed = original.clone();
        changed.renderer_id = "another-renderer".into();
        variants.push(changed);

        let mut changed = original.clone();
        changed.renderer_build_sha256 = digest('f');
        variants.push(changed);

        for changed in variants {
            assert!(
                matches!(
                    resolve_capability(
                        &changed,
                        std::slice::from_ref(&qualified),
                        &trust
                    ),
                    CapabilityDecision::Unknown { .. }
                ),
                "neighbouring request inherited support: {changed:?}"
            );
        }
    }

    #[test]
    fn procedural_capability_does_not_require_a_sample_asset_digest() {
        let mut request = request();
        request.realization = CapabilityRealization::ProceduralSynthesis;
        let key = signing_key();
        let trust = trust_store(&key);
        let mut proof = proof(
            &request,
            CapabilityEvidenceOutcome::Qualified,
            None,
            &key,
        );
        proof.asset_sha256 = None;
        let cell = CapabilityEvidenceCell {
            request: request.clone(),
            outcome: CapabilityEvidenceOutcome::Qualified,
            reason: None,
            proof: Some(proof),
        };
        // The signature is deliberately regenerated after changing the asset
        // field because the attestation binds the complete proof metadata.
        let mut cell = cell;
        let message = capability_attestation_message(&cell).unwrap();
        cell.proof.as_mut().unwrap().attestation_signature_hex =
            encode_hex(&key.sign(&message).to_bytes());
        assert!(matches!(
            resolve_capability(&request, &[cell], &trust),
            CapabilityDecision::Supported { .. }
        ));
    }

    #[test]
    fn explicit_negative_evidence_is_unsupported_not_unknown() {
        let request = request();
        let key = signing_key();
        let trust = trust_store(&key);
        let cell = cell(
            &request,
            CapabilityEvidenceOutcome::Unsupported,
            Some("this renderer has no arco sample at this pitch"),
            &key,
        );
        assert!(matches!(
            resolve_capability(&request, &[cell], &trust),
            CapabilityDecision::Unsupported { .. }
        ));
    }

    #[test]
    fn unsigned_or_malformed_qualified_proof_is_invalid_not_supported() {
        let request = request();
        let key = signing_key();
        let trust = trust_store(&key);

        let mut unsigned = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);
        unsigned.proof = None;
        assert!(matches!(
            resolve_capability(&request, &[unsigned], &trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut missing_asset = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);
        missing_asset.proof.as_mut().unwrap().asset_sha256 = None;
        assert!(matches!(
            resolve_capability(&request, &[missing_asset], &trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut wrong_renderer = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);
        wrong_renderer.proof.as_mut().unwrap().renderer_build_sha256 = digest('f');
        assert!(matches!(
            resolve_capability(&request, &[wrong_renderer], &trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut bad_signature = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);
        bad_signature.proof.as_mut().unwrap().attestation_signature_hex = "00".repeat(64);
        assert!(matches!(
            resolve_capability(&request, &[bad_signature], &trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }

    #[test]
    fn no_pinned_trust_root_means_claims_cannot_be_promoted() {
        let request = request();
        let key = signing_key();
        let empty_trust = CapabilityTrustStore::default();
        let qualified = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);
        assert!(matches!(
            resolve_capability(&request, &[qualified], &empty_trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }

    #[test]
    fn attestation_binds_outcome_reason_and_verifier_build() {
        let request = request();
        let key = signing_key();
        let trust = trust_store(&key);
        let qualified = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);

        let mut forged_outcome = qualified.clone();
        forged_outcome.outcome = CapabilityEvidenceOutcome::Unsupported;
        forged_outcome.reason = Some("forged negative reason".into());
        assert!(matches!(
            resolve_capability(&request, &[forged_outcome], &trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut forged_build = qualified;
        forged_build.proof.as_mut().unwrap().independent_verifier_sha256 = digest('f');
        assert!(matches!(
            resolve_capability(&request, &[forged_build], &trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }

    #[test]
    fn duplicate_exact_cells_are_invalid_evidence() {
        let request = request();
        let key = signing_key();
        let trust = trust_store(&key);
        let qualified = cell(&request, CapabilityEvidenceOutcome::Qualified, None, &key);
        assert!(matches!(
            resolve_capability(
                &request,
                &[qualified.clone(), qualified],
                &trust
            ),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }

    #[test]
    fn not_evaluated_cell_remains_unknown() {
        let request = request();
        let empty_trust = CapabilityTrustStore::default();
        let cell = CapabilityEvidenceCell {
            request: request.clone(),
            outcome: CapabilityEvidenceOutcome::NotEvaluated,
            reason: Some("not yet rendered".into()),
            proof: None,
        };
        assert_eq!(
            resolve_capability(&request, &[cell], &empty_trust),
            CapabilityDecision::Unknown {
                reason: "not yet rendered".into()
            }
        );
    }

    #[test]
    fn malformed_requests_and_unknown_instruments_fail_closed() {
        let empty_trust = CapabilityTrustStore::default();
        let mut malformed = request();
        malformed.renderer_build_sha256 = "not-a-digest".into();
        assert!(matches!(
            resolve_capability(&malformed, &[], &empty_trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut unknown = request();
        unknown.instrument_id = "not-in-catalogue".into();
        assert!(matches!(
            resolve_capability(&unknown, &[], &empty_trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut bad_velocity = request();
        bad_velocity.velocity = 200;
        assert!(matches!(
            resolve_capability(&bad_velocity, &[], &empty_trust),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }
}
