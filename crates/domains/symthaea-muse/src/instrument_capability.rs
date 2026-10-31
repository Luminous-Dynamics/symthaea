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

use serde::{Deserialize, Serialize};

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
    /// Independent verifier implementation/build digest.
    pub independent_verifier_sha256: String,
    /// Must equal the renderer build requested by the capability cell.
    pub renderer_build_sha256: String,
    /// Required for sample-backed or hybrid realization; optional for a
    /// purely procedural synthesis model.
    pub asset_sha256: Option<String>,
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

fn validate_proof(
    request: &CapabilityRequest,
    proof: &CapabilityProof,
) -> Result<(), String> {
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
    Ok(())
}

/// Resolve a request against immutable evidence cells.
///
/// - No exact cell: `Unknown`.
/// - One well-formed qualified cell: `Supported`.
/// - One well-formed negative cell: `Unsupported`.
/// - Duplicate exact cells or malformed/conflicting receipts:
///   `InvalidEvidence`.
///
/// No wildcard matching, register interpolation, velocity interpolation,
/// instrument-family inheritance, or renderer fallback is performed here.
pub fn resolve_capability(
    request: &CapabilityRequest,
    cells: &[CapabilityEvidenceCell],
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
            match validate_proof(request, proof) {
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
            match validate_proof(request, proof) {
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

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
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

    fn proof(request: &CapabilityRequest) -> CapabilityProof {
        CapabilityProof {
            receipt_sha256: digest('b'),
            source_manifest_sha256: digest('c'),
            independent_verifier_sha256: digest('d'),
            renderer_build_sha256: request.renderer_build_sha256.clone(),
            asset_sha256: Some(digest('e')),
        }
    }

    fn qualified_cell(request: &CapabilityRequest) -> CapabilityEvidenceCell {
        CapabilityEvidenceCell {
            request: request.clone(),
            outcome: CapabilityEvidenceOutcome::Qualified,
            reason: None,
            proof: Some(proof(request)),
        }
    }

    #[test]
    fn exact_qualified_cell_is_supported() {
        let request = request();
        let decision = resolve_capability(&request, &[qualified_cell(&request)]);
        assert_eq!(
            decision,
            CapabilityDecision::Supported {
                receipt_sha256: digest('b')
            }
        );
    }

    #[test]
    fn evidence_does_not_transfer_across_pitch_velocity_technique_or_build() {
        let original = request();
        let cell = qualified_cell(&original);
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
                    resolve_capability(&changed, std::slice::from_ref(&cell)),
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
        let mut proof = proof(&request);
        proof.asset_sha256 = None;
        let cell = CapabilityEvidenceCell {
            request: request.clone(),
            outcome: CapabilityEvidenceOutcome::Qualified,
            reason: None,
            proof: Some(proof),
        };
        assert!(matches!(
            resolve_capability(&request, &[cell]),
            CapabilityDecision::Supported { .. }
        ));
    }

    #[test]
    fn explicit_negative_evidence_is_unsupported_not_unknown() {
        let request = request();
        let cell = CapabilityEvidenceCell {
            request: request.clone(),
            outcome: CapabilityEvidenceOutcome::Unsupported,
            reason: Some("this renderer has no arco sample at this pitch".into()),
            proof: Some(proof(&request)),
        };
        assert!(matches!(
            resolve_capability(&request, &[cell]),
            CapabilityDecision::Unsupported { .. }
        ));
    }

    #[test]
    fn missing_or_malformed_qualified_proof_is_invalid_not_supported() {
        let request = request();
        let mut cell = qualified_cell(&request);
        cell.proof = None;
        assert!(matches!(
            resolve_capability(&request, &[cell]),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut cell = qualified_cell(&request);
        cell.proof.as_mut().unwrap().asset_sha256 = None;
        assert!(matches!(
            resolve_capability(&request, &[cell]),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut cell = qualified_cell(&request);
        cell.proof.as_mut().unwrap().renderer_build_sha256 = digest('f');
        assert!(matches!(
            resolve_capability(&request, &[cell]),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }

    #[test]
    fn duplicate_exact_cells_are_invalid_evidence() {
        let request = request();
        let cell = qualified_cell(&request);
        assert!(matches!(
            resolve_capability(&request, &[cell.clone(), cell]),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }

    #[test]
    fn not_evaluated_cell_remains_unknown() {
        let request = request();
        let cell = CapabilityEvidenceCell {
            request: request.clone(),
            outcome: CapabilityEvidenceOutcome::NotEvaluated,
            reason: Some("not yet rendered".into()),
            proof: None,
        };
        assert_eq!(
            resolve_capability(&request, &[cell]),
            CapabilityDecision::Unknown {
                reason: "not yet rendered".into()
            }
        );
    }

    #[test]
    fn malformed_requests_and_unknown_instruments_fail_closed() {
        let mut malformed = request();
        malformed.renderer_build_sha256 = "not-a-digest".into();
        assert!(matches!(
            resolve_capability(&malformed, &[]),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut unknown = request();
        unknown.instrument_id = "not-in-catalogue".into();
        assert!(matches!(
            resolve_capability(&unknown, &[]),
            CapabilityDecision::InvalidEvidence { .. }
        ));

        let mut bad_velocity = request();
        bad_velocity.velocity = 200;
        assert!(matches!(
            resolve_capability(&bad_velocity, &[]),
            CapabilityDecision::InvalidEvidence { .. }
        ));
    }
}
