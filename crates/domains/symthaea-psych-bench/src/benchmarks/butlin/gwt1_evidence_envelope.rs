// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Raw-artifact integrity envelope for direct Butlin GWT-1 qualification.
//!
//! A summary receipt is not sufficient evidence by itself. The authoritative
//! resolver in this module requires the exact raw observation bytes, verifies
//! their declared length and BLAKE3 digest, and only then delegates to the
//! typed GWT-1 qualification resolver.

use serde::{Deserialize, Serialize};

use super::gwt1_qualification::{
    Gwt1QualificationOutcomeV1, Gwt1QualificationResolutionV1,
    Gwt1SpecialistQualificationReceiptV1, resolve_gwt1_v1,
};

pub const GWT1_EVIDENCE_ENVELOPE_SCHEMA_V1: &str = "butlin-gwt1-evidence-envelope-v1";
pub const GWT1_RAW_OBSERVATION_MEDIA_TYPE_V1: &str = "application/json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1RawObservationArtifactV1 {
    pub schema: String,
    pub media_type: String,
    pub byte_len: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gwt1EvidenceEnvelopeV1 {
    pub schema: String,
    pub raw_observations: Gwt1RawObservationArtifactV1,
    pub receipt: Gwt1SpecialistQualificationReceiptV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Gwt1ArtifactIntegrityFailureV1 {
    InvalidEnvelopeSchema { observed: String },
    MissingRawObservationSchema,
    InvalidMediaType { observed: String },
    InvalidDigestFormat { observed: String },
    RawObservationLengthMismatch { declared: u64, observed: u64 },
    RawObservationDigestMismatch { declared: String, observed: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gwt1EvidenceEnvelopeResolutionV1 {
    pub outcome: Gwt1QualificationOutcomeV1,
    pub artifact_failures: Vec<Gwt1ArtifactIntegrityFailureV1>,
    pub receipt_resolution: Gwt1QualificationResolutionV1,
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn raw_observation_blake3(raw_observations: &[u8]) -> String {
    blake3::hash(raw_observations).to_hex().to_string()
}

pub fn describe_raw_observations_v1(
    schema: impl Into<String>,
    raw_observations: &[u8],
) -> Gwt1RawObservationArtifactV1 {
    Gwt1RawObservationArtifactV1 {
        schema: schema.into(),
        media_type: GWT1_RAW_OBSERVATION_MEDIA_TYPE_V1.to_string(),
        byte_len: raw_observations.len() as u64,
        blake3: raw_observation_blake3(raw_observations),
    }
}

pub fn resolve_gwt1_evidence_envelope_v1(
    envelope: &Gwt1EvidenceEnvelopeV1,
    raw_observations: &[u8],
) -> Gwt1EvidenceEnvelopeResolutionV1 {
    use Gwt1ArtifactIntegrityFailureV1::*;

    let mut artifact_failures = Vec::new();

    if envelope.schema != GWT1_EVIDENCE_ENVELOPE_SCHEMA_V1 {
        artifact_failures.push(InvalidEnvelopeSchema {
            observed: envelope.schema.clone(),
        });
    }

    if envelope.raw_observations.schema.trim().is_empty() {
        artifact_failures.push(MissingRawObservationSchema);
    }

    if envelope.raw_observations.media_type != GWT1_RAW_OBSERVATION_MEDIA_TYPE_V1 {
        artifact_failures.push(InvalidMediaType {
            observed: envelope.raw_observations.media_type.clone(),
        });
    }

    if !is_lower_hex_64(&envelope.raw_observations.blake3) {
        artifact_failures.push(InvalidDigestFormat {
            observed: envelope.raw_observations.blake3.clone(),
        });
    }

    let observed_len = raw_observations.len() as u64;
    if envelope.raw_observations.byte_len != observed_len {
        artifact_failures.push(RawObservationLengthMismatch {
            declared: envelope.raw_observations.byte_len,
            observed: observed_len,
        });
    }

    let observed_digest = raw_observation_blake3(raw_observations);
    if envelope.raw_observations.blake3 != observed_digest {
        artifact_failures.push(RawObservationDigestMismatch {
            declared: envelope.raw_observations.blake3.clone(),
            observed: observed_digest,
        });
    }

    let receipt_resolution = resolve_gwt1_v1(&envelope.receipt);
    let outcome = if artifact_failures.is_empty() {
        receipt_resolution.outcome
    } else {
        Gwt1QualificationOutcomeV1::Inconclusive
    };

    Gwt1EvidenceEnvelopeResolutionV1 {
        outcome,
        artifact_failures,
        receipt_resolution,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::benchmarks::butlin::gwt1_qualification::{
        GWT1_PERTURBATIONS_V1, GWT1_QUALIFICATION_SCHEMA_V1, GWT1_SPECIALISTS_V1,
        Gwt1PerturbationObservationV1, Gwt1SpecialistIdentityV1,
    };

    fn map_all(value: bool) -> BTreeMap<String, bool> {
        GWT1_SPECIALISTS_V1
            .iter()
            .map(|id| ((*id).to_string(), value))
            .collect()
    }

    fn canonical_receipt() -> Gwt1SpecialistQualificationReceiptV1 {
        Gwt1SpecialistQualificationReceiptV1 {
            schema: GWT1_QUALIFICATION_SCHEMA_V1.to_string(),
            source_commit_sha: "a".repeat(40),
            source_tree_sha: "b".repeat(40),
            execution_run_id: "ci-run-123".to_string(),
            toolchain: "rustc-test".to_string(),
            specialists: GWT1_SPECIALISTS_V1
                .iter()
                .map(|id| Gwt1SpecialistIdentityV1 {
                    id: (*id).to_string(),
                    implementation_path: format!("src/cognitive_loop/managers/{id}.rs"),
                    source_blob_sha: "c".repeat(40),
                })
                .collect(),
            perturbations: GWT1_PERTURBATIONS_V1
                .iter()
                .map(|(id, field, target)| Gwt1PerturbationObservationV1 {
                    id: (*id).to_string(),
                    field: (*field).to_string(),
                    target_specialist: (*target).to_string(),
                    baseline_value: 0.4,
                    perturbed_value: 0.8,
                    changed_specialists: BTreeSet::from([(*target).to_string()]),
                })
                .collect(),
            solo_panel_equal: map_all(true),
            trajectory_steps: 48,
            sequential_parallel_equal: map_all(true),
            requested_workers: 4,
            barrier_participants: 4,
            distinct_workers_observed: 4,
            completed_specialists: GWT1_SPECIALISTS_V1
                .iter()
                .map(|id| (*id).to_string())
                .collect(),
            checkpoint_equal_supplementary: Some(true),
        }
    }

    fn canonical_envelope(raw: &[u8]) -> Gwt1EvidenceEnvelopeV1 {
        Gwt1EvidenceEnvelopeV1 {
            schema: GWT1_EVIDENCE_ENVELOPE_SCHEMA_V1.to_string(),
            raw_observations: describe_raw_observations_v1(
                "butlin-gwt1-raw-observations-v1",
                raw,
            ),
            receipt: canonical_receipt(),
        }
    }

    #[test]
    fn intact_artifact_can_resolve_qualified_receipt() {
        let raw = br#"{"schema":"butlin-gwt1-raw-observations-v1","observations":[]}"#;
        let resolution = resolve_gwt1_evidence_envelope_v1(&canonical_envelope(raw), raw);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Qualified);
        assert!(resolution.artifact_failures.is_empty());
    }

    #[test]
    fn detached_or_tampered_artifact_is_inconclusive() {
        let raw = br#"{"schema":"butlin-gwt1-raw-observations-v1","observations":[]}"#;
        let envelope = canonical_envelope(raw);
        let tampered = br#"{"schema":"butlin-gwt1-raw-observations-v1","observations":[1]}"#;
        let resolution = resolve_gwt1_evidence_envelope_v1(&envelope, tampered);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
        assert!(resolution.artifact_failures.iter().any(|failure| matches!(
            failure,
            Gwt1ArtifactIntegrityFailureV1::RawObservationDigestMismatch { .. }
        )));
    }

    #[test]
    fn summary_receipt_cannot_override_length_mismatch() {
        let raw = b"canonical raw bytes";
        let mut envelope = canonical_envelope(raw);
        envelope.raw_observations.byte_len += 1;
        let resolution = resolve_gwt1_evidence_envelope_v1(&envelope, raw);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
    }

    #[test]
    fn malformed_digest_is_inconclusive_even_if_receipt_is_qualified() {
        let raw = b"canonical raw bytes";
        let mut envelope = canonical_envelope(raw);
        envelope.raw_observations.blake3 = "ABC".to_string();
        let resolution = resolve_gwt1_evidence_envelope_v1(&envelope, raw);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Inconclusive);
        assert!(resolution.artifact_failures.iter().any(|failure| matches!(
            failure,
            Gwt1ArtifactIntegrityFailureV1::InvalidDigestFormat { .. }
        )));
    }

    #[test]
    fn valid_artifact_preserves_inner_contradicted_outcome() {
        let raw = b"canonical raw bytes";
        let mut envelope = canonical_envelope(raw);
        envelope.receipt.sequential_parallel_equal.insert(
            "memory_manager".to_string(),
            false,
        );
        let resolution = resolve_gwt1_evidence_envelope_v1(&envelope, raw);
        assert_eq!(resolution.outcome, Gwt1QualificationOutcomeV1::Contradicted);
        assert!(resolution.artifact_failures.is_empty());
    }

    #[test]
    fn envelope_json_roundtrip_preserves_integrity_descriptor() {
        let raw = b"canonical raw bytes";
        let envelope = canonical_envelope(raw);
        let json = serde_json::to_vec(&envelope).expect("serialize envelope");
        let restored: Gwt1EvidenceEnvelopeV1 =
            serde_json::from_slice(&json).expect("deserialize envelope");
        assert_eq!(restored, envelope);
        assert_eq!(
            resolve_gwt1_evidence_envelope_v1(&restored, raw).outcome,
            Gwt1QualificationOutcomeV1::Qualified
        );
    }
}
