// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integrity and execution-identity envelope for causal Butlin GWT-1 evidence.
//!
//! A causal claim is not allowed to detach the target-active lesion corpus from
//! its matched omission control. This envelope binds both exact JSON byte
//! streams, one execution identity, and the fail-closed causal resolver.
//!
//! Integrity or identity failure forces the outer result to `Inconclusive`,
//! even when the inner causal protocol would otherwise resolve more strongly.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea::benchmarks::gwt1_causal_lesion::{
    GWT1_CAUSAL_LESION_SCHEMA_V1, Gwt1CausalObservationsV1,
};
use symthaea::benchmarks::gwt1_causal_matched_sham::{
    GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1, Gwt1MatchedShamObservationsV1,
};

use super::gwt1_causal_resolution::{
    Gwt1CausalQualificationOutcomeV1, Gwt1CausalQualificationResolutionV1,
    resolve_gwt1_causal_v1, run_gwt1_causal_evidence_v1,
};
use super::gwt1_end_to_end::Gwt1ExecutionIdentityV1;
use super::gwt1_evidence_envelope::{
    GWT1_RAW_OBSERVATION_MEDIA_TYPE_V1, raw_observation_blake3,
};
use super::gwt1_qualification::GWT1_SPECIALISTS_V1;

pub const GWT1_CAUSAL_EVIDENCE_ENVELOPE_SCHEMA_V1: &str =
    "butlin-gwt1-causal-evidence-envelope-v1";

const CAUSAL_ARTIFACT_ID: &str = "causal_observations";
const MATCHED_SHAM_ARTIFACT_ID: &str = "matched_sham_observations";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalArtifactDescriptorV1 {
    pub schema: String,
    pub media_type: String,
    pub byte_len: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalEvidenceEnvelopeV1 {
    pub schema: String,
    pub causal_observations: Gwt1CausalArtifactDescriptorV1,
    pub matched_sham_observations: Gwt1CausalArtifactDescriptorV1,
    pub execution_identity: Gwt1ExecutionIdentityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Gwt1CausalArtifactFailureV1 {
    InvalidEnvelopeSchema {
        observed: String,
    },
    InvalidArtifactSchema {
        artifact: String,
        observed: String,
        expected: String,
    },
    InvalidMediaType {
        artifact: String,
        observed: String,
    },
    InvalidDigestFormat {
        artifact: String,
        observed: String,
    },
    LengthMismatch {
        artifact: String,
        declared: u64,
        observed: u64,
    },
    DigestMismatch {
        artifact: String,
        declared: String,
        observed: String,
    },
    JsonDecode {
        artifact: String,
        error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Gwt1CausalIdentityFailureV1 {
    InvalidSourceCommitSha { observed: String },
    InvalidSourceTreeSha { observed: String },
    EmptyExecutionRunId,
    EmptyToolchain,
    SpecialistSetMismatch { observed: Vec<String> },
    InvalidSpecialistBlobSha { specialist: String, observed: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalEvidenceEnvelopeResolutionV1 {
    pub outcome: Gwt1CausalQualificationOutcomeV1,
    pub artifact_failures: Vec<Gwt1CausalArtifactFailureV1>,
    pub identity_failures: Vec<Gwt1CausalIdentityFailureV1>,
    pub causal_resolution: Option<Gwt1CausalQualificationResolutionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gwt1CausalEnvelopedEvidenceV1 {
    pub causal_raw_bytes: Vec<u8>,
    pub matched_sham_raw_bytes: Vec<u8>,
    pub envelope: Gwt1CausalEvidenceEnvelopeV1,
    pub resolution: Gwt1CausalEvidenceEnvelopeResolutionV1,
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn describe_gwt1_causal_artifact_v1(
    schema: &str,
    raw_bytes: &[u8],
) -> Gwt1CausalArtifactDescriptorV1 {
    Gwt1CausalArtifactDescriptorV1 {
        schema: schema.to_string(),
        media_type: GWT1_RAW_OBSERVATION_MEDIA_TYPE_V1.to_string(),
        byte_len: raw_bytes.len() as u64,
        blake3: raw_observation_blake3(raw_bytes),
    }
}

pub fn build_gwt1_causal_envelope_v1(
    identity: &Gwt1ExecutionIdentityV1,
    causal_raw_bytes: &[u8],
    matched_sham_raw_bytes: &[u8],
) -> Gwt1CausalEvidenceEnvelopeV1 {
    Gwt1CausalEvidenceEnvelopeV1 {
        schema: GWT1_CAUSAL_EVIDENCE_ENVELOPE_SCHEMA_V1.to_string(),
        causal_observations: describe_gwt1_causal_artifact_v1(
            GWT1_CAUSAL_LESION_SCHEMA_V1,
            causal_raw_bytes,
        ),
        matched_sham_observations: describe_gwt1_causal_artifact_v1(
            GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1,
            matched_sham_raw_bytes,
        ),
        execution_identity: identity.clone(),
    }
}

fn validate_artifact(
    artifact_id: &str,
    descriptor: &Gwt1CausalArtifactDescriptorV1,
    expected_schema: &str,
    raw_bytes: &[u8],
    failures: &mut Vec<Gwt1CausalArtifactFailureV1>,
) {
    if descriptor.schema != expected_schema {
        failures.push(Gwt1CausalArtifactFailureV1::InvalidArtifactSchema {
            artifact: artifact_id.to_string(),
            observed: descriptor.schema.clone(),
            expected: expected_schema.to_string(),
        });
    }
    if descriptor.media_type != GWT1_RAW_OBSERVATION_MEDIA_TYPE_V1 {
        failures.push(Gwt1CausalArtifactFailureV1::InvalidMediaType {
            artifact: artifact_id.to_string(),
            observed: descriptor.media_type.clone(),
        });
    }
    if !is_lower_hex(&descriptor.blake3, 64) {
        failures.push(Gwt1CausalArtifactFailureV1::InvalidDigestFormat {
            artifact: artifact_id.to_string(),
            observed: descriptor.blake3.clone(),
        });
    }

    let observed_len = raw_bytes.len() as u64;
    if descriptor.byte_len != observed_len {
        failures.push(Gwt1CausalArtifactFailureV1::LengthMismatch {
            artifact: artifact_id.to_string(),
            declared: descriptor.byte_len,
            observed: observed_len,
        });
    }

    let observed_digest = raw_observation_blake3(raw_bytes);
    if descriptor.blake3 != observed_digest {
        failures.push(Gwt1CausalArtifactFailureV1::DigestMismatch {
            artifact: artifact_id.to_string(),
            declared: descriptor.blake3.clone(),
            observed: observed_digest,
        });
    }
}

fn validate_identity(
    identity: &Gwt1ExecutionIdentityV1,
) -> Vec<Gwt1CausalIdentityFailureV1> {
    let mut failures = Vec::new();

    if !is_lower_hex(&identity.source_commit_sha, 40) {
        failures.push(Gwt1CausalIdentityFailureV1::InvalidSourceCommitSha {
            observed: identity.source_commit_sha.clone(),
        });
    }
    if !is_lower_hex(&identity.source_tree_sha, 40) {
        failures.push(Gwt1CausalIdentityFailureV1::InvalidSourceTreeSha {
            observed: identity.source_tree_sha.clone(),
        });
    }
    if identity.execution_run_id.trim().is_empty() {
        failures.push(Gwt1CausalIdentityFailureV1::EmptyExecutionRunId);
    }
    if identity.toolchain.trim().is_empty() {
        failures.push(Gwt1CausalIdentityFailureV1::EmptyToolchain);
    }

    let expected: BTreeSet<String> = GWT1_SPECIALISTS_V1
        .iter()
        .map(|id| (*id).to_string())
        .collect();
    let observed: BTreeSet<String> = identity.specialist_blob_shas.keys().cloned().collect();
    if observed != expected {
        failures.push(Gwt1CausalIdentityFailureV1::SpecialistSetMismatch {
            observed: observed.into_iter().collect(),
        });
    }

    for (specialist, sha) in &identity.specialist_blob_shas {
        if !is_lower_hex(sha, 40) {
            failures.push(Gwt1CausalIdentityFailureV1::InvalidSpecialistBlobSha {
                specialist: specialist.clone(),
                observed: sha.clone(),
            });
        }
    }

    failures
}

pub fn resolve_gwt1_causal_envelope_v1(
    envelope: &Gwt1CausalEvidenceEnvelopeV1,
    causal_raw_bytes: &[u8],
    matched_sham_raw_bytes: &[u8],
) -> Gwt1CausalEvidenceEnvelopeResolutionV1 {
    let mut artifact_failures = Vec::new();

    if envelope.schema != GWT1_CAUSAL_EVIDENCE_ENVELOPE_SCHEMA_V1 {
        artifact_failures.push(Gwt1CausalArtifactFailureV1::InvalidEnvelopeSchema {
            observed: envelope.schema.clone(),
        });
    }

    validate_artifact(
        CAUSAL_ARTIFACT_ID,
        &envelope.causal_observations,
        GWT1_CAUSAL_LESION_SCHEMA_V1,
        causal_raw_bytes,
        &mut artifact_failures,
    );
    validate_artifact(
        MATCHED_SHAM_ARTIFACT_ID,
        &envelope.matched_sham_observations,
        GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1,
        matched_sham_raw_bytes,
        &mut artifact_failures,
    );

    let causal = match serde_json::from_slice::<Gwt1CausalObservationsV1>(causal_raw_bytes) {
        Ok(value) => Some(value),
        Err(error) => {
            artifact_failures.push(Gwt1CausalArtifactFailureV1::JsonDecode {
                artifact: CAUSAL_ARTIFACT_ID.to_string(),
                error: error.to_string(),
            });
            None
        }
    };
    let matched_sham =
        match serde_json::from_slice::<Gwt1MatchedShamObservationsV1>(matched_sham_raw_bytes) {
            Ok(value) => Some(value),
            Err(error) => {
                artifact_failures.push(Gwt1CausalArtifactFailureV1::JsonDecode {
                    artifact: MATCHED_SHAM_ARTIFACT_ID.to_string(),
                    error: error.to_string(),
                });
                None
            }
        };

    let identity_failures = validate_identity(&envelope.execution_identity);
    let causal_resolution = causal
        .as_ref()
        .zip(matched_sham.as_ref())
        .map(|(causal, sham)| resolve_gwt1_causal_v1(causal, sham));

    let outcome = if artifact_failures.is_empty() && identity_failures.is_empty() {
        causal_resolution
            .as_ref()
            .map(|resolution| resolution.outcome)
            .unwrap_or(Gwt1CausalQualificationOutcomeV1::Inconclusive)
    } else {
        Gwt1CausalQualificationOutcomeV1::Inconclusive
    };

    Gwt1CausalEvidenceEnvelopeResolutionV1 {
        outcome,
        artifact_failures,
        identity_failures,
        causal_resolution,
    }
}

pub fn run_gwt1_causal_enveloped_evidence_v1(
    identity: &Gwt1ExecutionIdentityV1,
) -> Result<Gwt1CausalEnvelopedEvidenceV1, String> {
    let raw = run_gwt1_causal_evidence_v1()?;
    let envelope = build_gwt1_causal_envelope_v1(
        identity,
        &raw.causal_raw_bytes,
        &raw.matched_sham_raw_bytes,
    );
    let resolution = resolve_gwt1_causal_envelope_v1(
        &envelope,
        &raw.causal_raw_bytes,
        &raw.matched_sham_raw_bytes,
    );

    Ok(Gwt1CausalEnvelopedEvidenceV1 {
        causal_raw_bytes: raw.causal_raw_bytes,
        matched_sham_raw_bytes: raw.matched_sham_raw_bytes,
        envelope,
        resolution,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use symthaea::benchmarks::gwt1_causal_lesion::run_gwt1_causal_lesion_v1;
    use symthaea::benchmarks::gwt1_causal_matched_sham::run_gwt1_causal_matched_sham_v1;
    use symthaea::benchmarks::gwt1_causal_verifier::recompute_gwt1_collector_v1;

    fn identity() -> Gwt1ExecutionIdentityV1 {
        Gwt1ExecutionIdentityV1 {
            source_commit_sha: "a".repeat(40),
            source_tree_sha: "b".repeat(40),
            execution_run_id: "run-123/1".to_string(),
            toolchain: "rustc 1.96.0 (test)".to_string(),
            specialist_blob_shas: GWT1_SPECIALISTS_V1
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    let digit = char::from(b'c' + index as u8);
                    ((*id).to_string(), digit.to_string().repeat(40))
                })
                .collect::<BTreeMap<_, _>>(),
        }
    }

    fn canonical_raw() -> (Vec<u8>, Vec<u8>) {
        let causal = serde_json::to_vec(&run_gwt1_causal_lesion_v1()).expect("causal JSON");
        let sham =
            serde_json::to_vec(&run_gwt1_causal_matched_sham_v1()).expect("matched sham JSON");
        (causal, sham)
    }

    #[test]
    fn intact_pair_can_preserve_protocol_qualified_outcome() {
        let (causal, sham) = canonical_raw();
        let envelope = build_gwt1_causal_envelope_v1(&identity(), &causal, &sham);
        let resolution = resolve_gwt1_causal_envelope_v1(&envelope, &causal, &sham);
        assert_eq!(
            resolution.outcome,
            Gwt1CausalQualificationOutcomeV1::Qualified
        );
        assert!(resolution.artifact_failures.is_empty());
        assert!(resolution.identity_failures.is_empty());
        assert_eq!(
            resolution.causal_resolution.as_ref().map(|inner| inner.outcome),
            Some(Gwt1CausalQualificationOutcomeV1::Qualified)
        );
    }

    #[test]
    fn tampered_causal_bytes_force_outer_inconclusive() {
        let (causal, sham) = canonical_raw();
        let envelope = build_gwt1_causal_envelope_v1(&identity(), &causal, &sham);
        let mut tampered = causal.clone();
        tampered.push(b' ');
        let resolution = resolve_gwt1_causal_envelope_v1(&envelope, &tampered, &sham);
        assert_eq!(
            resolution.outcome,
            Gwt1CausalQualificationOutcomeV1::Inconclusive
        );
        assert!(resolution.artifact_failures.iter().any(|failure| matches!(
            failure,
            Gwt1CausalArtifactFailureV1::DigestMismatch { artifact, .. }
                if artifact == CAUSAL_ARTIFACT_ID
        )));
    }

    #[test]
    fn matched_sham_cannot_be_detached_or_substituted() {
        let (causal, sham) = canonical_raw();
        let envelope = build_gwt1_causal_envelope_v1(&identity(), &causal, &sham);
        let resolution = resolve_gwt1_causal_envelope_v1(&envelope, &causal, &causal);
        assert_eq!(
            resolution.outcome,
            Gwt1CausalQualificationOutcomeV1::Inconclusive
        );
        assert!(resolution.artifact_failures.iter().any(|failure| matches!(
            failure,
            Gwt1CausalArtifactFailureV1::DigestMismatch { artifact, .. }
                if artifact == MATCHED_SHAM_ARTIFACT_ID
        )));
    }

    #[test]
    fn incomplete_execution_identity_is_inconclusive() {
        let (causal, sham) = canonical_raw();
        let mut bad_identity = identity();
        bad_identity.specialist_blob_shas.remove(GWT1_SPECIALISTS_V1[0]);
        let envelope = build_gwt1_causal_envelope_v1(&bad_identity, &causal, &sham);
        let resolution = resolve_gwt1_causal_envelope_v1(&envelope, &causal, &sham);
        assert_eq!(
            resolution.outcome,
            Gwt1CausalQualificationOutcomeV1::Inconclusive
        );
        assert!(resolution.identity_failures.iter().any(|failure| matches!(
            failure,
            Gwt1CausalIdentityFailureV1::SpecialistSetMismatch { .. }
        )));
    }

    #[test]
    fn intact_envelope_preserves_inner_contradicted_outcome() {
        let mut causal = run_gwt1_causal_lesion_v1();
        let sham = run_gwt1_causal_matched_sham_v1();

        let row = &mut causal.rows[0];
        let control = row
            .lesion
            .executed_outputs
            .get_mut("memory_manager")
            .expect("memory control");
        control.confidence_delta ^= 1;
        let recomputed = recompute_gwt1_collector_v1(&row.lesion.executed_outputs, None);
        row.lesion.recorded_contributors = recomputed.0;
        row.lesion.integrated = recomputed.1;

        let causal_bytes = serde_json::to_vec(&causal).expect("causal JSON");
        let sham_bytes = serde_json::to_vec(&sham).expect("sham JSON");
        let envelope =
            build_gwt1_causal_envelope_v1(&identity(), &causal_bytes, &sham_bytes);
        let resolution =
            resolve_gwt1_causal_envelope_v1(&envelope, &causal_bytes, &sham_bytes);

        assert!(resolution.artifact_failures.is_empty());
        assert!(resolution.identity_failures.is_empty());
        assert_eq!(
            resolution.outcome,
            Gwt1CausalQualificationOutcomeV1::Contradicted
        );
    }
}
