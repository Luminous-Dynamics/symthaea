// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Re-verifiable persistence boundary for generativity evidence.
//!
//! Serialized assessments and capsules are deliberately treated as untrusted data.
//! A reserved evidence-plane `kind` string is never sufficient to establish that a
//! generativity assessment is backed by a qualified evidence-plane run. Verification
//! reconstructs every embedded [`RunEvidence`], recomputes its evidence-plane integrity
//! and BLAKE3 envelope, checks the claimed digest, and requires a one-to-one binding
//! between reserved assessment evidence and verified capsules.
//!
//! This layer still provides evidence integrity rather than producer authentication.
//! Provenance/signature references are descriptive metadata until a separate verifier
//! authenticates them.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use symthaea_evidence_plane::RunEvidence;

use super::evidence_binding::{EvidenceEnvelopeError, EvidencePlaneEnvelope};
use super::generativity::{GenerativityAssessment, GenerativityValidationError};

pub const PERSISTED_EVIDENCE_CAPSULE_SCHEMA: &str =
    "symthaea-persisted-evidence-capsule-v1";
pub const PERSISTED_GENERATIVITY_BUNDLE_SCHEMA: &str =
    "symthaea-persisted-generativity-bundle-v1";

const QUALIFIED_KIND: &str = "symthaea-evidence-plane/qualified-v1";
const VIOLATED_KIND: &str = "symthaea-evidence-plane/violated-v1";
const RESERVED_KIND_PREFIX: &str = "symthaea-evidence-plane/";

pub const MAX_CAPSULES_PER_BUNDLE: usize = 256;
pub const MAX_PROVENANCE_REFS: usize = 32;
pub const MAX_PROVENANCE_KIND_LEN: usize = 96;
pub const MAX_PROVENANCE_REFERENCE_LEN: usize = 2_048;
pub const MAX_PROVENANCE_NOTE_LEN: usize = 2_000;

/// Descriptive reference to external authenticity/provenance material.
///
/// Merely carrying this reference does not authenticate the capsule. A future Xenia,
/// Mycelix, signature, TEE, or remote-attestation verifier must explicitly qualify it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceReference {
    pub kind: String,
    pub reference: String,
    pub note: Option<String>,
}

/// Serializable, explicitly untrusted evidence capsule.
///
/// The embedded run is sufficient to recompute the qualified envelope; the claimed
/// digest is checked rather than trusted. External provenance is intentionally outside
/// the envelope digest because it describes authentication of the evidence artifact,
/// not the experiment's measured contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEvidenceCapsule {
    schema: String,
    run: RunEvidence,
    claimed_envelope_digest: String,
    provenance: Vec<ProvenanceReference>,
}

impl PersistedEvidenceCapsule {
    pub fn from_run(
        run: RunEvidence,
        provenance: Vec<ProvenanceReference>,
    ) -> Result<Self, PersistedEvidenceError> {
        validate_provenance(&provenance)?;
        let envelope = EvidencePlaneEnvelope::from_run(&run)?;
        Ok(Self {
            schema: PERSISTED_EVIDENCE_CAPSULE_SCHEMA.to_string(),
            claimed_envelope_digest: envelope.envelope_digest().to_string(),
            run,
            provenance,
        })
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn run(&self) -> &RunEvidence {
        &self.run
    }

    pub fn claimed_envelope_digest(&self) -> &str {
        &self.claimed_envelope_digest
    }

    pub fn provenance(&self) -> &[ProvenanceReference] {
        &self.provenance
    }

    /// Recompute the complete evidence-plane qualification from embedded content.
    pub fn verify(&self) -> Result<VerifiedEvidenceCapsule, PersistedEvidenceError> {
        if self.schema != PERSISTED_EVIDENCE_CAPSULE_SCHEMA {
            return Err(PersistedEvidenceError::UnsupportedCapsuleSchema(
                self.schema.clone(),
            ));
        }
        validate_digest(&self.claimed_envelope_digest)?;
        validate_provenance(&self.provenance)?;

        let envelope = EvidencePlaneEnvelope::from_run(&self.run)?;
        if envelope.envelope_digest() != self.claimed_envelope_digest {
            return Err(PersistedEvidenceError::EnvelopeDigestMismatch {
                claimed: self.claimed_envelope_digest.clone(),
                recomputed: envelope.envelope_digest().to_string(),
            });
        }

        Ok(VerifiedEvidenceCapsule {
            envelope,
            provenance: self.provenance.clone(),
        })
    }
}

/// Constructor-only verified capsule. Intentionally not deserializable.
#[derive(Debug, Clone, Serialize)]
pub struct VerifiedEvidenceCapsule {
    envelope: EvidencePlaneEnvelope,
    provenance: Vec<ProvenanceReference>,
}

impl VerifiedEvidenceCapsule {
    pub fn envelope(&self) -> &EvidencePlaneEnvelope {
        &self.envelope
    }

    pub fn provenance(&self) -> &[ProvenanceReference] {
        &self.provenance
    }

    fn expected_binding(&self) -> EvidenceBindingKey {
        EvidenceBindingKey {
            evidence_id: format!("evidence-plane:{}", self.envelope.run_id()),
            kind: if self.envelope.integrity_satisfied() {
                QUALIFIED_KIND.to_string()
            } else {
                VIOLATED_KIND.to_string()
            },
            reference: format!("blake3:{}", self.envelope.envelope_digest()),
        }
    }
}

/// Serializable bundle combining a generativity assessment with the evidence-plane runs
/// needed to independently re-qualify every reserved evidence-plane reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedGenerativityBundle {
    schema: String,
    assessment: GenerativityAssessment,
    capsules: Vec<PersistedEvidenceCapsule>,
}

impl PersistedGenerativityBundle {
    /// Construct an unverified persistence object. Call [`Self::verify`] before using any
    /// reserved evidence-plane reference as qualified evidence.
    pub fn new(
        assessment: GenerativityAssessment,
        capsules: Vec<PersistedEvidenceCapsule>,
    ) -> Self {
        Self {
            schema: PERSISTED_GENERATIVITY_BUNDLE_SCHEMA.to_string(),
            assessment,
            capsules,
        }
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn assessment(&self) -> &GenerativityAssessment {
        &self.assessment
    }

    pub fn capsules(&self) -> &[PersistedEvidenceCapsule] {
        &self.capsules
    }

    /// Re-qualify every capsule and close evidence-laundering/duplication paths.
    ///
    /// The verifier requires a strict one-to-one relationship:
    ///
    /// - every capsule must correspond to exactly one reserved evidence item;
    /// - every reserved evidence item must correspond to exactly one verified capsule;
    /// - capsule digests and run IDs must be unique inside the bundle.
    ///
    /// Generic/non-reserved evidence remains allowed and is not upgraded by this method.
    pub fn verify(&self) -> Result<VerifiedGenerativityBundle, PersistedEvidenceError> {
        if self.schema != PERSISTED_GENERATIVITY_BUNDLE_SCHEMA {
            return Err(PersistedEvidenceError::UnsupportedBundleSchema(
                self.schema.clone(),
            ));
        }
        self.assessment.validate()?;
        if self.capsules.len() > MAX_CAPSULES_PER_BUNDLE {
            return Err(PersistedEvidenceError::TooManyCapsules {
                count: self.capsules.len(),
                max: MAX_CAPSULES_PER_BUNDLE,
            });
        }

        let mut seen_digests = HashSet::new();
        let mut seen_run_ids = HashSet::new();
        let mut verified_capsules = Vec::with_capacity(self.capsules.len());
        let mut expected_bindings: HashMap<EvidenceBindingKey, usize> = HashMap::new();

        for capsule in &self.capsules {
            let verified = capsule.verify()?;
            let digest = verified.envelope().envelope_digest().to_string();
            if !seen_digests.insert(digest.clone()) {
                return Err(PersistedEvidenceError::DuplicateCapsuleDigest(digest));
            }
            let run_id = verified.envelope().run_id().to_string();
            if !seen_run_ids.insert(run_id.clone()) {
                return Err(PersistedEvidenceError::DuplicateRunId(run_id));
            }
            let key = verified.expected_binding();
            expected_bindings.insert(key, 0);
            verified_capsules.push(verified);
        }

        for evidence in &self.assessment.evidence {
            if !evidence.kind.starts_with(RESERVED_KIND_PREFIX) {
                continue;
            }
            if evidence.kind != QUALIFIED_KIND && evidence.kind != VIOLATED_KIND {
                return Err(PersistedEvidenceError::UnsupportedReservedEvidenceKind(
                    evidence.kind.clone(),
                ));
            }
            let reference = evidence.reference.clone().ok_or_else(|| {
                PersistedEvidenceError::UnbackedReservedEvidence {
                    evidence_id: evidence.evidence_id.clone(),
                }
            })?;
            let key = EvidenceBindingKey {
                evidence_id: evidence.evidence_id.clone(),
                kind: evidence.kind.clone(),
                reference,
            };
            let count = expected_bindings.get_mut(&key).ok_or_else(|| {
                PersistedEvidenceError::UnbackedReservedEvidence {
                    evidence_id: evidence.evidence_id.clone(),
                }
            })?;
            *count += 1;
            if *count > 1 {
                return Err(PersistedEvidenceError::DuplicateEvidenceBinding {
                    evidence_id: evidence.evidence_id.clone(),
                });
            }
        }

        for (binding, count) in &expected_bindings {
            if *count == 0 {
                return Err(PersistedEvidenceError::UnreferencedCapsule {
                    evidence_id: binding.evidence_id.clone(),
                });
            }
        }

        Ok(VerifiedGenerativityBundle {
            assessment: self.assessment.clone(),
            capsules: verified_capsules,
        })
    }
}

/// Non-deserializable verified view. No score, confidence boost, or action authority is
/// derived from the number of qualified capsules.
#[derive(Debug, Clone, Serialize)]
pub struct VerifiedGenerativityBundle {
    assessment: GenerativityAssessment,
    capsules: Vec<VerifiedEvidenceCapsule>,
}

impl VerifiedGenerativityBundle {
    pub fn assessment(&self) -> &GenerativityAssessment {
        &self.assessment
    }

    pub fn capsules(&self) -> &[VerifiedEvidenceCapsule] {
        &self.capsules
    }

    pub fn qualified_capsule_count(&self) -> usize {
        self.capsules
            .iter()
            .filter(|capsule| capsule.envelope().integrity_satisfied())
            .count()
    }

    pub fn violated_capsule_count(&self) -> usize {
        self.capsules.len() - self.qualified_capsule_count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EvidenceBindingKey {
    evidence_id: String,
    kind: String,
    reference: String,
}

#[derive(Debug)]
pub enum PersistedEvidenceError {
    UnsupportedCapsuleSchema(String),
    UnsupportedBundleSchema(String),
    InvalidClaimedDigest,
    EnvelopeDigestMismatch { claimed: String, recomputed: String },
    TooManyCapsules { count: usize, max: usize },
    TooManyProvenanceRefs { count: usize, max: usize },
    InvalidProvenanceField(&'static str),
    DuplicateCapsuleDigest(String),
    DuplicateRunId(String),
    UnsupportedReservedEvidenceKind(String),
    UnbackedReservedEvidence { evidence_id: String },
    DuplicateEvidenceBinding { evidence_id: String },
    UnreferencedCapsule { evidence_id: String },
    Envelope(EvidenceEnvelopeError),
    Assessment(GenerativityValidationError),
}

impl From<EvidenceEnvelopeError> for PersistedEvidenceError {
    fn from(value: EvidenceEnvelopeError) -> Self {
        Self::Envelope(value)
    }
}

impl From<GenerativityValidationError> for PersistedEvidenceError {
    fn from(value: GenerativityValidationError) -> Self {
        Self::Assessment(value)
    }
}

impl std::fmt::Display for PersistedEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedCapsuleSchema(schema) => {
                write!(f, "unsupported persisted evidence capsule schema: {schema}")
            }
            Self::UnsupportedBundleSchema(schema) => {
                write!(f, "unsupported persisted generativity bundle schema: {schema}")
            }
            Self::InvalidClaimedDigest => {
                write!(f, "claimed envelope digest must be lowercase 64-hex BLAKE3")
            }
            Self::EnvelopeDigestMismatch { claimed, recomputed } => write!(
                f,
                "persisted evidence digest mismatch: claimed={claimed}, recomputed={recomputed}"
            ),
            Self::TooManyCapsules { count, max } => {
                write!(f, "persisted bundle has {count} capsules; maximum is {max}")
            }
            Self::TooManyProvenanceRefs { count, max } => {
                write!(f, "capsule has {count} provenance references; maximum is {max}")
            }
            Self::InvalidProvenanceField(field) => {
                write!(f, "invalid or unbounded provenance field: {field}")
            }
            Self::DuplicateCapsuleDigest(digest) => {
                write!(f, "duplicate evidence capsule digest: {digest}")
            }
            Self::DuplicateRunId(run_id) => write!(f, "duplicate evidence run_id: {run_id}"),
            Self::UnsupportedReservedEvidenceKind(kind) => {
                write!(f, "unsupported reserved evidence-plane kind: {kind}")
            }
            Self::UnbackedReservedEvidence { evidence_id } => write!(
                f,
                "reserved evidence-plane reference has no verified backing capsule: {evidence_id}"
            ),
            Self::DuplicateEvidenceBinding { evidence_id } => write!(
                f,
                "one evidence capsule is bound more than once in the assessment: {evidence_id}"
            ),
            Self::UnreferencedCapsule { evidence_id } => write!(
                f,
                "verified evidence capsule is not referenced by the assessment: {evidence_id}"
            ),
            Self::Envelope(error) => write!(f, "evidence envelope verification failed: {error}"),
            Self::Assessment(error) => write!(f, "generativity assessment is invalid: {error}"),
        }
    }
}

impl std::error::Error for PersistedEvidenceError {}

fn validate_digest(digest: &str) -> Result<(), PersistedEvidenceError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(PersistedEvidenceError::InvalidClaimedDigest);
    }
    Ok(())
}

fn validate_provenance(
    provenance: &[ProvenanceReference],
) -> Result<(), PersistedEvidenceError> {
    if provenance.len() > MAX_PROVENANCE_REFS {
        return Err(PersistedEvidenceError::TooManyProvenanceRefs {
            count: provenance.len(),
            max: MAX_PROVENANCE_REFS,
        });
    }
    for item in provenance {
        if item.kind.trim().is_empty() || item.kind.len() > MAX_PROVENANCE_KIND_LEN {
            return Err(PersistedEvidenceError::InvalidProvenanceField("kind"));
        }
        if item.reference.trim().is_empty()
            || item.reference.len() > MAX_PROVENANCE_REFERENCE_LEN
        {
            return Err(PersistedEvidenceError::InvalidProvenanceField(
                "reference",
            ));
        }
        if item
            .note
            .as_ref()
            .is_some_and(|note| note.len() > MAX_PROVENANCE_NOTE_LEN)
        {
            return Err(PersistedEvidenceError::InvalidProvenanceField("note"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

    use super::*;
    use crate::exploration::generativity::{
        GenerativityAssessment, GenerativityEstimate, GenerativityEvidence,
        GenerativityVector,
    };

    fn run(run_id: &str, calls: f64) -> RunEvidence {
        let mut declared = BTreeMap::new();
        declared.insert("mechanism_calls".into(), Expectation::MustBePositive);
        let mut measured = EvidenceCounters::new();
        measured.record("mechanism_calls", calls);
        RunEvidence::new(RunId::new(run_id), &("mode", "active"), declared, measured)
    }

    fn vector() -> GenerativityVector {
        let estimate = || GenerativityEstimate::new(0.5, 0.7).unwrap();
        GenerativityVector {
            immediate_utility: estimate(),
            epistemic_gain: estimate(),
            option_value: estimate(),
            diversity: estimate(),
            capability_gain: estimate(),
            diffusion: estimate(),
            commons_gain: estimate(),
            regeneration: estimate(),
            dependency_risk: estimate(),
            concentration_risk: estimate(),
            irreversibility_risk: estimate(),
        }
    }

    fn bound_bundle(run_id: &str) -> PersistedGenerativityBundle {
        let run = run(run_id, 4.0);
        let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
        let capsule = PersistedEvidenceCapsule::from_run(run, vec![]).unwrap();
        let mut assessment = GenerativityAssessment::new("subject", "context", vector());
        envelope
            .bind_qualified(&mut assessment, Some("qualified measurement".into()))
            .unwrap();
        PersistedGenerativityBundle::new(assessment, vec![capsule])
    }

    #[test]
    fn serialized_bundle_must_reverify_after_round_trip() {
        let bundle = bound_bundle("run:1");
        let json = serde_json::to_string(&bundle).unwrap();
        let restored: PersistedGenerativityBundle = serde_json::from_str(&json).unwrap();
        let verified = restored.verify().unwrap();
        assert_eq!(verified.qualified_capsule_count(), 1);
        assert_eq!(verified.violated_capsule_count(), 0);
    }

    #[test]
    fn rejects_tampered_claimed_digest() {
        let mut bundle = bound_bundle("run:1");
        bundle.capsules[0].claimed_envelope_digest = "0".repeat(64);
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::EnvelopeDigestMismatch { .. })
        ));
    }

    #[test]
    fn rejects_mutated_embedded_run_with_stale_digest() {
        let mut bundle = bound_bundle("run:1");
        bundle.capsules[0]
            .run
            .measured
            .record("mechanism_calls", 5.0);
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::EnvelopeDigestMismatch { .. })
        ));
    }

    #[test]
    fn rejects_forged_reserved_qualified_string_without_capsule() {
        let mut assessment = GenerativityAssessment::new("subject", "context", vector());
        assessment.evidence.push(GenerativityEvidence {
            evidence_id: "evidence-plane:fake".into(),
            kind: QUALIFIED_KIND.into(),
            reference: Some(format!("blake3:{}", "a".repeat(64))),
            note: None,
        });
        let bundle = PersistedGenerativityBundle::new(assessment, vec![]);
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::UnbackedReservedEvidence { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_capsules() {
        let mut bundle = bound_bundle("run:1");
        bundle.capsules.push(bundle.capsules[0].clone());
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::DuplicateCapsuleDigest(_))
        ));
    }

    #[test]
    fn rejects_duplicate_assessment_binding() {
        let mut bundle = bound_bundle("run:1");
        bundle
            .assessment
            .evidence
            .push(bundle.assessment.evidence[0].clone());
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::DuplicateEvidenceBinding { .. })
        ));
    }

    #[test]
    fn rejects_orphan_capsule() {
        let mut bundle = bound_bundle("run:1");
        bundle.assessment.evidence.clear();
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::UnreferencedCapsule { .. })
        ));
    }

    #[test]
    fn failed_run_cannot_be_laundered_as_qualified() {
        let failed = run("failed", 0.0);
        let capsule = PersistedEvidenceCapsule::from_run(failed, vec![]).unwrap();
        let mut assessment = GenerativityAssessment::new("subject", "context", vector());
        assessment.evidence.push(GenerativityEvidence {
            evidence_id: "evidence-plane:failed".into(),
            kind: QUALIFIED_KIND.into(),
            reference: Some(format!(
                "blake3:{}",
                capsule.claimed_envelope_digest()
            )),
            note: None,
        });
        let bundle = PersistedGenerativityBundle::new(assessment, vec![capsule]);
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::UnbackedReservedEvidence { .. })
        ));
    }

    #[test]
    fn failed_run_can_be_preserved_as_negative_evidence() {
        let failed = run("failed", 0.0);
        let envelope = EvidencePlaneEnvelope::from_run(&failed).unwrap();
        let capsule = PersistedEvidenceCapsule::from_run(failed, vec![]).unwrap();
        let mut assessment = GenerativityAssessment::new("subject", "context", vector());
        envelope.bind_observation(&mut assessment, Some("negative result".into()));
        let verified = PersistedGenerativityBundle::new(assessment, vec![capsule])
            .verify()
            .unwrap();
        assert_eq!(verified.qualified_capsule_count(), 0);
        assert_eq!(verified.violated_capsule_count(), 1);
    }

    #[test]
    fn rejects_duplicate_logical_run_id_even_with_distinct_digest() {
        let run_a = run("same-id", 4.0);
        let run_b = run("same-id", 5.0);
        let envelope_a = EvidencePlaneEnvelope::from_run(&run_a).unwrap();
        let envelope_b = EvidencePlaneEnvelope::from_run(&run_b).unwrap();
        let capsule_a = PersistedEvidenceCapsule::from_run(run_a, vec![]).unwrap();
        let capsule_b = PersistedEvidenceCapsule::from_run(run_b, vec![]).unwrap();
        let mut assessment = GenerativityAssessment::new("subject", "context", vector());
        envelope_a.bind_qualified(&mut assessment, None).unwrap();
        envelope_b.bind_qualified(&mut assessment, None).unwrap();
        let bundle = PersistedGenerativityBundle::new(assessment, vec![capsule_a, capsule_b]);
        assert!(matches!(
            bundle.verify(),
            Err(PersistedEvidenceError::DuplicateRunId(id)) if id == "same-id"
        ));
    }

    #[test]
    fn provenance_metadata_does_not_change_experiment_digest() {
        let run = run("run:1", 4.0);
        let a = PersistedEvidenceCapsule::from_run(run.clone(), vec![]).unwrap();
        let b = PersistedEvidenceCapsule::from_run(
            run,
            vec![ProvenanceReference {
                kind: "signature-ref".into(),
                reference: "xenia:signature:123".into(),
                note: Some("not automatically authenticated here".into()),
            }],
        )
        .unwrap();
        assert_eq!(a.claimed_envelope_digest(), b.claimed_envelope_digest());
    }
}
