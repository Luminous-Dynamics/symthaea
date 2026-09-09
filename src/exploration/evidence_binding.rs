// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Content-addressed binding between Symthaea's evidence plane and generativity.
//!
//! This bridge deliberately does **not** turn evidence into action authority. It provides
//! a deterministic BLAKE3 content identity for one `RunEvidence` record after
//! independently re-checking the declared/measured integrity contract.
//!
//! The digest is content integrity, not authentication: it proves that committed content
//! is unchanged, not who produced it. Signatures, trusted execution, remote attestation,
//! or Mycelix provenance remain separate layers.
//!
//! The evidence plane's existing `config_hash` is explicitly non-cryptographic. The
//! envelope commits that stored metadata string but does not upgrade it into exact or
//! security-sensitive configuration identity; `RunEvidence` does not retain the original
//! configuration object.

#![deny(unsafe_code)]

use std::collections::HashMap;

use serde::Serialize;
use symthaea_evidence_plane::{
    check_integrity, Expectation, FailedExpectation, RunEvidence,
};

use super::generativity::{GenerativityAssessment, GenerativityEvidence};

/// Versioned canonicalization contract used before BLAKE3 hashing.
pub const EVIDENCE_ENVELOPE_SCHEMA: &str = "symthaea-evidence-plane-envelope-v1";

/// A constructor-qualified, content-addressed summary of one evidence-plane run.
///
/// Fields are intentionally private and this type intentionally does not implement
/// `Deserialize`. Safe callers cannot manufacture a "qualified" envelope from arbitrary
/// wire data and then call [`Self::bind_qualified`]; construction must pass through
/// [`Self::from_run`], which recomputes the evidence-plane integrity contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidencePlaneEnvelope {
    schema: String,
    run_id: String,
    config_hash_metadata: String,
    integrity_satisfied: bool,
    violation_count: usize,
    /// Lowercase hex BLAKE3 digest over the canonicalized `RunEvidence` record.
    envelope_digest: String,
}

impl EvidencePlaneEnvelope {
    /// Recompute the evidence-plane integrity result, reject inconsistent/non-finite
    /// records, and derive a deterministic content identity.
    pub fn from_run(run: &RunEvidence) -> Result<Self, EvidenceEnvelopeError> {
        if run.run_id.0.trim().is_empty() {
            return Err(EvidenceEnvelopeError::EmptyRunId);
        }

        validate_finite_evidence(run)?;

        let declared: HashMap<String, Expectation> = run
            .declared
            .iter()
            .map(|(name, expectation)| (name.clone(), *expectation))
            .collect();
        let recomputed = check_integrity(&declared, &run.measured);
        let (integrity_satisfied, mut recomputed_violations) = match recomputed {
            Ok(()) => (true, Vec::new()),
            Err(violation) => (false, violation.failures),
        };

        // RunEvidence is serializable/public and can be mutated after construction. Never
        // trust its cached boolean/violations without comparing them to a fresh check.
        let mut recorded_violations = run.violations.clone();
        sort_failures(&mut recomputed_violations);
        sort_failures(&mut recorded_violations);
        if run.satisfied != integrity_satisfied || recorded_violations != recomputed_violations {
            return Err(EvidenceEnvelopeError::CachedIntegrityMismatch {
                recorded_satisfied: run.satisfied,
                recomputed_satisfied: integrity_satisfied,
            });
        }

        let envelope_digest = canonical_digest(run, integrity_satisfied, &recomputed_violations);

        Ok(Self {
            schema: EVIDENCE_ENVELOPE_SCHEMA.to_string(),
            run_id: run.run_id.0.clone(),
            config_hash_metadata: run.config_hash.clone(),
            integrity_satisfied,
            violation_count: recomputed_violations.len(),
            envelope_digest,
        })
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Non-cryptographic metadata copied from `RunEvidence`; never use as authority.
    pub fn config_hash_metadata(&self) -> &str {
        &self.config_hash_metadata
    }

    pub fn integrity_satisfied(&self) -> bool {
        self.integrity_satisfied
    }

    pub fn violation_count(&self) -> usize {
        self.violation_count
    }

    pub fn envelope_digest(&self) -> &str {
        &self.envelope_digest
    }

    /// Attach this envelope as explicit evidence, including failed-integrity runs.
    ///
    /// Failed runs are useful negative evidence, but their kind is visibly different so
    /// consumers can require the qualified kind when that distinction matters.
    pub fn bind_observation(
        &self,
        assessment: &mut GenerativityAssessment,
        note: Option<String>,
    ) {
        assessment.evidence.push(GenerativityEvidence {
            evidence_id: format!("evidence-plane:{}", self.run_id),
            kind: if self.integrity_satisfied {
                "symthaea-evidence-plane/qualified-v1".into()
            } else {
                "symthaea-evidence-plane/violated-v1".into()
            },
            reference: Some(format!("blake3:{}", self.envelope_digest)),
            note,
        });
    }

    /// Attach only if the declared/measured evidence-plane contract was satisfied.
    pub fn bind_qualified(
        &self,
        assessment: &mut GenerativityAssessment,
        note: Option<String>,
    ) -> Result<(), EvidenceEnvelopeError> {
        if !self.integrity_satisfied {
            return Err(EvidenceEnvelopeError::IntegrityNotSatisfied {
                violations: self.violation_count,
            });
        }
        self.bind_observation(assessment, note);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceEnvelopeError {
    EmptyRunId,
    NonFiniteMeasurement { name: String, value: f64 },
    NonFiniteExpectation { name: String, value: f64 },
    CachedIntegrityMismatch {
        recorded_satisfied: bool,
        recomputed_satisfied: bool,
    },
    IntegrityNotSatisfied { violations: usize },
}

impl std::fmt::Display for EvidenceEnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRunId => write!(f, "evidence-plane run_id must not be empty"),
            Self::NonFiniteMeasurement { name, value } => {
                write!(f, "evidence measurement '{name}' must be finite, got {value}")
            }
            Self::NonFiniteExpectation { name, value } => {
                write!(f, "evidence expectation '{name}' must be finite, got {value}")
            }
            Self::CachedIntegrityMismatch {
                recorded_satisfied,
                recomputed_satisfied,
            } => write!(
                f,
                "cached RunEvidence integrity disagrees with recomputation: recorded={recorded_satisfied}, recomputed={recomputed_satisfied}"
            ),
            Self::IntegrityNotSatisfied { violations } => write!(
                f,
                "evidence-plane run is not qualified: {violations} integrity violation(s)"
            ),
        }
    }
}

impl std::error::Error for EvidenceEnvelopeError {}

fn validate_finite_evidence(run: &RunEvidence) -> Result<(), EvidenceEnvelopeError> {
    for (name, value) in run.measured.iter() {
        if !value.is_finite() {
            return Err(EvidenceEnvelopeError::NonFiniteMeasurement {
                name: name.clone(),
                value: *value,
            });
        }
    }

    for (name, expectation) in &run.declared {
        let threshold = match expectation {
            Expectation::MustExceed(value) | Expectation::MustBeBelow(value) => Some(*value),
            Expectation::MustBeZero | Expectation::MustBePositive => None,
        };
        if let Some(value) = threshold {
            if !value.is_finite() {
                return Err(EvidenceEnvelopeError::NonFiniteExpectation {
                    name: name.clone(),
                    value,
                });
            }
        }
    }
    Ok(())
}

fn sort_failures(failures: &mut [FailedExpectation]) {
    failures.sort_by(|left, right| left.name.cmp(&right.name));
}

/// Canonical binary encoding built explicitly rather than relying on serde/map ordering.
/// Every variable-length field is length-prefixed and all floats are committed by their
/// exact IEEE-754 bit pattern after non-finite values have been rejected.
fn canonical_digest(
    run: &RunEvidence,
    integrity_satisfied: bool,
    violations: &[FailedExpectation],
) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, EVIDENCE_ENVELOPE_SCHEMA);
    put_str(&mut hasher, &run.run_id.0);
    put_str(&mut hasher, &run.config_hash);

    put_u64(&mut hasher, run.declared.len() as u64);
    for (name, expectation) in &run.declared {
        put_str(&mut hasher, name);
        put_expectation(&mut hasher, *expectation);
    }

    let mut measured: Vec<(&String, &f64)> = run.measured.iter().collect();
    measured.sort_by(|(left, _), (right, _)| left.cmp(right));
    put_u64(&mut hasher, measured.len() as u64);
    for (name, value) in measured {
        put_str(&mut hasher, name);
        put_f64(&mut hasher, *value);
    }

    hasher.update(&[u8::from(integrity_satisfied)]);
    put_u64(&mut hasher, violations.len() as u64);
    for failure in violations {
        put_str(&mut hasher, &failure.name);
        put_expectation(&mut hasher, failure.expectation);
        put_f64(&mut hasher, failure.measured);
    }

    hasher.finalize().to_hex().to_string()
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn put_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn put_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn put_expectation(hasher: &mut blake3::Hasher, expectation: Expectation) {
    match expectation {
        Expectation::MustBeZero => {
            hasher.update(&[0]);
        }
        Expectation::MustBePositive => {
            hasher.update(&[1]);
        }
        Expectation::MustExceed(value) => {
            hasher.update(&[2]);
            put_f64(hasher, value);
        }
        Expectation::MustBeBelow(value) => {
            hasher.update(&[3]);
            put_f64(hasher, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

    use super::*;
    use crate::exploration::generativity::{GenerativityEstimate, GenerativityVector};

    fn run_with_order(reverse: bool) -> RunEvidence {
        let mut declared = BTreeMap::new();
        declared.insert("mechanism_calls".into(), Expectation::MustBePositive);
        declared.insert("forbidden_calls".into(), Expectation::MustBeZero);

        let mut measured = EvidenceCounters::new();
        if reverse {
            measured.record("forbidden_calls", 0.0);
            measured.record("mechanism_calls", 4.0);
        } else {
            measured.record("mechanism_calls", 4.0);
            measured.record("forbidden_calls", 0.0);
        }

        RunEvidence::new(
            RunId::new("generativity:test:run"),
            &("mode", "active"),
            declared,
            measured,
        )
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

    #[test]
    fn digest_is_stable_across_measurement_insertion_order() {
        let a = EvidencePlaneEnvelope::from_run(&run_with_order(false)).unwrap();
        let b = EvidencePlaneEnvelope::from_run(&run_with_order(true)).unwrap();
        assert_eq!(a.envelope_digest(), b.envelope_digest());
        assert!(a.integrity_satisfied());
    }

    #[test]
    fn digest_changes_when_measurement_changes() {
        let a = EvidencePlaneEnvelope::from_run(&run_with_order(false)).unwrap();
        let mut changed = run_with_order(false);
        changed.measured.record("mechanism_calls", 5.0);
        let b = EvidencePlaneEnvelope::from_run(&changed).unwrap();
        assert_ne!(a.envelope_digest(), b.envelope_digest());
    }

    #[test]
    fn rejects_tampered_cached_integrity_flag() {
        let mut run = run_with_order(false);
        run.satisfied = false;
        let error = EvidencePlaneEnvelope::from_run(&run).unwrap_err();
        assert!(matches!(
            error,
            EvidenceEnvelopeError::CachedIntegrityMismatch { .. }
        ));
    }

    #[test]
    fn rejects_nonfinite_undeclared_measurement() {
        let mut run = run_with_order(false);
        run.measured.record("unrelated", f64::NAN);
        assert!(matches!(
            EvidencePlaneEnvelope::from_run(&run),
            Err(EvidenceEnvelopeError::NonFiniteMeasurement { .. })
        ));
    }

    #[test]
    fn failed_run_is_addressable_but_not_qualified() {
        let mut declared = BTreeMap::new();
        declared.insert("mechanism_calls".into(), Expectation::MustBePositive);
        let measured = EvidenceCounters::new();
        let run = RunEvidence::new(
            RunId::new("failed"),
            &("mode", "active"),
            declared,
            measured,
        );
        let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
        assert!(!envelope.integrity_satisfied());
        assert_eq!(envelope.violation_count(), 1);

        let mut assessment = GenerativityAssessment::new("subject", "context", vector());
        assert!(matches!(
            envelope.bind_qualified(&mut assessment, None),
            Err(EvidenceEnvelopeError::IntegrityNotSatisfied { violations: 1 })
        ));
        assert!(assessment.evidence.is_empty());

        envelope.bind_observation(&mut assessment, Some("negative evidence".into()));
        assert_eq!(assessment.evidence.len(), 1);
        assert_eq!(
            assessment.evidence[0].kind,
            "symthaea-evidence-plane/violated-v1"
        );
    }

    #[test]
    fn qualified_run_binds_blake3_content_identity() {
        let envelope = EvidencePlaneEnvelope::from_run(&run_with_order(false)).unwrap();
        let mut assessment = GenerativityAssessment::new("subject", "context", vector());
        envelope
            .bind_qualified(&mut assessment, Some("mechanism integrity passed".into()))
            .unwrap();
        let evidence = assessment.evidence.last().unwrap();
        assert_eq!(evidence.kind, "symthaea-evidence-plane/qualified-v1");
        let expected = format!("blake3:{}", envelope.envelope_digest());
        assert_eq!(evidence.reference.as_deref(), Some(expected.as_str()));
        assert!(assessment.validate().is_ok());
    }

    #[test]
    fn envelope_exports_config_hash_only_as_metadata() {
        let envelope = EvidencePlaneEnvelope::from_run(&run_with_order(false)).unwrap();
        assert_eq!(envelope.schema(), EVIDENCE_ENVELOPE_SCHEMA);
        assert_eq!(envelope.run_id(), "generativity:test:run");
        assert!(!envelope.config_hash_metadata().is_empty());
    }
}
