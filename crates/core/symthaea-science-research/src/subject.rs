// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::{FramedDigest, ResearchId, Sha256Digest};
use serde::{Deserialize, Serialize};

pub const SCIENTIFIC_SUBJECT_SCHEMA: &str = "symthaea.science-subject.v1";
const SUBJECT_DOMAIN: &str = "symthaea.science-subject.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SubjectKind {
    EmpiricalHypothesis,
    CausalHypothesis,
    MathematicalConjecture,
    FormalTheorem,
    PhysicalModel,
    StatisticalModel,
    Dataset,
    ExperimentalProtocol,
    NumericalReproductionTarget,
    ScientificMeasurement,
    BenchmarkTask,
    LiteratureClaim,
    Other,
}

/// Draft form. Qualification identities and evidence are intentionally absent
/// from the stable subject identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificSubject {
    pub subject_id: ResearchId,
    pub kind: SubjectKind,
    pub content_sha256: Sha256Digest,
    pub source_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubjectIssue {
    EmptySourceRevision,
}

impl ScientificSubject {
    pub fn validate(&self) -> Vec<SubjectIssue> {
        let mut issues = Vec::new();
        if self.source_revision.trim().is_empty() {
            issues.push(SubjectIssue::EmptySourceRevision);
        }
        issues
    }

    pub fn freeze(self) -> Result<FrozenScientificSubject, Vec<SubjectIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let mut digest = FramedDigest::new(SUBJECT_DOMAIN);
        digest.text(SCIENTIFIC_SUBJECT_SCHEMA);
        digest.text(self.subject_id.as_str());
        digest.text(subject_kind_tag(self.kind));
        digest.text(self.content_sha256.as_str());
        digest.text(&self.source_revision);
        Ok(FrozenScientificSubject {
            subject: self,
            subject_sha256: digest.digest(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenScientificSubject {
    subject: ScientificSubject,
    subject_sha256: Sha256Digest,
}

impl FrozenScientificSubject {
    pub fn subject(&self) -> &ScientificSubject {
        &self.subject
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }
}

fn subject_kind_tag(kind: SubjectKind) -> &'static str {
    match kind {
        SubjectKind::EmpiricalHypothesis => "empirical-hypothesis",
        SubjectKind::CausalHypothesis => "causal-hypothesis",
        SubjectKind::MathematicalConjecture => "mathematical-conjecture",
        SubjectKind::FormalTheorem => "formal-theorem",
        SubjectKind::PhysicalModel => "physical-model",
        SubjectKind::StatisticalModel => "statistical-model",
        SubjectKind::Dataset => "dataset",
        SubjectKind::ExperimentalProtocol => "experimental-protocol",
        SubjectKind::NumericalReproductionTarget => "numerical-reproduction-target",
        SubjectKind::ScientificMeasurement => "scientific-measurement",
        SubjectKind::BenchmarkTask => "benchmark-task",
        SubjectKind::LiteratureClaim => "literature-claim",
        SubjectKind::Other => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> ScientificSubject {
        ScientificSubject {
            subject_id: ResearchId::parse("SCI-TEST-001").unwrap(),
            kind: SubjectKind::EmpiricalHypothesis,
            content_sha256: Sha256Digest::of_bytes(b"hypothesis"),
            source_revision: "source-v1".into(),
        }
    }

    #[test]
    fn stable_subject_identity_is_deterministic() {
        let a = subject().freeze().unwrap();
        let b = subject().freeze().unwrap();
        assert_eq!(a.subject_sha256(), b.subject_sha256());
    }

    #[test]
    fn subject_identity_changes_when_subject_changes() {
        let a = subject().freeze().unwrap();
        let mut changed = subject();
        changed.content_sha256 = Sha256Digest::of_bytes(b"different-hypothesis");
        let changed = changed.freeze().unwrap();
        assert_ne!(a.subject_sha256(), changed.subject_sha256());
    }

    #[test]
    fn empty_source_revision_fails_closed() {
        let mut subject = subject();
        subject.source_revision.clear();
        assert!(subject.freeze().is_err());
    }
}
