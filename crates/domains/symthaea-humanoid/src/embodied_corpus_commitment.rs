// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Collision-resistant commitment to the complete embodied-control evidence corpus.
//!
//! `EmbodiedControlCertificate::scenario_fingerprint` is a compact legacy
//! diagnostic fingerprint. It intentionally remains unchanged here for wire and
//! compatibility stability, but it does not commit to every field that affects
//! certification. This module adds a separate BLAKE3 commitment over the exact
//! ordered case corpus and certification criteria so future authority/evidence
//! joins do not depend on the legacy 64-bit fingerprint.
//!
//! Encoding v1 is explicitly implementation-lineage scoped: domain-separated
//! BLAKE3 over length-prefixed `serde_json` bytes produced by the workspace-pinned
//! Rust serializer. It is not claimed as a universal cross-language canonical
//! JSON format. Any encoding change requires a new schema/domain string.

use serde::{Deserialize, Serialize};

use crate::embodied_certification::{EmbodiedCertificationCriteria, EmbodiedControlCase};

pub const EMBODIED_CONTROL_CORPUS_COMMITMENT_SCHEMA_VERSION: u32 = 1;
const DOMAIN: &[u8] = b"symthaea-humanoid/embodied-control-corpus/v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodiedControlCorpusCommitmentV1 {
    pub schema_version: u32,
    pub case_count: u32,
    pub digest: [u8; 32],
}

impl EmbodiedControlCorpusCommitmentV1 {
    pub fn validate(&self) -> bool {
        self.schema_version == EMBODIED_CONTROL_CORPUS_COMMITMENT_SCHEMA_VERSION
            && self.case_count > 0
            && self.digest != [0; 32]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbodiedControlCorpusCommitmentError {
    EmptyCorpus,
    TooManyCases,
    InvalidCase { index: usize },
    SerializationFailed,
}

impl std::fmt::Display for EmbodiedControlCorpusCommitmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCorpus => f.write_str("embodied-control corpus is empty"),
            Self::TooManyCases => f.write_str("embodied-control corpus exceeds v1 case count"),
            Self::InvalidCase { index } => {
                write!(f, "embodied-control case {index} is structurally invalid")
            }
            Self::SerializationFailed => {
                f.write_str("embodied-control corpus could not be serialized for commitment")
            }
        }
    }
}

impl std::error::Error for EmbodiedControlCorpusCommitmentError {}

/// Commit to the exact ordered evidence corpus and the exact criteria under
/// which that corpus is interpreted.
///
/// The complete serialized `EmbodiedControlCase` includes the full hierarchical
/// control report, dynamics fidelity, terrain uncertainty/freshness, contacts,
/// fall/recovery outcomes, and the subject fingerprint. Therefore changing any
/// serialized field changes the commitment even when a legacy diagnostic
/// fingerprint omits that field.
pub fn commit_embodied_control_corpus(
    cases: &[EmbodiedControlCase],
    criteria: EmbodiedCertificationCriteria,
) -> Result<EmbodiedControlCorpusCommitmentV1, EmbodiedControlCorpusCommitmentError> {
    if cases.is_empty() {
        return Err(EmbodiedControlCorpusCommitmentError::EmptyCorpus);
    }
    let case_count = u32::try_from(cases.len())
        .map_err(|_| EmbodiedControlCorpusCommitmentError::TooManyCases)?;
    for (index, case) in cases.iter().enumerate() {
        if !case.validate() {
            return Err(EmbodiedControlCorpusCommitmentError::InvalidCase { index });
        }
    }

    let criteria_bytes = serde_json::to_vec(&criteria)
        .map_err(|_| EmbodiedControlCorpusCommitmentError::SerializationFailed)?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN);
    feed_bytes(&mut hasher, &case_count.to_le_bytes());
    feed_bytes(&mut hasher, &criteria_bytes);
    for case in cases {
        let bytes = serde_json::to_vec(case)
            .map_err(|_| EmbodiedControlCorpusCommitmentError::SerializationFailed)?;
        feed_bytes(&mut hasher, &bytes);
    }

    Ok(EmbodiedControlCorpusCommitmentV1 {
        schema_version: EMBODIED_CONTROL_CORPUS_COMMITMENT_SCHEMA_VERSION,
        case_count,
        digest: *hasher.finalize().as_bytes(),
    })
}

fn feed_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodied_certification::certify_embodied_control;
    use crate::full_dynamics::DynamicsFidelity;
    use crate::hierarchical::HierarchicalHumanoidController;
    use crate::morphology::HumanoidMorphology;
    use crate::types::{HumanoidCommand, HumanoidState, HumanoidTask};

    fn case(id: &str) -> EmbodiedControlCase {
        let state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        let zero = HumanoidCommand::zero();
        let (_, report) = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21)
            .synthesize(HumanoidTask::Stand, &state, &zero, &zero, 1.0, 0.0);
        EmbodiedControlCase {
            scenario_id: id.to_string(),
            subject_fingerprint: 7,
            dynamics_fidelity: DynamicsFidelity::SolverDerived,
            active_contacts: 2,
            terrain_height_std_m: 0.01,
            terrain_evidence_age_s: 0.01,
            report,
            fell: false,
            recovered: false,
        }
    }

    fn criteria() -> EmbodiedCertificationCriteria {
        EmbodiedCertificationCriteria {
            require_upper_body_contact_case: false,
            ..EmbodiedCertificationCriteria::default()
        }
    }

    #[test]
    fn identical_corpus_and_policy_have_identical_commitment() {
        let cases = vec![case("a")];
        let first = commit_embodied_control_corpus(&cases, criteria()).unwrap();
        let second = commit_embodied_control_corpus(&cases, criteria()).unwrap();
        assert!(first.validate());
        assert_eq!(first, second);
    }

    #[test]
    fn terrain_age_mutation_changes_commitment_even_when_legacy_fingerprint_does_not() {
        let original = vec![case("a")];
        let mut changed = vec![case("a")];
        changed[0].terrain_evidence_age_s = 0.02;

        let legacy_original = certify_embodied_control(&original, criteria());
        let legacy_changed = certify_embodied_control(&changed, criteria());
        assert_eq!(
            legacy_original.scenario_fingerprint,
            legacy_changed.scenario_fingerprint,
            "documents the legacy diagnostic gap this commitment closes"
        );

        let committed_original = commit_embodied_control_corpus(&original, criteria()).unwrap();
        let committed_changed = commit_embodied_control_corpus(&changed, criteria()).unwrap();
        assert_ne!(committed_original.digest, committed_changed.digest);
    }

    #[test]
    fn dynamics_fidelity_mutation_changes_commitment() {
        let original = vec![case("a")];
        let mut changed = vec![case("a")];
        changed[0].dynamics_fidelity = DynamicsFidelity::ReducedOrder;
        assert_ne!(
            commit_embodied_control_corpus(&original, criteria())
                .unwrap()
                .digest,
            commit_embodied_control_corpus(&changed, criteria())
                .unwrap()
                .digest
        );
    }

    #[test]
    fn control_report_mutation_changes_commitment() {
        let original = vec![case("a")];
        let mut changed = vec![case("a")];
        changed[0].report.contact_dynamics_fallback =
            !changed[0].report.contact_dynamics_fallback;
        assert_ne!(
            commit_embodied_control_corpus(&original, criteria())
                .unwrap()
                .digest,
            commit_embodied_control_corpus(&changed, criteria())
                .unwrap()
                .digest
        );
    }

    #[test]
    fn certification_policy_mutation_changes_commitment() {
        let cases = vec![case("a")];
        let original = criteria();
        let mut changed = criteria();
        changed.maximum_terrain_evidence_age_s += 0.01;
        assert_ne!(
            commit_embodied_control_corpus(&cases, original).unwrap().digest,
            commit_embodied_control_corpus(&cases, changed).unwrap().digest
        );
    }

    #[test]
    fn invalid_case_fails_before_commitment() {
        let mut invalid = case("a");
        invalid.subject_fingerprint = 0;
        assert_eq!(
            commit_embodied_control_corpus(&[invalid], criteria()).unwrap_err(),
            EmbodiedControlCorpusCommitmentError::InvalidCase { index: 0 }
        );
    }
}
