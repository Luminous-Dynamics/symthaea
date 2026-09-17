// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic qualifier-recipe identity and witness binding.
//!
//! This layer never executes candidate code and never manufactures a governance
//! decision. It deterministically binds immutable recipe artifacts, candidate
//! PASS receipts, predecessor relation, and an optional external bootstrap
//! admission basis supplied by a higher-trust process.

use super::{
    ArtifactIdentity, QualificationDigest, QualifiedHeadReceipt, QualificationError, Writer,
};
use serde::{Deserialize, Serialize};
use std::fmt;

const RECIPE_DOMAIN: &[u8] = b"symthaea.qualification.recipe.v1\0";
const WITNESS_DOMAIN: &[u8] = b"symthaea.qualification.recipe-witness.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeArtifact {
    pub logical_name: String,
    pub identity: ArtifactIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualifierRecipe {
    pub revision: u32,
    pub artifacts: Vec<RecipeArtifact>,
}

impl QualifierRecipe {
    pub fn new(
        revision: u32,
        mut artifacts: Vec<RecipeArtifact>,
    ) -> Result<Self, RecipeWitnessError> {
        if artifacts.is_empty() {
            return Err(RecipeWitnessError::EmptyRecipe);
        }
        if artifacts.iter().any(|artifact| artifact.logical_name.is_empty()) {
            return Err(RecipeWitnessError::EmptyArtifactName);
        }

        artifacts.sort_by(|a, b| a.logical_name.cmp(&b.logical_name));
        if artifacts
            .windows(2)
            .any(|pair| pair[0].logical_name == pair[1].logical_name)
        {
            return Err(RecipeWitnessError::DuplicateArtifactName);
        }

        Ok(Self { revision, artifacts })
    }

    pub fn identity(&self) -> QualificationDigest {
        let mut w = Writer::new(RECIPE_DOMAIN);
        w.u32(self.revision);
        w.u32(self.artifacts.len() as u32);
        for artifact in &self.artifacts {
            w.str(&artifact.logical_name);
            w.artifact(artifact.identity);
        }
        w.finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecipeRelation {
    BaseIdentical,
    RecipeChanged,
    BootstrapNoPredecessor,
}

pub fn relation(
    candidate: QualificationDigest,
    predecessor: Option<QualificationDigest>,
) -> RecipeRelation {
    match predecessor {
        None => RecipeRelation::BootstrapNoPredecessor,
        Some(root) if root == candidate => RecipeRelation::BaseIdentical,
        Some(_) => RecipeRelation::RecipeChanged,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecipeWitnessDisposition {
    /// Candidate PASS is bound to an externally supplied bootstrap admission
    /// record. Trust in that record is a governance concern outside this kernel.
    BootstrapAdmissionBound,
    /// Candidate recipe is byte-identical to the already trusted predecessor.
    BaseIdenticalCandidatePass,
    /// Candidate execution may be useful conformance evidence, but cannot mint
    /// ordinary qualification authority until the new recipe is admitted.
    RecipeChangedConformanceOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualifierRecipeWitness {
    revision: u32,
    candidate_receipt_root: QualificationDigest,
    candidate_recipe_root: QualificationDigest,
    predecessor_recipe_root: Option<QualificationDigest>,
    relation: RecipeRelation,
    disposition: RecipeWitnessDisposition,
    /// Immutable external admission/review record for first bootstrap only.
    /// Presence binds the record; this kernel does not claim the record itself
    /// is trustworthy.
    bootstrap_admission_basis: Option<ArtifactIdentity>,
}

impl QualifierRecipeWitness {
    pub fn bind(
        candidate_receipt: &QualifiedHeadReceipt,
        candidate_recipe: &QualifierRecipe,
        predecessor_recipe_root: Option<QualificationDigest>,
        bootstrap_admission_basis: Option<ArtifactIdentity>,
    ) -> Result<Self, RecipeWitnessError> {
        let candidate_recipe_root = candidate_recipe.identity();
        let relation = relation(candidate_recipe_root, predecessor_recipe_root);

        let disposition = match relation {
            RecipeRelation::BootstrapNoPredecessor => {
                if bootstrap_admission_basis.is_none() {
                    return Err(RecipeWitnessError::MissingBootstrapAdmissionBasis);
                }
                RecipeWitnessDisposition::BootstrapAdmissionBound
            }
            RecipeRelation::BaseIdentical => {
                if bootstrap_admission_basis.is_some() {
                    return Err(RecipeWitnessError::UnexpectedBootstrapAdmissionBasis);
                }
                RecipeWitnessDisposition::BaseIdenticalCandidatePass
            }
            RecipeRelation::RecipeChanged => {
                if bootstrap_admission_basis.is_some() {
                    return Err(RecipeWitnessError::UnexpectedBootstrapAdmissionBasis);
                }
                RecipeWitnessDisposition::RecipeChangedConformanceOnly
            }
        };

        Ok(Self {
            revision: 1,
            candidate_receipt_root: candidate_receipt.identity(),
            candidate_recipe_root,
            predecessor_recipe_root,
            relation,
            disposition,
            bootstrap_admission_basis,
        })
    }


    pub fn relation(&self) -> RecipeRelation {
        self.relation
    }

    pub fn disposition(&self) -> RecipeWitnessDisposition {
        self.disposition
    }

    pub fn candidate_receipt_root(&self) -> QualificationDigest {
        self.candidate_receipt_root
    }

    pub fn candidate_recipe_root(&self) -> QualificationDigest {
        self.candidate_recipe_root
    }

    pub fn predecessor_recipe_root(&self) -> Option<QualificationDigest> {
        self.predecessor_recipe_root
    }

    pub fn bootstrap_admission_basis(&self) -> Option<ArtifactIdentity> {
        self.bootstrap_admission_basis
    }

    /// True only for dispositions that can participate in trusted authority
    /// after the external trust root for the bootstrap basis/predecessor is
    /// separately established.
    pub fn authority_eligible(&self) -> bool {
        matches!(
            self.disposition,
            RecipeWitnessDisposition::BootstrapAdmissionBound
                | RecipeWitnessDisposition::BaseIdenticalCandidatePass
        )
    }

    pub fn identity(&self) -> QualificationDigest {
        let mut w = Writer::new(WITNESS_DOMAIN);
        w.u32(self.revision);
        w.digest(self.candidate_receipt_root);
        w.digest(self.candidate_recipe_root);

        match self.predecessor_recipe_root {
            Some(root) => {
                w.u8(1);
                w.digest(root);
            }
            None => w.u8(0),
        }

        w.u8(relation_tag(self.relation));
        w.u8(disposition_tag(self.disposition));

        match self.bootstrap_admission_basis {
            Some(identity) => {
                w.u8(1);
                w.artifact(identity);
            }
            None => w.u8(0),
        }

        w.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecipeWitnessError {
    EmptyRecipe,
    EmptyArtifactName,
    DuplicateArtifactName,
    MissingBootstrapAdmissionBasis,
    UnexpectedBootstrapAdmissionBasis,
    CandidateReceipt(QualificationError),
}

impl fmt::Display for RecipeWitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RecipeWitnessError {}

fn relation_tag(value: RecipeRelation) -> u8 {
    match value {
        RecipeRelation::BaseIdentical => 0,
        RecipeRelation::RecipeChanged => 1,
        RecipeRelation::BootstrapNoPredecessor => 2,
    }
}

fn disposition_tag(value: RecipeWitnessDisposition) -> u8 {
    match value {
        RecipeWitnessDisposition::BootstrapAdmissionBound => 0,
        RecipeWitnessDisposition::BaseIdenticalCandidatePass => 1,
        RecipeWitnessDisposition::RecipeChangedConformanceOnly => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qualification::{
        ArtifactIdentity, JobDisposition, ObservedQualificationJob,
        ObservedQualificationStep, QualificationJobRequirement, QualificationLane,
        QualificationProfile, QualificationRunObservation,
    };

    const SUBJECT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn workflow(byte: u8) -> ArtifactIdentity {
        ArtifactIdentity::GitBlobSha1([byte; 20])
    }

    fn passing_receipt() -> QualifiedHeadReceipt {
        let profile = QualificationProfile::new(
            1,
            QualificationLane::FullExactHead,
            "Luminous-Dynamics/symthaea",
            ".github/workflows/qualify.yml",
            workflow(1),
            vec!["pull_request".into()],
            vec![QualificationJobRequirement::required("qualify")],
        )
        .unwrap();

        let observation = QualificationRunObservation {
            repository: "Luminous-Dynamics/symthaea".into(),
            workflow_run_id: 42,
            workflow_id: 7,
            workflow_path: ".github/workflows/qualify.yml".into(),
            workflow_definition: workflow(1),
            run_attempt: 1,
            event: "pull_request".into(),
            exact_head_sha: SUBJECT.into(),
            jobs: vec![ObservedQualificationJob {
                job_id: 9,
                name: "qualify".into(),
                disposition: JobDisposition::Passed,
                steps: vec![ObservedQualificationStep {
                    ordinal: 1,
                    name: "run".into(),
                    disposition: JobDisposition::Passed,
                }],
                provider_status: Some("completed".into()),
                provider_conclusion: Some("success".into()),
            }],
            materializer_revision: 1,
            head_ref: None,
            base_ref: None,
            provider_status: Some("completed".into()),
            provider_conclusion: Some("success".into()),
        };

        QualifiedHeadReceipt::try_new(&profile, SUBJECT, &observation).unwrap()
    }

    fn recipe(workflow_byte: u8) -> QualifierRecipe {
        QualifierRecipe::new(
            1,
            vec![RecipeArtifact {
                logical_name: "workflow".into(),
                identity: workflow(workflow_byte),
            }],
        )
        .unwrap()
    }

    #[test]
    fn bootstrap_requires_external_admission_basis() {
        let receipt = passing_receipt();
        let recipe = recipe(1);

        assert_eq!(
            QualifierRecipeWitness::bind(&receipt, &recipe, None, None),
            Err(RecipeWitnessError::MissingBootstrapAdmissionBasis)
        );

        let witness = QualifierRecipeWitness::bind(
            &receipt,
            &recipe,
            None,
            Some(ArtifactIdentity::Sha256([9; 32])),
        )
        .unwrap();
        assert_eq!(witness.relation, RecipeRelation::BootstrapNoPredecessor);
        assert_eq!(
            witness.disposition,
            RecipeWitnessDisposition::BootstrapAdmissionBound
        );
        assert!(witness.authority_eligible());
    }

    #[test]
    fn base_identical_candidate_pass_is_authority_eligible() {
        let receipt = passing_receipt();
        let recipe = recipe(1);
        let root = recipe.identity();

        let witness =
            QualifierRecipeWitness::bind(&receipt, &recipe, Some(root), None).unwrap();
        assert_eq!(witness.relation, RecipeRelation::BaseIdentical);
        assert_eq!(
            witness.disposition,
            RecipeWitnessDisposition::BaseIdenticalCandidatePass
        );
        assert!(witness.authority_eligible());
    }

    #[test]
    fn changed_recipe_is_conformance_only() {
        let receipt = passing_receipt();
        let old = recipe(1);
        let changed = recipe(2);

        let witness = QualifierRecipeWitness::bind(
            &receipt,
            &changed,
            Some(old.identity()),
            None,
        )
        .unwrap();
        assert_eq!(witness.relation, RecipeRelation::RecipeChanged);
        assert_eq!(
            witness.disposition,
            RecipeWitnessDisposition::RecipeChangedConformanceOnly
        );
        assert!(!witness.authority_eligible());
    }

    #[test]
    fn recipe_artifact_order_is_normalized() {
        let a = QualifierRecipe::new(
            1,
            vec![
                RecipeArtifact {
                    logical_name: "workflow".into(),
                    identity: workflow(1),
                },
                RecipeArtifact {
                    logical_name: "verifier".into(),
                    identity: ArtifactIdentity::Sha256([2; 32]),
                },
            ],
        )
        .unwrap();
        let b = QualifierRecipe::new(
            1,
            vec![
                RecipeArtifact {
                    logical_name: "verifier".into(),
                    identity: ArtifactIdentity::Sha256([2; 32]),
                },
                RecipeArtifact {
                    logical_name: "workflow".into(),
                    identity: workflow(1),
                },
            ],
        )
        .unwrap();

        assert_eq!(a.identity(), b.identity());
    }

    #[test]
    fn duplicate_recipe_artifact_name_fails_closed() {
        let result = QualifierRecipe::new(
            1,
            vec![
                RecipeArtifact {
                    logical_name: "workflow".into(),
                    identity: workflow(1),
                },
                RecipeArtifact {
                    logical_name: "workflow".into(),
                    identity: workflow(2),
                },
            ],
        );

        assert_eq!(result, Err(RecipeWitnessError::DuplicateArtifactName));
    }

    #[test]
    fn bootstrap_basis_changes_witness_identity() {
        let receipt = passing_receipt();
        let recipe = recipe(1);

        let a = QualifierRecipeWitness::bind(
            &receipt,
            &recipe,
            None,
            Some(ArtifactIdentity::Sha256([1; 32])),
        )
        .unwrap();
        let b = QualifierRecipeWitness::bind(
            &receipt,
            &recipe,
            None,
            Some(ArtifactIdentity::Sha256([2; 32])),
        )
        .unwrap();

        assert_ne!(a.identity(), b.identity());
    }
}
