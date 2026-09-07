// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical evidence binding for relativistic electronic-structure receipts.
//!
//! `matter_relativistic` defines which physics a local computational claim must
//! explicitly support. This module answers a different question: do the free-form
//! identities carried by that receipt resolve to a locally consistent bundle of
//! canonical external evidence references?
//!
//! The binding is intentionally reference-only. It does not fetch or re-hash an
//! artifact, authenticate an issuer, verify a timestamp, validate a solver's
//! numerical correctness, or upgrade a computational result into experiment or
//! independent-replication authority.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, DeclaredChronologyInterpretation, EvidenceReferenceInterpretation,
    EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError, ExternalEvidenceReference,
    Sha256Digest,
};

use super::matter_relativistic::{
    RelativisticAdmissionError, RelativisticElectronicStructureReceipt,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelativisticEvidenceBindingError {
    InvalidReceipt(RelativisticAdmissionError),
    ExternalEvidence(ExternalEvidenceError),
    SemanticSlotIdentityAlias {
        left_slot: &'static str,
        right_slot: &'static str,
    },
    SemanticSlotContentAlias {
        left_slot: &'static str,
        right_slot: &'static str,
    },
    DuplicateReferenceDatasetIdentity(String),
    DuplicateReferenceDatasetContent {
        first_id: String,
        second_id: String,
    },
    ReferenceDatasetAliasesExecutionArtifact(String),
    ReferenceDatasetAliasesCoreContent {
        reference_id: String,
        core_slot: &'static str,
    },
    MissingClaimedDate(String),
    DependencyPostdatesExecution {
        evidence_id: String,
        dependency_yyyymmdd: u32,
        execution_yyyymmdd: u32,
    },
    OutputPredatesExecution {
        output_yyyymmdd: u32,
        execution_yyyymmdd: u32,
    },
}

impl From<RelativisticAdmissionError> for RelativisticEvidenceBindingError {
    fn from(value: RelativisticAdmissionError) -> Self {
        Self::InvalidReceipt(value)
    }
}

impl From<ExternalEvidenceError> for RelativisticEvidenceBindingError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for RelativisticEvidenceBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReceipt(error) => write!(f, "relativistic receipt rejected: {error}"),
            Self::ExternalEvidence(error) => write!(f, "{error}"),
            Self::SemanticSlotIdentityAlias {
                left_slot,
                right_slot,
            } => write!(
                f,
                "relativistic semantic slots `{left_slot}` and `{right_slot}` use the same evidence identity"
            ),
            Self::SemanticSlotContentAlias {
                left_slot,
                right_slot,
            } => write!(
                f,
                "relativistic semantic slots `{left_slot}` and `{right_slot}` resolve to the same content identity"
            ),
            Self::DuplicateReferenceDatasetIdentity(id) => write!(
                f,
                "reference dataset evidence identity `{id}` occurs more than once"
            ),
            Self::DuplicateReferenceDatasetContent {
                first_id,
                second_id,
            } => write!(
                f,
                "reference datasets `{first_id}` and `{second_id}` resolve to the same content identity"
            ),
            Self::ReferenceDatasetAliasesExecutionArtifact(id) => write!(
                f,
                "reference dataset identity `{id}` aliases a solver execution/input/output/method artifact"
            ),
            Self::ReferenceDatasetAliasesCoreContent {
                reference_id,
                core_slot,
            } => write!(
                f,
                "reference dataset `{reference_id}` resolves to the same content identity as relativistic `{core_slot}`"
            ),
            Self::MissingClaimedDate(id) => {
                write!(f, "external evidence `{id}` has no claimed UTC date")
            }
            Self::DependencyPostdatesExecution {
                evidence_id,
                dependency_yyyymmdd,
                execution_yyyymmdd,
            } => write!(
                f,
                "relativistic dependency `{evidence_id}` date {dependency_yyyymmdd} postdates execution date {execution_yyyymmdd}"
            ),
            Self::OutputPredatesExecution {
                output_yyyymmdd,
                execution_yyyymmdd,
            } => write!(
                f,
                "relativistic output date {output_yyyymmdd} predates execution date {execution_yyyymmdd}"
            ),
        }
    }
}

impl std::error::Error for RelativisticEvidenceBindingError {}

/// One reference dataset bound to exact content and declared chronology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativisticReferenceDatasetBinding {
    pub evidence_id: String,
    pub content_sha256: Sha256Digest,
    pub claimed_date: ClaimedUtcDate,
}

/// Exact canonical references attached to one relativistic execution receipt.
///
/// The semantic slots are local to electronic-structure evidence while the
/// referenced artifacts remain generic evidence-plane `ArtifactContent` values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativisticExecutionReferenceBinding {
    pub execution_evidence_id: String,
    pub input_evidence_id: String,
    pub output_evidence_id: String,
    pub hamiltonian_evidence_id: String,
    pub basis_or_pseudopotential_evidence_id: String,
    pub execution_content_sha256: Sha256Digest,
    pub input_content_sha256: Sha256Digest,
    pub output_content_sha256: Sha256Digest,
    pub hamiltonian_content_sha256: Sha256Digest,
    pub basis_or_pseudopotential_content_sha256: Sha256Digest,
    pub execution_claimed_date: ClaimedUtcDate,
    pub input_claimed_date: ClaimedUtcDate,
    pub output_claimed_date: ClaimedUtcDate,
    pub hamiltonian_claimed_date: ClaimedUtcDate,
    pub basis_or_pseudopotential_claimed_date: ClaimedUtcDate,
    pub reference_datasets: Vec<RelativisticReferenceDatasetBinding>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

/// Bind all identities carried by a relativistic electronic-structure receipt to
/// canonical evidence references.
///
/// Required canonical roles:
///
/// - receipt id -> `SolverExecution`
/// - input/output/Hamiltonian/basis-ECP/reference datasets -> `ArtifactContent`
///
/// Every core semantic slot must have a distinct evidence id and distinct content
/// identity. Reference datasets must also be distinct from the execution artifacts
/// and from one another by both id and content digest.
///
/// Method/input/reference artifacts may not claim dates after the execution.
/// Output may be produced on the same calendar date or later, but not earlier.
/// Date comparisons are caller-declared chronology only.
pub fn bind_relativistic_execution_references(
    receipt: &RelativisticElectronicStructureReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<RelativisticExecutionReferenceBinding, RelativisticEvidenceBindingError> {
    receipt.validate()?;

    ensure_core_identity_distinctness(receipt)?;

    let execution = bundle.require_role(&receipt.receipt_id, EvidenceRole::SolverExecution)?;
    let input = bundle.require_role(&receipt.input_identity, EvidenceRole::ArtifactContent)?;
    let output = bundle.require_role(&receipt.output_identity, EvidenceRole::ArtifactContent)?;
    let hamiltonian = bundle.require_role(&receipt.hamiltonian_id, EvidenceRole::ArtifactContent)?;
    let basis = bundle.require_role(
        &receipt.basis_or_pseudopotential_id,
        EvidenceRole::ArtifactContent,
    )?;

    let core_references = [
        ("execution", execution),
        ("input", input),
        ("output", output),
        ("hamiltonian", hamiltonian),
        ("basis_or_pseudopotential", basis),
    ];
    ensure_core_content_distinctness(&core_references)?;

    let execution_date = require_date(execution.id.as_str(), execution.claimed_utc_date)?;
    let input_date = require_date(input.id.as_str(), input.claimed_utc_date)?;
    let output_date = require_date(output.id.as_str(), output.claimed_utc_date)?;
    let hamiltonian_date = require_date(hamiltonian.id.as_str(), hamiltonian.claimed_utc_date)?;
    let basis_date = require_date(basis.id.as_str(), basis.claimed_utc_date)?;

    for (id, date) in [
        (input.id.as_str(), input_date),
        (hamiltonian.id.as_str(), hamiltonian_date),
        (basis.id.as_str(), basis_date),
    ] {
        require_not_after_execution(id, date, execution_date)?;
    }

    if output_date < execution_date {
        return Err(RelativisticEvidenceBindingError::OutputPredatesExecution {
            output_yyyymmdd: output_date.yyyymmdd(),
            execution_yyyymmdd: execution_date.yyyymmdd(),
        });
    }

    let reserved: BTreeSet<&str> = [
        receipt.receipt_id.as_str(),
        receipt.input_identity.as_str(),
        receipt.output_identity.as_str(),
        receipt.hamiltonian_id.as_str(),
        receipt.basis_or_pseudopotential_id.as_str(),
    ]
    .into_iter()
    .collect();

    let mut seen_reference_ids = BTreeSet::new();
    let mut reference_bindings: Vec<RelativisticReferenceDatasetBinding> = Vec::new();
    for reference_id in &receipt.reference_dataset_ids {
        if !seen_reference_ids.insert(reference_id.as_str()) {
            return Err(
                RelativisticEvidenceBindingError::DuplicateReferenceDatasetIdentity(
                    reference_id.clone(),
                ),
            );
        }
        if reserved.contains(reference_id.as_str()) {
            return Err(
                RelativisticEvidenceBindingError::ReferenceDatasetAliasesExecutionArtifact(
                    reference_id.clone(),
                ),
            );
        }

        let reference = bundle.require_role(reference_id, EvidenceRole::ArtifactContent)?;
        if let Some(core_slot) = core_references.iter().find_map(|(slot, core)| {
            (core.content_sha256 == reference.content_sha256).then_some(*slot)
        }) {
            return Err(
                RelativisticEvidenceBindingError::ReferenceDatasetAliasesCoreContent {
                    reference_id: reference_id.clone(),
                    core_slot,
                },
            );
        }
        if let Some(previous) = reference_bindings
            .iter()
            .find(|bound| bound.content_sha256 == reference.content_sha256)
        {
            return Err(
                RelativisticEvidenceBindingError::DuplicateReferenceDatasetContent {
                    first_id: previous.evidence_id.clone(),
                    second_id: reference_id.clone(),
                },
            );
        }

        let reference_date = require_date(reference.id.as_str(), reference.claimed_utc_date)?;
        require_not_after_execution(reference.id.as_str(), reference_date, execution_date)?;
        reference_bindings.push(RelativisticReferenceDatasetBinding {
            evidence_id: reference.id.as_str().to_string(),
            content_sha256: reference.content_sha256.clone(),
            claimed_date: reference_date,
        });
    }

    Ok(RelativisticExecutionReferenceBinding {
        execution_evidence_id: execution.id.as_str().to_string(),
        input_evidence_id: input.id.as_str().to_string(),
        output_evidence_id: output.id.as_str().to_string(),
        hamiltonian_evidence_id: hamiltonian.id.as_str().to_string(),
        basis_or_pseudopotential_evidence_id: basis.id.as_str().to_string(),
        execution_content_sha256: execution.content_sha256.clone(),
        input_content_sha256: input.content_sha256.clone(),
        output_content_sha256: output.content_sha256.clone(),
        hamiltonian_content_sha256: hamiltonian.content_sha256.clone(),
        basis_or_pseudopotential_content_sha256: basis.content_sha256.clone(),
        execution_claimed_date: execution_date,
        input_claimed_date: input_date,
        output_claimed_date: output_date,
        hamiltonian_claimed_date: hamiltonian_date,
        basis_or_pseudopotential_claimed_date: basis_date,
        reference_datasets: reference_bindings,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

fn ensure_core_identity_distinctness(
    receipt: &RelativisticElectronicStructureReceipt,
) -> Result<(), RelativisticEvidenceBindingError> {
    let slots = [
        ("execution", receipt.receipt_id.as_str()),
        ("input", receipt.input_identity.as_str()),
        ("output", receipt.output_identity.as_str()),
        ("hamiltonian", receipt.hamiltonian_id.as_str()),
        (
            "basis_or_pseudopotential",
            receipt.basis_or_pseudopotential_id.as_str(),
        ),
    ];
    for left in 0..slots.len() {
        for right in left + 1..slots.len() {
            if slots[left].1 == slots[right].1 {
                return Err(RelativisticEvidenceBindingError::SemanticSlotIdentityAlias {
                    left_slot: slots[left].0,
                    right_slot: slots[right].0,
                });
            }
        }
    }
    Ok(())
}

fn ensure_core_content_distinctness(
    core: &[(&'static str, &ExternalEvidenceReference)],
) -> Result<(), RelativisticEvidenceBindingError> {
    for left in 0..core.len() {
        for right in left + 1..core.len() {
            if core[left].1.content_sha256 == core[right].1.content_sha256 {
                return Err(RelativisticEvidenceBindingError::SemanticSlotContentAlias {
                    left_slot: core[left].0,
                    right_slot: core[right].0,
                });
            }
        }
    }
    Ok(())
}

fn require_date(
    evidence_id: &str,
    date: Option<ClaimedUtcDate>,
) -> Result<ClaimedUtcDate, RelativisticEvidenceBindingError> {
    date.ok_or_else(|| RelativisticEvidenceBindingError::MissingClaimedDate(evidence_id.to_string()))
}

fn require_not_after_execution(
    evidence_id: &str,
    dependency_date: ClaimedUtcDate,
    execution_date: ClaimedUtcDate,
) -> Result<(), RelativisticEvidenceBindingError> {
    if dependency_date > execution_date {
        return Err(RelativisticEvidenceBindingError::DependencyPostdatesExecution {
            evidence_id: evidence_id.to_string(),
            dependency_yyyymmdd: dependency_date.yyyymmdd(),
            execution_yyyymmdd: execution_date.yyyymmdd(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::matter_relativistic::RelativisticElectronicRequirement;
    use symthaea_epistemic_types::{
        MatterSolverCapability, MatterSolverProfile, MatterValidationStage,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn reference(
        id: &str,
        role: EvidenceRole,
        digest_char: char,
        date: u32,
    ) -> ExternalEvidenceReference {
        ExternalEvidenceReference::try_new(
            id,
            role,
            digest(digest_char),
            format!("artifact://{id}"),
            format!("subject:{id}"),
            Some(date),
            Some("fixture issuer".to_string()),
        )
        .unwrap()
    }

    fn receipt(reference_ids: Vec<String>) -> RelativisticElectronicStructureReceipt {
        RelativisticElectronicStructureReceipt {
            receipt_id: "execution".to_string(),
            solver: MatterSolverProfile {
                name: "external-rel-qc".to_string(),
                version: "1.0".to_string(),
                method: "test spin-orbit method".to_string(),
                capabilities: vec![MatterSolverCapability::SpinOrbitElectronicStructure],
                limitations: vec!["fixture only".to_string()],
            },
            requirement: RelativisticElectronicRequirement::SpinOrbitResolved,
            element_atomic_numbers: vec![8, 118],
            hamiltonian_id: "hamiltonian".to_string(),
            basis_or_pseudopotential_id: "basis".to_string(),
            input_identity: "input".to_string(),
            output_identity: "output".to_string(),
            validation_stage: if reference_ids.is_empty() {
                MatterValidationStage::HighFidelitySimulated
            } else {
                MatterValidationStage::ReferenceBenchmarked
            },
            reference_dataset_ids: reference_ids,
            limitations: vec!["fixture only".to_string()],
        }
    }

    fn standard_bundle(reference_date: Option<u32>) -> ExternalEvidenceBundle {
        let mut refs = vec![
            reference("execution", EvidenceRole::SolverExecution, 'a', 20260210),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260209),
            reference("output", EvidenceRole::ArtifactContent, 'c', 20260210),
            reference("hamiltonian", EvidenceRole::ArtifactContent, 'd', 20250101),
            reference("basis", EvidenceRole::ArtifactContent, 'e', 20240101),
        ];
        if let Some(date) = reference_date {
            refs.push(reference("benchmark", EvidenceRole::ArtifactContent, 'f', date));
        }
        ExternalEvidenceBundle::new(refs).unwrap()
    }

    #[test]
    fn complete_binding_remains_reference_only_and_retains_reference_digest() {
        let receipt = receipt(vec!["benchmark".to_string()]);
        let bundle = standard_bundle(Some(20200101));
        let binding = bind_relativistic_execution_references(&receipt, &bundle).unwrap();

        assert_eq!(
            binding.reference_interpretation,
            EvidenceReferenceInterpretation::ReferenceOnly
        );
        assert_eq!(
            binding.chronology_interpretation,
            DeclaredChronologyInterpretation::DeclaredChronologyOnly
        );
        assert_eq!(binding.execution_evidence_id, "execution");
        assert_eq!(binding.reference_datasets.len(), 1);
        assert_eq!(binding.reference_datasets[0].evidence_id, "benchmark");
        assert_eq!(binding.reference_datasets[0].content_sha256.as_str(), digest('f'));
    }

    #[test]
    fn execution_requires_solver_execution_role() {
        let receipt = receipt(Vec::new());
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("execution", EvidenceRole::ArtifactContent, 'a', 20260210),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260209),
            reference("output", EvidenceRole::ArtifactContent, 'c', 20260210),
            reference("hamiltonian", EvidenceRole::ArtifactContent, 'd', 20250101),
            reference("basis", EvidenceRole::ArtifactContent, 'e', 20240101),
        ])
        .unwrap();

        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::ExternalEvidence(
                ExternalEvidenceError::RoleMismatch { .. }
            ))
        ));
    }

    #[test]
    fn core_semantic_slots_cannot_alias_by_identity() {
        let mut receipt = receipt(Vec::new());
        receipt.hamiltonian_id = "input".to_string();
        let bundle = standard_bundle(None);
        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::SemanticSlotIdentityAlias { .. })
        ));
    }

    #[test]
    fn core_semantic_slots_cannot_alias_by_content() {
        let receipt = receipt(Vec::new());
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("execution", EvidenceRole::SolverExecution, 'a', 20260210),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260209),
            reference("output", EvidenceRole::ArtifactContent, 'c', 20260210),
            reference("hamiltonian", EvidenceRole::ArtifactContent, 'b', 20250101),
            reference("basis", EvidenceRole::ArtifactContent, 'e', 20240101),
        ])
        .unwrap();
        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::SemanticSlotContentAlias { .. })
        ));
    }

    #[test]
    fn future_reference_dataset_fails_closed() {
        let receipt = receipt(vec!["benchmark".to_string()]);
        let bundle = standard_bundle(Some(20260301));
        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::DependencyPostdatesExecution { .. })
        ));
    }

    #[test]
    fn output_cannot_predate_execution() {
        let receipt = receipt(Vec::new());
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("execution", EvidenceRole::SolverExecution, 'a', 20260210),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260209),
            reference("output", EvidenceRole::ArtifactContent, 'c', 20260208),
            reference("hamiltonian", EvidenceRole::ArtifactContent, 'd', 20250101),
            reference("basis", EvidenceRole::ArtifactContent, 'e', 20240101),
        ])
        .unwrap();
        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::OutputPredatesExecution { .. })
        ));
    }

    #[test]
    fn reference_dataset_cannot_alias_core_artifact_by_identity() {
        let mut receipt = receipt(vec!["input".to_string()]);
        receipt.validation_stage = MatterValidationStage::ReferenceBenchmarked;
        let bundle = standard_bundle(None);
        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::ReferenceDatasetAliasesExecutionArtifact(id))
                if id == "input"
        ));
    }

    #[test]
    fn reference_dataset_cannot_alias_core_artifact_by_content() {
        let receipt = receipt(vec!["benchmark".to_string()]);
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("execution", EvidenceRole::SolverExecution, 'a', 20260210),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260209),
            reference("output", EvidenceRole::ArtifactContent, 'c', 20260210),
            reference("hamiltonian", EvidenceRole::ArtifactContent, 'd', 20250101),
            reference("basis", EvidenceRole::ArtifactContent, 'e', 20240101),
            reference("benchmark", EvidenceRole::ArtifactContent, 'b', 20200101),
        ])
        .unwrap();
        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::ReferenceDatasetAliasesCoreContent { .. })
        ));
    }

    #[test]
    fn reference_datasets_cannot_duplicate_content_under_different_ids() {
        let receipt = receipt(vec!["benchmark-a".to_string(), "benchmark-b".to_string()]);
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("execution", EvidenceRole::SolverExecution, 'a', 20260210),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260209),
            reference("output", EvidenceRole::ArtifactContent, 'c', 20260210),
            reference("hamiltonian", EvidenceRole::ArtifactContent, 'd', 20250101),
            reference("basis", EvidenceRole::ArtifactContent, 'e', 20240101),
            reference("benchmark-a", EvidenceRole::ArtifactContent, 'f', 20200101),
            reference("benchmark-b", EvidenceRole::ArtifactContent, 'f', 20210101),
        ])
        .unwrap();
        assert!(matches!(
            bind_relativistic_execution_references(&receipt, &bundle),
            Err(RelativisticEvidenceBindingError::DuplicateReferenceDatasetContent { .. })
        ));
    }
}
