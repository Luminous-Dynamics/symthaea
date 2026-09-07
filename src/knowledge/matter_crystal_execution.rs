// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Solver-neutral execution receipts for crystal/phase evidence.
//!
//! This layer moves the Matter Observatory from evidence labels toward executed
//! physics without embedding Quantum ESPRESSO, VASP, CP2K, AiiDA, atomate2, or
//! any other particular workflow system. Adapters may emit this contract after
//! they have executed and parsed an external solver.
//!
//! Process completion is deliberately distinct from scientific convergence.
//! `exit_code == 0` means only that the declared process completed successfully;
//! relaxation, phase-competition, and lattice-dynamics criteria remain governed
//! by the numerical admission rules in `matter_crystal_phase`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_epistemic_types::{
    EpistemicContext, MatterClaim, MatterScale, MatterSolverCapability, MatterValidationStage,
};
use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, DeclaredChronologyInterpretation, DeclaredTemporalRelation,
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError,
    ExternalEvidenceReference, Sha256Digest,
};

use super::matter_crystal_evidence_binding::{
    bind_crystal_phase_evidence, CrystalEvidenceBindingError, EvidenceBoundCrystalPhaseMatterClaim,
};
use super::matter_crystal_phase::{CrystalPhaseDiagnosticSnapshot, CrystalPhaseEvidenceSlot};

/// Exact scientific task performed by one external crystal-solver execution.
///
/// These values are intentionally more specific than a generic "DFT" or
/// "materials" label. They describe the task associated with one evidence slot;
/// they are not a claim that the calculation was converged or scientifically
/// correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrystalSolverTask {
    PeriodicElectronicStructure,
    StructuralRelaxation,
    ThermodynamicPhaseCompetition,
    LatticeDynamics,
}

impl CrystalSolverTask {
    pub fn required_slot(self) -> CrystalPhaseEvidenceSlot {
        match self {
            Self::PeriodicElectronicStructure => CrystalPhaseEvidenceSlot::PeriodicElectronicStructure,
            Self::StructuralRelaxation => CrystalPhaseEvidenceSlot::StructuralRelaxation,
            Self::ThermodynamicPhaseCompetition => {
                CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition
            }
            Self::LatticeDynamics => CrystalPhaseEvidenceSlot::DynamicalStability,
        }
    }
}

/// Producer-declared execution receipt for a parsed external crystal solver.
///
/// All string identities resolve through the canonical external evidence plane
/// during binding. The millisecond interval is retained exactly but is not a
/// trusted timestamp; external chronology remains `DeclaredChronologyOnly`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystalSolverExecutionReceipt {
    /// Exact Matter claim produced from this execution.
    pub claim_id: String,
    pub task: CrystalSolverTask,
    /// Canonical evidence id expected to have role `SolverExecution`.
    pub execution_evidence_id: String,
    /// Calendar date asserted for the execution reference.
    pub declared_execution_yyyymmdd: u32,
    /// Producer-declared UTC Unix millisecond interval.
    pub started_unix_ms: u64,
    pub finished_unix_ms: u64,
    /// A scientific execution is admitted only for process exit code zero.
    pub exit_code: i32,
    pub solver_name: String,
    pub solver_version: String,
    pub adapter_name: String,
    pub adapter_version: String,
    /// Fully rendered solver input.
    pub input_artifact_id: String,
    /// Full numerical configuration/k-point/cutoff/etc. representation.
    pub configuration_artifact_id: String,
    /// Reproducible environment/capsule or solver implementation snapshot.
    pub environment_snapshot_id: String,
    /// Parser/normalizer implementation snapshot.
    pub parser_snapshot_id: String,
    /// Raw/structured outputs consumed by the parser. At least one is required.
    pub output_artifact_ids: Vec<String>,
    /// Frozen external dependencies such as pseudopotential/library snapshots.
    pub dependency_snapshot_ids: Vec<String>,
}

impl CrystalSolverExecutionReceipt {
    pub fn validate(&self) -> Result<(), CrystalExecutionError> {
        for (field, value) in [
            ("claim_id", self.claim_id.as_str()),
            ("execution_evidence_id", self.execution_evidence_id.as_str()),
            ("solver_name", self.solver_name.as_str()),
            ("solver_version", self.solver_version.as_str()),
            ("adapter_name", self.adapter_name.as_str()),
            ("adapter_version", self.adapter_version.as_str()),
            ("input_artifact_id", self.input_artifact_id.as_str()),
            ("configuration_artifact_id", self.configuration_artifact_id.as_str()),
            ("environment_snapshot_id", self.environment_snapshot_id.as_str()),
            ("parser_snapshot_id", self.parser_snapshot_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CrystalExecutionError::EmptyField(field));
            }
        }
        if self.declared_execution_yyyymmdd == 0 {
            return Err(CrystalExecutionError::MissingDeclaredExecutionDate);
        }
        if self.started_unix_ms == 0 || self.finished_unix_ms < self.started_unix_ms {
            return Err(CrystalExecutionError::InvalidExecutionInterval);
        }
        if self.exit_code != 0 {
            return Err(CrystalExecutionError::NonZeroExitCode(self.exit_code));
        }
        if self.output_artifact_ids.is_empty() {
            return Err(CrystalExecutionError::MissingOutputArtifacts);
        }
        if self
            .output_artifact_ids
            .iter()
            .chain(self.dependency_snapshot_ids.iter())
            .any(|id| id.trim().is_empty())
        {
            return Err(CrystalExecutionError::EmptyArtifactIdentity);
        }

        let mut ids = BTreeSet::new();
        for id in self.all_semantic_ids() {
            if !ids.insert(id) {
                return Err(CrystalExecutionError::DuplicateSemanticIdentity(id.to_string()));
            }
        }
        Ok(())
    }

    fn all_semantic_ids(&self) -> impl Iterator<Item = &str> {
        [
            self.execution_evidence_id.as_str(),
            self.input_artifact_id.as_str(),
            self.configuration_artifact_id.as_str(),
            self.environment_snapshot_id.as_str(),
            self.parser_snapshot_id.as_str(),
        ]
        .into_iter()
        .chain(self.output_artifact_ids.iter().map(String::as_str))
        .chain(self.dependency_snapshot_ids.iter().map(String::as_str))
    }
}

/// Canonically bound external execution and its exact artifact identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystalSolverExecutionBinding {
    pub claim_id: String,
    pub task: CrystalSolverTask,
    pub execution_evidence_id: String,
    pub execution_content_sha256: Sha256Digest,
    pub execution_claimed_date: ClaimedUtcDate,
    pub started_unix_ms: u64,
    pub finished_unix_ms: u64,
    pub solver_name: String,
    pub solver_version: String,
    pub adapter_name: String,
    pub adapter_version: String,
    pub input_content_sha256: Sha256Digest,
    pub configuration_content_sha256: Sha256Digest,
    pub environment_content_sha256: Sha256Digest,
    pub parser_content_sha256: Sha256Digest,
    pub output_content_sha256: Vec<Sha256Digest>,
    pub dependency_content_sha256: Vec<Sha256Digest>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

/// Declared criterion-before-execution relationship for one numerical slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystalCriterionExecutionChronology {
    pub slot: CrystalPhaseEvidenceSlot,
    pub relation: DeclaredTemporalRelation,
}

/// Fully evidence-bound crystal candidate with one external execution per
/// required physics slot.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionBoundCrystalPhaseMatterClaim {
    pub evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    pub executions: Vec<CrystalSolverExecutionBinding>,
    pub criterion_chronology: Vec<CrystalCriterionExecutionChronology>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

/// Bind exactly one external execution to each required crystal evidence slot.
///
/// The parent #801 binding is reconstructed first, so a hand-edited parent
/// object cannot bypass its provenance and authority checks. Numerical
/// preregistration chronology is then required to be strictly earlier, at the
/// evidence plane's current calendar-day granularity, than the execution for
/// the corresponding relaxation/hull/phonon task.
pub fn bind_crystal_solver_executions(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    receipts: &[CrystalSolverExecutionReceipt],
    bundle: &ExternalEvidenceBundle,
) -> Result<ExecutionBoundCrystalPhaseMatterClaim, CrystalExecutionError> {
    let rebound = bind_crystal_phase_evidence(evidence_bound.admitted.clone(), bundle)?;
    if rebound != evidence_bound {
        return Err(CrystalExecutionError::ParentEvidenceBindingMismatch);
    }

    if receipts.len() != 4 {
        return Err(CrystalExecutionError::IncompleteExecutionSet);
    }

    let required_snapshots: BTreeMap<CrystalPhaseEvidenceSlot, &MatterClaim> = evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .filter(|snapshot| snapshot.slot != CrystalPhaseEvidenceSlot::HeuristicCompositionScreening)
        .map(|snapshot| (snapshot.slot, &snapshot.claim))
        .collect();
    if required_snapshots.len() != 4 {
        return Err(CrystalExecutionError::IncompleteExecutionSet);
    }

    let mut seen_tasks = BTreeSet::new();
    let mut seen_execution_ids = BTreeSet::new();
    let mut bound = Vec::with_capacity(4);
    let mut execution_id_by_slot = BTreeMap::new();

    for receipt in receipts {
        receipt.validate()?;
        if !seen_tasks.insert(receipt.task) {
            return Err(CrystalExecutionError::DuplicateTask(receipt.task));
        }
        if !seen_execution_ids.insert(receipt.execution_evidence_id.clone()) {
            return Err(CrystalExecutionError::DuplicateExecutionIdentity(
                receipt.execution_evidence_id.clone(),
            ));
        }

        let slot = receipt.task.required_slot();
        let claim = required_snapshots
            .get(&slot)
            .ok_or(CrystalExecutionError::MissingClaimForTask(receipt.task))?;
        validate_receipt_against_claim(receipt, claim)?;
        let binding = bind_execution_references(receipt, bundle)?;
        execution_id_by_slot.insert(slot, binding.execution_evidence_id.clone());
        bound.push(binding);
    }

    for task in [
        CrystalSolverTask::PeriodicElectronicStructure,
        CrystalSolverTask::StructuralRelaxation,
        CrystalSolverTask::ThermodynamicPhaseCompetition,
        CrystalSolverTask::LatticeDynamics,
    ] {
        if !seen_tasks.contains(&task) {
            return Err(CrystalExecutionError::MissingTask(task));
        }
    }

    let mut chronology = Vec::with_capacity(3);
    for criterion in &evidence_bound.criterion_bindings {
        let execution_id = execution_id_by_slot
            .get(&criterion.slot)
            .ok_or(CrystalExecutionError::CriterionHasNoExecution(criterion.slot))?;
        let relation = bundle.require_preregistration_before(&criterion.evidence_id, execution_id)?;
        chronology.push(CrystalCriterionExecutionChronology {
            slot: criterion.slot,
            relation,
        });
    }
    if chronology.len() != 3 {
        return Err(CrystalExecutionError::IncompleteCriterionChronology);
    }

    Ok(ExecutionBoundCrystalPhaseMatterClaim {
        evidence_bound,
        executions: bound,
        criterion_chronology: chronology,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

fn validate_receipt_against_claim(
    receipt: &CrystalSolverExecutionReceipt,
    claim: &MatterClaim,
) -> Result<(), CrystalExecutionError> {
    if receipt.claim_id != claim.claim_id {
        return Err(CrystalExecutionError::ClaimIdentityMismatch {
            receipt: receipt.claim_id.clone(),
            claim: claim.claim_id.clone(),
        });
    }
    if !matches!(claim.scale, MatterScale::Crystal | MatterScale::Multiscale) {
        return Err(CrystalExecutionError::ClaimScaleMismatch(claim.scale));
    }
    if claim.epistemic.context != EpistemicContext::Scientific {
        return Err(CrystalExecutionError::ClaimNotScientific);
    }
    if !matches!(
        claim.validation_stage,
        MatterValidationStage::PhysicsSimulated
            | MatterValidationStage::CrossModelSupported
            | MatterValidationStage::HighFidelitySimulated
            | MatterValidationStage::ReferenceBenchmarked
    ) {
        return Err(CrystalExecutionError::ClaimStageNotComputational(
            claim.validation_stage,
        ));
    }
    let solver = claim
        .solver
        .as_ref()
        .ok_or(CrystalExecutionError::MissingSolverProfile)?;
    if solver.name != receipt.solver_name || solver.version != receipt.solver_version {
        return Err(CrystalExecutionError::SolverIdentityMismatch);
    }
    if receipt.task == CrystalSolverTask::PeriodicElectronicStructure
        && !solver.supports(MatterSolverCapability::PeriodicElectronicStructure)
    {
        return Err(CrystalExecutionError::PeriodicCapabilityMissing);
    }
    if !claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == receipt.execution_evidence_id)
    {
        return Err(CrystalExecutionError::ExecutionEvidenceMissingFromClaim);
    }
    Ok(())
}

fn bind_execution_references(
    receipt: &CrystalSolverExecutionReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<CrystalSolverExecutionBinding, CrystalExecutionError> {
    let execution = bundle.require_role(&receipt.execution_evidence_id, EvidenceRole::SolverExecution)?;
    let input = bundle.require_role(&receipt.input_artifact_id, EvidenceRole::ArtifactContent)?;
    let configuration = bundle.require_role(
        &receipt.configuration_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    let environment = bundle.require_role(
        &receipt.environment_snapshot_id,
        EvidenceRole::ImplementationSnapshot,
    )?;
    let parser = bundle.require_role(
        &receipt.parser_snapshot_id,
        EvidenceRole::ImplementationSnapshot,
    )?;

    let execution_date = require_date(execution)?;
    if execution_date.yyyymmdd() != receipt.declared_execution_yyyymmdd {
        return Err(CrystalExecutionError::ExecutionDateMismatch {
            receipt_yyyymmdd: receipt.declared_execution_yyyymmdd,
            evidence_yyyymmdd: execution_date.yyyymmdd(),
        });
    }

    let mut core = vec![execution, input, configuration, environment, parser];
    let mut output_refs = Vec::new();
    for id in &receipt.output_artifact_ids {
        let reference = bundle.require_role(id, EvidenceRole::ArtifactContent)?;
        output_refs.push(reference);
        core.push(reference);
    }
    let mut dependency_refs = Vec::new();
    for id in &receipt.dependency_snapshot_ids {
        let reference = bundle.require_role(id, EvidenceRole::DependencySnapshot)?;
        dependency_refs.push(reference);
        core.push(reference);
    }
    ensure_distinct_content(&core)?;

    for reference in [input, configuration, environment, parser]
        .into_iter()
        .chain(dependency_refs.iter().copied())
    {
        let date = require_date(reference)?;
        if date > execution_date {
            return Err(CrystalExecutionError::DependencyPostdatesExecution {
                evidence_id: reference.id.as_str().to_string(),
                dependency_yyyymmdd: date.yyyymmdd(),
                execution_yyyymmdd: execution_date.yyyymmdd(),
            });
        }
    }
    for reference in &output_refs {
        let date = require_date(reference)?;
        if date < execution_date {
            return Err(CrystalExecutionError::OutputPredatesExecution {
                evidence_id: reference.id.as_str().to_string(),
                output_yyyymmdd: date.yyyymmdd(),
                execution_yyyymmdd: execution_date.yyyymmdd(),
            });
        }
    }

    Ok(CrystalSolverExecutionBinding {
        claim_id: receipt.claim_id.clone(),
        task: receipt.task,
        execution_evidence_id: execution.id.as_str().to_string(),
        execution_content_sha256: execution.content_sha256.clone(),
        execution_claimed_date: execution_date,
        started_unix_ms: receipt.started_unix_ms,
        finished_unix_ms: receipt.finished_unix_ms,
        solver_name: receipt.solver_name.clone(),
        solver_version: receipt.solver_version.clone(),
        adapter_name: receipt.adapter_name.clone(),
        adapter_version: receipt.adapter_version.clone(),
        input_content_sha256: input.content_sha256.clone(),
        configuration_content_sha256: configuration.content_sha256.clone(),
        environment_content_sha256: environment.content_sha256.clone(),
        parser_content_sha256: parser.content_sha256.clone(),
        output_content_sha256: output_refs
            .iter()
            .map(|reference| reference.content_sha256.clone())
            .collect(),
        dependency_content_sha256: dependency_refs
            .iter()
            .map(|reference| reference.content_sha256.clone())
            .collect(),
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

fn ensure_distinct_content(
    references: &[&ExternalEvidenceReference],
) -> Result<(), CrystalExecutionError> {
    let mut seen = BTreeMap::<String, String>::new();
    for reference in references {
        let digest = reference.content_sha256.as_str().to_string();
        if let Some(first_id) = seen.insert(digest, reference.id.as_str().to_string()) {
            return Err(CrystalExecutionError::DuplicateContentIdentity {
                first_id,
                second_id: reference.id.as_str().to_string(),
            });
        }
    }
    Ok(())
}

fn require_date(
    reference: &ExternalEvidenceReference,
) -> Result<ClaimedUtcDate, CrystalExecutionError> {
    reference
        .claimed_utc_date
        .ok_or_else(|| CrystalExecutionError::MissingClaimedDate(reference.id.as_str().to_string()))
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrystalExecutionError {
    CrystalEvidence(CrystalEvidenceBindingError),
    ExternalEvidence(ExternalEvidenceError),
    EmptyField(&'static str),
    MissingDeclaredExecutionDate,
    InvalidExecutionInterval,
    NonZeroExitCode(i32),
    MissingOutputArtifacts,
    EmptyArtifactIdentity,
    DuplicateSemanticIdentity(String),
    ParentEvidenceBindingMismatch,
    IncompleteExecutionSet,
    DuplicateTask(CrystalSolverTask),
    DuplicateExecutionIdentity(String),
    MissingClaimForTask(CrystalSolverTask),
    MissingTask(CrystalSolverTask),
    CriterionHasNoExecution(CrystalPhaseEvidenceSlot),
    IncompleteCriterionChronology,
    ClaimIdentityMismatch { receipt: String, claim: String },
    ClaimScaleMismatch(MatterScale),
    ClaimNotScientific,
    ClaimStageNotComputational(MatterValidationStage),
    MissingSolverProfile,
    SolverIdentityMismatch,
    PeriodicCapabilityMissing,
    ExecutionEvidenceMissingFromClaim,
    ExecutionDateMismatch {
        receipt_yyyymmdd: u32,
        evidence_yyyymmdd: u32,
    },
    DuplicateContentIdentity { first_id: String, second_id: String },
    MissingClaimedDate(String),
    DependencyPostdatesExecution {
        evidence_id: String,
        dependency_yyyymmdd: u32,
        execution_yyyymmdd: u32,
    },
    OutputPredatesExecution {
        evidence_id: String,
        output_yyyymmdd: u32,
        execution_yyyymmdd: u32,
    },
}

impl From<CrystalEvidenceBindingError> for CrystalExecutionError {
    fn from(value: CrystalEvidenceBindingError) -> Self {
        Self::CrystalEvidence(value)
    }
}

impl From<ExternalEvidenceError> for CrystalExecutionError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for CrystalExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CrystalEvidence(error) => write!(f, "parent crystal evidence rejected: {error}"),
            Self::ExternalEvidence(error) => write!(f, "{error}"),
            Self::EmptyField(field) => write!(f, "crystal execution field `{field}` is empty"),
            Self::MissingDeclaredExecutionDate => write!(f, "declared execution date is missing"),
            Self::InvalidExecutionInterval => write!(f, "execution interval is invalid"),
            Self::NonZeroExitCode(code) => write!(f, "solver process exited with non-zero code {code}"),
            Self::MissingOutputArtifacts => write!(f, "solver execution requires at least one output artifact"),
            Self::EmptyArtifactIdentity => write!(f, "solver artifact identities must not be empty"),
            Self::DuplicateSemanticIdentity(id) => write!(f, "solver execution reuses semantic evidence id `{id}`"),
            Self::ParentEvidenceBindingMismatch => write!(f, "stored #801 crystal evidence binding does not equal a fresh canonical rebind"),
            Self::IncompleteExecutionSet => write!(f, "exactly four required crystal solver executions are required"),
            Self::DuplicateTask(task) => write!(f, "crystal solver task {task:?} occurs more than once"),
            Self::DuplicateExecutionIdentity(id) => write!(f, "solver execution evidence id `{id}` occurs more than once"),
            Self::MissingClaimForTask(task) => write!(f, "no crystal evidence claim exists for task {task:?}"),
            Self::MissingTask(task) => write!(f, "required crystal solver task {task:?} is missing"),
            Self::CriterionHasNoExecution(slot) => write!(f, "criterion slot {slot:?} has no corresponding execution"),
            Self::IncompleteCriterionChronology => write!(f, "all three numerical crystal criteria require declared preregistration-before-execution chronology"),
            Self::ClaimIdentityMismatch { receipt, claim } => write!(f, "execution receipt claim `{receipt}` does not match evidence claim `{claim}`"),
            Self::ClaimScaleMismatch(scale) => write!(f, "execution evidence claim has non-crystal scale {scale:?}"),
            Self::ClaimNotScientific => write!(f, "execution evidence claim is not in Scientific epistemic context"),
            Self::ClaimStageNotComputational(stage) => write!(f, "execution evidence claim stage {stage:?} is not an admissible computational physics stage"),
            Self::MissingSolverProfile => write!(f, "execution evidence claim has no solver profile"),
            Self::SolverIdentityMismatch => write!(f, "execution receipt solver name/version does not match the Matter claim solver profile"),
            Self::PeriodicCapabilityMissing => write!(f, "periodic execution claim does not advertise PeriodicElectronicStructure"),
            Self::ExecutionEvidenceMissingFromClaim => write!(f, "Matter claim does not retain the solver execution evidence id"),
            Self::ExecutionDateMismatch { receipt_yyyymmdd, evidence_yyyymmdd } => write!(f, "execution receipt date {receipt_yyyymmdd} does not match canonical execution-reference date {evidence_yyyymmdd}"),
            Self::DuplicateContentIdentity { first_id, second_id } => write!(f, "distinct execution semantic slots `{first_id}` and `{second_id}` resolve to identical content"),
            Self::MissingClaimedDate(id) => write!(f, "execution evidence `{id}` has no claimed UTC date"),
            Self::DependencyPostdatesExecution { evidence_id, dependency_yyyymmdd, execution_yyyymmdd } => write!(f, "solver dependency `{evidence_id}` date {dependency_yyyymmdd} postdates execution date {execution_yyyymmdd}"),
            Self::OutputPredatesExecution { evidence_id, output_yyyymmdd, execution_yyyymmdd } => write!(f, "solver output `{evidence_id}` date {output_yyyymmdd} predates execution date {execution_yyyymmdd}"),
        }
    }
}

impl std::error::Error for CrystalExecutionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{MatterEvidenceRef, MatterSolverProfile};
    use symthaea_evidence_plane::external_receipt::ExternalEvidenceReference;

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

    fn claim(execution_id: &str) -> MatterClaim {
        MatterClaim::local_computational(
            "claim:periodic",
            "periodic crystal fixture",
            MatterScale::Crystal,
            MatterValidationStage::HighFidelitySimulated,
            MatterSolverProfile {
                name: "external-solid-solver".into(),
                version: "1.0".into(),
                method: "periodic fixture".into(),
                capabilities: vec![MatterSolverCapability::PeriodicElectronicStructure],
                limitations: vec!["fixture only".into()],
            },
        )
        .unwrap()
        .with_evidence(MatterEvidenceRef {
            evidence_id: execution_id.into(),
            run_id: None,
            config_hash: None,
            dataset_ids: Vec::new(),
        })
    }

    fn receipt() -> CrystalSolverExecutionReceipt {
        CrystalSolverExecutionReceipt {
            claim_id: "claim:periodic".into(),
            task: CrystalSolverTask::PeriodicElectronicStructure,
            execution_evidence_id: "exec".into(),
            declared_execution_yyyymmdd: 20260907,
            started_unix_ms: 1_000,
            finished_unix_ms: 2_000,
            exit_code: 0,
            solver_name: "external-solid-solver".into(),
            solver_version: "1.0".into(),
            adapter_name: "symthaea-test-adapter".into(),
            adapter_version: "1".into(),
            input_artifact_id: "input".into(),
            configuration_artifact_id: "config".into(),
            environment_snapshot_id: "env".into(),
            parser_snapshot_id: "parser".into(),
            output_artifact_ids: vec!["output".into()],
            dependency_snapshot_ids: vec!["dependency".into()],
        }
    }

    fn bundle() -> ExternalEvidenceBundle {
        ExternalEvidenceBundle::new(vec![
            reference("exec", EvidenceRole::SolverExecution, 'a', 20260907),
            reference("input", EvidenceRole::ArtifactContent, 'b', 20260906),
            reference("config", EvidenceRole::ArtifactContent, 'c', 20260906),
            reference("env", EvidenceRole::ImplementationSnapshot, 'd', 20260906),
            reference("parser", EvidenceRole::ImplementationSnapshot, 'e', 20260906),
            reference("output", EvidenceRole::ArtifactContent, 'f', 20260907),
            reference("dependency", EvidenceRole::DependencySnapshot, '1', 20260905),
        ])
        .unwrap()
    }

    #[test]
    fn task_specific_execution_binds_exact_artifacts() {
        let receipt = receipt();
        receipt.validate().unwrap();
        let claim = claim("exec");
        validate_receipt_against_claim(&receipt, &claim).unwrap();
        let binding = bind_execution_references(&receipt, &bundle()).unwrap();
        assert_eq!(binding.task, CrystalSolverTask::PeriodicElectronicStructure);
        assert_eq!(binding.execution_claimed_date.yyyymmdd(), 20260907);
        assert_eq!(binding.output_content_sha256.len(), 1);
        assert_eq!(binding.dependency_content_sha256.len(), 1);
        assert_eq!(
            binding.reference_interpretation,
            EvidenceReferenceInterpretation::ReferenceOnly
        );
    }

    #[test]
    fn exit_zero_is_required_but_does_not_encode_convergence() {
        let mut receipt = receipt();
        receipt.exit_code = 2;
        assert_eq!(
            receipt.validate().unwrap_err(),
            CrystalExecutionError::NonZeroExitCode(2)
        );
        // No `converged` or `stable` field exists on the receipt: numerical
        // scientific admission remains in the parent crystal criteria.
    }

    #[test]
    fn solver_identity_must_match_claim() {
        let mut receipt = receipt();
        receipt.solver_version = "different".into();
        assert_eq!(
            validate_receipt_against_claim(&receipt, &claim("exec")).unwrap_err(),
            CrystalExecutionError::SolverIdentityMismatch
        );
    }

    #[test]
    fn execution_id_must_be_retained_by_claim() {
        assert_eq!(
            validate_receipt_against_claim(&receipt(), &claim("other")).unwrap_err(),
            CrystalExecutionError::ExecutionEvidenceMissingFromClaim
        );
    }

    #[test]
    fn dependency_cannot_postdate_execution() {
        let mut refs = bundle().references().to_vec();
        refs.retain(|reference| reference.id.as_str() != "dependency");
        refs.push(reference(
            "dependency",
            EvidenceRole::DependencySnapshot,
            '1',
            20260908,
        ));
        let bad = ExternalEvidenceBundle::new(refs).unwrap();
        assert!(matches!(
            bind_execution_references(&receipt(), &bad),
            Err(CrystalExecutionError::DependencyPostdatesExecution { .. })
        ));
    }

    #[test]
    fn distinct_ids_cannot_hide_same_content() {
        let mut refs = bundle().references().to_vec();
        refs.retain(|reference| reference.id.as_str() != "config");
        refs.push(reference("config", EvidenceRole::ArtifactContent, 'b', 20260906));
        let bad = ExternalEvidenceBundle::new(refs).unwrap();
        assert!(matches!(
            bind_execution_references(&receipt(), &bad),
            Err(CrystalExecutionError::DuplicateContentIdentity { .. })
        ));
    }
}
