//! Evidence-gated fusion engineering decision compiler contract, v0.1.
//!
//! This is a composition and claim-boundary contract. It is not a fusion
//! systems code, a specialist solver, a physical qualification authority,
//! a manufacturing release system, or an experiment authorization service.

use std::collections::{BTreeSet, VecDeque};

pub mod authority_to_recovery;
pub mod cross_solver_comparison;
pub mod evidence_dependence;
pub mod plant_service_gate;
pub mod screening_funnel;
pub mod validation_domain;

pub const COMPILER_VERSION: &str = "fusion-decision-compiler/0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    SyntheticFixture,
    ExternalReference,
    ModelOutput,
    MeasuredObservation,
    IndependentVerification,
    PhysicalQualification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimClass {
    Synthetic,
    ExternalReported,
    Modelled,
    Measured,
    IndependentlyVerified,
    PhysicallyQualified,
}

impl EvidenceClass {
    fn supports(self, claim: ClaimClass) -> bool {
        matches!(
            (self, claim),
            (Self::SyntheticFixture, ClaimClass::Synthetic)
                | (Self::ExternalReference, ClaimClass::ExternalReported)
                | (Self::ModelOutput, ClaimClass::Modelled)
                | (Self::MeasuredObservation, ClaimClass::Measured)
                | (
                    Self::IndependentVerification,
                    ClaimClass::IndependentlyVerified
                )
                | (Self::PhysicalQualification, ClaimClass::PhysicallyQualified)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRef {
    pub id: String,
    pub subject_ref: String,
    pub subject_revision: String,
    pub scope_ref: String,
    /// Declared SHA-256 identity. This crate does not fetch or hash payload bytes.
    pub content_sha256: String,
    pub class: EvidenceClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimRequest {
    pub claim_id: String,
    pub subject_ref: String,
    pub subject_revision: String,
    pub evidence_id: String,
    pub requested_class: ClaimClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HardGateKind {
    EnergyBoundary,
    MaterialApplicability,
    ManufacturingCapability,
    InspectionAccess,
    AssemblyAccess,
    ReplacementAccess,
}

impl HardGateKind {
    pub const REQUIRED: [Self; 6] = [
        Self::EnergyBoundary,
        Self::MaterialApplicability,
        Self::ManufacturingCapability,
        Self::InspectionAccess,
        Self::AssemblyAccess,
        Self::ReplacementAccess,
    ];

    pub const fn scope_ref(self) -> &'static str {
        match self {
            Self::EnergyBoundary => "energy-boundary",
            Self::MaterialApplicability => "material-applicability",
            Self::ManufacturingCapability => "manufacturing-capability",
            Self::InspectionAccess => "inspection-access",
            Self::AssemblyAccess => "assembly-access",
            Self::ReplacementAccess => "replacement-access",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    Pass,
    Fail,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardGate {
    pub kind: HardGateKind,
    pub state: GateState,
    pub reason_code: Option<String>,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelUse {
    EngineeringScreening,
    SingleStepPrediction,
    OpenLoopRollout,
    OptimizationInsideSurrogate,
    ReferenceSolverEvaluation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    InDeclaredDomain,
    OutOfDomain,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convergence {
    Converged,
    NonConverged,
    NotReported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredModelDependencies {
    pub dataset_group: Option<String>,
    pub code_group: Option<String>,
    pub assumption_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverRun {
    pub run_id: String,
    pub solver_id: String,
    pub solver_revision: String,
    /// Exact candidate design revision evaluated by this run.
    pub candidate_revision: String,
    /// Exact geometry identity evaluated by this run.
    pub geometry_sha256: String,
    pub supported_regimes: Vec<String>,
    pub supported_observables: Vec<String>,
    pub supported_uses: Vec<ModelUse>,
    pub applicability: Applicability,
    pub exit_code: Option<i32>,
    pub convergence: Convergence,
    pub output_observables: Vec<String>,
    pub input_sha256: String,
    pub output_sha256: String,
    pub dependencies: DeclaredModelDependencies,
}

/// Typed service metric. The unit is encoded in the variant so target gain,
/// plasma gain, driver wall-plug gain and net plant electricity are never
/// silently treated as interchangeable quantities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceMetric {
    TargetGainDimensionless,
    PlasmaGainDimensionless,
    DriverWallPlugGainDimensionless,
    GrossThermalPowerMegawatt,
    GrossElectricPowerMegawatt,
    NetElectricExportPowerMegawatt,
    UsefulThermalPowerMegawatt,
    LifetimeNetElectricEnergyMegawattHour,
    LifetimeUsefulThermalEnergyMegawattHour,
    AvailabilityFraction,
}

impl ServiceMetric {
    /// Exact evidence scope required for this objective's declared service boundary.
    pub const fn boundary_scope_ref(self) -> &'static str {
        match self {
            Self::TargetGainDimensionless => "target-gain-boundary",
            Self::PlasmaGainDimensionless => "plasma-gain-boundary",
            Self::DriverWallPlugGainDimensionless => "driver-wall-plug-boundary",
            Self::GrossThermalPowerMegawatt => "gross-thermal-boundary",
            Self::GrossElectricPowerMegawatt => "gross-electric-boundary",
            Self::NetElectricExportPowerMegawatt => "net-electric-export-boundary",
            Self::UsefulThermalPowerMegawatt => "useful-thermal-service-boundary",
            Self::LifetimeNetElectricEnergyMegawattHour => "lifetime-net-electric-energy-boundary",
            Self::LifetimeUsefulThermalEnergyMegawattHour => {
                "lifetime-useful-thermal-energy-boundary"
            }
            Self::AvailabilityFraction => "plant-availability-boundary",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateInput {
    pub candidate_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub objective_id: String,
    pub service_metric: ServiceMetric,
    /// Missing service boundary is unknown, never an implicit zero.
    pub service_boundary_ref: Option<String>,
    pub required_regime: String,
    pub required_model_use: ModelUse,
    pub required_observables: Vec<String>,
    pub evidence: Vec<EvidenceRef>,
    pub claims: Vec<ClaimRequest>,
    pub hard_gates: Vec<HardGate>,
    pub solver_runs: Vec<SolverRun>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateDisposition {
    Blocked,
    Indeterminate,
    EligibleForFurtherEngineeringReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimCeiling {
    FurtherEngineeringReviewOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateFindingState {
    Pass,
    Fail,
    Unknown,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateFinding {
    pub kind: HardGateKind,
    pub state: GateFindingState,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverFindingState {
    CompletedWithinDeclaredProfile,
    UnsupportedRegime,
    UnsupportedUse,
    UnsupportedObservable,
    OutOfDomain,
    ApplicabilityUnknown,
    FailedExecution,
    NonConverged,
    ConvergenceNotReported,
    NotExecuted,
    MissingOutput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverFinding {
    pub run_id: String,
    pub solver_id: String,
    pub state: SolverFindingState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairwiseDependence {
    CommonModeSharedDependency,
    UnknownDependencyIdentity,
    IndependentUnderDeclaredFactors,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependenceFinding {
    pub first_run_id: String,
    pub second_run_id: String,
    pub disposition: PairwiseDependence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionBuild {
    pub compiler_version: &'static str,
    pub candidate_id: String,
    pub candidate_revision: String,
    pub objective_id: String,
    pub service_metric: ServiceMetric,
    pub disposition: CandidateDisposition,
    pub gate_findings: Vec<GateFinding>,
    pub solver_findings: Vec<SolverFinding>,
    pub dependence_findings: Vec<DependenceFinding>,
    pub diagnostics: Vec<String>,
    pub claim_ceiling: ClaimCeiling,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    EmptyField(&'static str),
    InvalidSha256(&'static str),
    DuplicateEvidenceId(String),
    DuplicateClaimId(String),
    DuplicateGate(HardGateKind),
    DuplicateSolverRunId(String),
    SolverCandidateRevisionMismatch(String),
    SolverGeometryMismatch(String),
    PassingGateWithoutEvidence(HardGateKind),
    NonPassingGateWithoutReason(HardGateKind),
    DanglingGateEvidence {
        gate: HardGateKind,
        evidence_id: String,
    },
    GateEvidenceScopeMismatch {
        gate: HardGateKind,
        evidence_id: String,
    },
    GateEvidenceSubjectMismatch {
        gate: HardGateKind,
        evidence_id: String,
    },
    DanglingClaimEvidence {
        claim_id: String,
        evidence_id: String,
    },
    ClaimSubjectMismatch(String),
    ClaimRevisionMismatch(String),
    ClaimClassMismatch(String),
    DuplicateRequiredObservable(String),
    MissingServiceBoundaryEvidence(String),
    ServiceBoundaryScopeMismatch(String),
    ServiceBoundarySubjectMismatch(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyEdge {
    pub upstream: String,
    pub downstream: String,
}

/// Compile a candidate into a bounded decision build.
///
/// A hard failure dominates unknowns. Missing information yields
/// Indeterminate. The strongest possible outcome is eligibility for further
/// engineering review; no physical or manufacturing authority is represented.
pub fn compile_candidate(
    candidate: &CandidateInput,
) -> Result<DecisionBuild, Vec<ValidationError>> {
    let errors = validate_candidate(candidate);
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut blocked = false;
    let mut indeterminate = false;
    let mut diagnostics = Vec::new();

    if candidate
        .service_boundary_ref
        .as_deref()
        .is_none_or(|value| value.trim().is_empty())
    {
        indeterminate = true;
        diagnostics.push("service boundary missing or unresolved".to_string());
    }

    let mut gate_findings = Vec::with_capacity(HardGateKind::REQUIRED.len());
    for kind in HardGateKind::REQUIRED {
        match candidate.hard_gates.iter().find(|gate| gate.kind == kind) {
            None => {
                indeterminate = true;
                diagnostics.push(format!("required gate missing: {kind:?}"));
                gate_findings.push(GateFinding {
                    kind,
                    state: GateFindingState::Missing,
                    reason_code: Some("missing-required-gate".to_string()),
                });
            }
            Some(gate) => {
                let state = match gate.state {
                    GateState::Pass => GateFindingState::Pass,
                    GateState::Fail => {
                        blocked = true;
                        diagnostics.push(format!("hard gate failed: {kind:?}"));
                        GateFindingState::Fail
                    }
                    GateState::Unknown => {
                        indeterminate = true;
                        diagnostics.push(format!("hard gate unknown: {kind:?}"));
                        GateFindingState::Unknown
                    }
                };
                gate_findings.push(GateFinding {
                    kind,
                    state,
                    reason_code: gate.reason_code.clone(),
                });
            }
        }
    }

    let mut solver_findings = Vec::with_capacity(candidate.solver_runs.len());
    if candidate.solver_runs.is_empty() {
        indeterminate = true;
        diagnostics.push("no solver run supplied".to_string());
    }

    for run in &candidate.solver_runs {
        let finding = evaluate_solver_run(candidate, run);
        match finding.state {
            SolverFindingState::CompletedWithinDeclaredProfile => {}
            SolverFindingState::UnsupportedRegime
            | SolverFindingState::UnsupportedObservable
            | SolverFindingState::UnsupportedUse
            | SolverFindingState::OutOfDomain
            | SolverFindingState::FailedExecution
            | SolverFindingState::NonConverged => {
                blocked = true;
                diagnostics.push(format!(
                    "solver run {} blocked: {:?}",
                    run.run_id, finding.state
                ));
            }
            SolverFindingState::ApplicabilityUnknown
            | SolverFindingState::ConvergenceNotReported
            | SolverFindingState::NotExecuted
            | SolverFindingState::MissingOutput => {
                indeterminate = true;
                diagnostics.push(format!(
                    "solver run {} incomplete: {:?}",
                    run.run_id, finding.state
                ));
            }
        }
        solver_findings.push(finding);
    }

    let disposition = if blocked {
        CandidateDisposition::Blocked
    } else if indeterminate {
        CandidateDisposition::Indeterminate
    } else {
        CandidateDisposition::EligibleForFurtherEngineeringReview
    };

    Ok(DecisionBuild {
        compiler_version: COMPILER_VERSION,
        candidate_id: candidate.candidate_id.clone(),
        candidate_revision: candidate.candidate_revision.clone(),
        objective_id: candidate.objective_id.clone(),
        service_metric: candidate.service_metric,
        disposition,
        gate_findings,
        solver_findings,
        dependence_findings: pairwise_dependence_findings(&candidate.solver_runs),
        diagnostics,
        claim_ceiling: ClaimCeiling::FurtherEngineeringReviewOnly,
    })
}

fn evaluate_solver_run(candidate: &CandidateInput, run: &SolverRun) -> SolverFinding {
    let finding = |state| SolverFinding {
        run_id: run.run_id.clone(),
        solver_id: run.solver_id.clone(),
        state,
    };

    if !run.supported_regimes.contains(&candidate.required_regime) {
        return finding(SolverFindingState::UnsupportedRegime);
    }

    if !run.supported_uses.contains(&candidate.required_model_use) {
        return finding(SolverFindingState::UnsupportedUse);
    }

    if candidate
        .required_observables
        .iter()
        .any(|required| !run.supported_observables.contains(required))
    {
        return finding(SolverFindingState::UnsupportedObservable);
    }

    match run.applicability {
        Applicability::OutOfDomain => return finding(SolverFindingState::OutOfDomain),
        Applicability::Unknown => return finding(SolverFindingState::ApplicabilityUnknown),
        Applicability::InDeclaredDomain => {}
    }

    match run.exit_code {
        None => return finding(SolverFindingState::NotExecuted),
        Some(code) if code != 0 => return finding(SolverFindingState::FailedExecution),
        Some(_) => {}
    }

    match run.convergence {
        Convergence::NonConverged => return finding(SolverFindingState::NonConverged),
        Convergence::NotReported => {
            return finding(SolverFindingState::ConvergenceNotReported);
        }
        Convergence::Converged => {}
    }

    if candidate
        .required_observables
        .iter()
        .any(|required| !run.output_observables.contains(required))
    {
        return finding(SolverFindingState::MissingOutput);
    }

    finding(SolverFindingState::CompletedWithinDeclaredProfile)
}

fn validate_candidate(candidate: &CandidateInput) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    require_nonempty(&candidate.candidate_id, "candidate_id", &mut errors);
    require_nonempty(
        &candidate.candidate_revision,
        "candidate_revision",
        &mut errors,
    );
    require_nonempty(&candidate.objective_id, "objective_id", &mut errors);
    require_nonempty(&candidate.required_regime, "required_regime", &mut errors);

    if !is_sha256(&candidate.geometry_sha256) {
        errors.push(ValidationError::InvalidSha256("geometry_sha256"));
    }

    if let Some(boundary_ref) = candidate.service_boundary_ref.as_deref()
        && !boundary_ref.trim().is_empty()
    {
        match candidate
            .evidence
            .iter()
            .find(|item| item.id == boundary_ref)
        {
            None => errors.push(ValidationError::MissingServiceBoundaryEvidence(
                boundary_ref.to_string(),
            )),
            Some(evidence) => {
                if evidence.scope_ref != candidate.service_metric.boundary_scope_ref() {
                    errors.push(ValidationError::ServiceBoundaryScopeMismatch(
                        boundary_ref.to_string(),
                    ));
                }
                if evidence.subject_ref != candidate.candidate_id
                    || evidence.subject_revision != candidate.candidate_revision
                {
                    errors.push(ValidationError::ServiceBoundarySubjectMismatch(
                        boundary_ref.to_string(),
                    ));
                }
            }
        }
    }

    let mut evidence_ids = BTreeSet::new();
    for evidence in &candidate.evidence {
        require_nonempty(&evidence.id, "evidence.id", &mut errors);
        require_nonempty(&evidence.subject_ref, "evidence.subject_ref", &mut errors);
        require_nonempty(
            &evidence.subject_revision,
            "evidence.subject_revision",
            &mut errors,
        );
        require_nonempty(&evidence.scope_ref, "evidence.scope_ref", &mut errors);
        if !is_sha256(&evidence.content_sha256) {
            errors.push(ValidationError::InvalidSha256("evidence.content_sha256"));
        }
        if !evidence_ids.insert(evidence.id.as_str()) {
            errors.push(ValidationError::DuplicateEvidenceId(evidence.id.clone()));
        }
    }

    let mut gate_kinds = BTreeSet::new();
    for gate in &candidate.hard_gates {
        if !gate_kinds.insert(gate.kind) {
            errors.push(ValidationError::DuplicateGate(gate.kind));
        }
        match gate.state {
            GateState::Pass if gate.evidence_ids.is_empty() => {
                errors.push(ValidationError::PassingGateWithoutEvidence(gate.kind));
            }
            GateState::Fail | GateState::Unknown
                if gate
                    .reason_code
                    .as_deref()
                    .is_none_or(|reason| reason.trim().is_empty()) =>
            {
                errors.push(ValidationError::NonPassingGateWithoutReason(gate.kind));
            }
            _ => {}
        }

        let mut gate_refs = BTreeSet::new();
        for evidence_id in &gate.evidence_ids {
            if !gate_refs.insert(evidence_id.as_str()) {
                errors.push(ValidationError::DanglingGateEvidence {
                    gate: gate.kind,
                    evidence_id: evidence_id.clone(),
                });
                continue;
            }

            let Some(evidence) = candidate
                .evidence
                .iter()
                .find(|item| item.id == *evidence_id)
            else {
                errors.push(ValidationError::DanglingGateEvidence {
                    gate: gate.kind,
                    evidence_id: evidence_id.clone(),
                });
                continue;
            };

            if evidence.scope_ref != gate.kind.scope_ref() {
                errors.push(ValidationError::GateEvidenceScopeMismatch {
                    gate: gate.kind,
                    evidence_id: evidence_id.clone(),
                });
            }
            if evidence.subject_ref != candidate.candidate_id
                || evidence.subject_revision != candidate.candidate_revision
            {
                errors.push(ValidationError::GateEvidenceSubjectMismatch {
                    gate: gate.kind,
                    evidence_id: evidence_id.clone(),
                });
            }
        }
    }

    let mut claim_ids = BTreeSet::new();
    for claim in &candidate.claims {
        if !claim_ids.insert(claim.claim_id.as_str()) {
            errors.push(ValidationError::DuplicateClaimId(claim.claim_id.clone()));
        }
        let Some(evidence) = candidate
            .evidence
            .iter()
            .find(|item| item.id == claim.evidence_id)
        else {
            errors.push(ValidationError::DanglingClaimEvidence {
                claim_id: claim.claim_id.clone(),
                evidence_id: claim.evidence_id.clone(),
            });
            continue;
        };

        if claim.subject_ref != evidence.subject_ref {
            errors.push(ValidationError::ClaimSubjectMismatch(
                claim.claim_id.clone(),
            ));
        }
        if claim.subject_revision != evidence.subject_revision {
            errors.push(ValidationError::ClaimRevisionMismatch(
                claim.claim_id.clone(),
            ));
        }
        if !evidence.class.supports(claim.requested_class) {
            errors.push(ValidationError::ClaimClassMismatch(claim.claim_id.clone()));
        }
    }

    let mut run_ids = BTreeSet::new();
    for run in &candidate.solver_runs {
        require_nonempty(&run.run_id, "solver_run.run_id", &mut errors);
        require_nonempty(&run.solver_id, "solver_run.solver_id", &mut errors);
        require_nonempty(
            &run.solver_revision,
            "solver_run.solver_revision",
            &mut errors,
        );
        require_nonempty(
            &run.candidate_revision,
            "solver_run.candidate_revision",
            &mut errors,
        );
        if run.candidate_revision != candidate.candidate_revision {
            errors.push(ValidationError::SolverCandidateRevisionMismatch(
                run.run_id.clone(),
            ));
        }
        if !is_sha256(&run.geometry_sha256) {
            errors.push(ValidationError::InvalidSha256("solver_run.geometry_sha256"));
        } else if run.geometry_sha256 != candidate.geometry_sha256 {
            errors.push(ValidationError::SolverGeometryMismatch(run.run_id.clone()));
        }
        if !run_ids.insert(run.run_id.as_str()) {
            errors.push(ValidationError::DuplicateSolverRunId(run.run_id.clone()));
        }
        if !is_sha256(&run.input_sha256) {
            errors.push(ValidationError::InvalidSha256("solver_run.input_sha256"));
        }
        if !is_sha256(&run.output_sha256) {
            errors.push(ValidationError::InvalidSha256("solver_run.output_sha256"));
        }
    }

    let mut required_observables = BTreeSet::new();
    for observable in &candidate.required_observables {
        if !required_observables.insert(observable.as_str()) {
            errors.push(ValidationError::DuplicateRequiredObservable(
                observable.clone(),
            ));
        }
    }

    errors
}

fn pairwise_dependence_findings(runs: &[SolverRun]) -> Vec<DependenceFinding> {
    let mut findings = Vec::new();
    for (first_index, first_run) in runs.iter().enumerate() {
        for second_run in runs.iter().skip(first_index + 1) {
            let a = &first_run.dependencies;
            let b = &second_run.dependencies;
            let shared = same_declared_group(&a.dataset_group, &b.dataset_group)
                || same_declared_group(&a.code_group, &b.code_group)
                || same_declared_group(&a.assumption_group, &b.assumption_group);
            let unknown = is_unknown_group(&a.dataset_group)
                || is_unknown_group(&b.dataset_group)
                || is_unknown_group(&a.code_group)
                || is_unknown_group(&b.code_group)
                || is_unknown_group(&a.assumption_group)
                || is_unknown_group(&b.assumption_group);

            let disposition = if shared {
                PairwiseDependence::CommonModeSharedDependency
            } else if unknown {
                PairwiseDependence::UnknownDependencyIdentity
            } else {
                PairwiseDependence::IndependentUnderDeclaredFactors
            };

            findings.push(DependenceFinding {
                first_run_id: first_run.run_id.clone(),
                second_run_id: second_run.run_id.clone(),
                disposition,
            });
        }
    }
    findings
}

fn same_declared_group(first: &Option<String>, second: &Option<String>) -> bool {
    matches!((first, second), (Some(a), Some(b)) if !a.trim().is_empty() && a == b)
}

fn is_unknown_group(group: &Option<String>) -> bool {
    group.as_deref().is_none_or(|value| value.trim().is_empty())
}

/// Return the changed nodes and all transitive downstream dependants.
///
/// The result is sorted and deterministic. Cycles do not cause unbounded
/// traversal. This is impact analysis, not authority revocation.
pub fn affected_by_change(changed_nodes: &[String], edges: &[DependencyEdge]) -> Vec<String> {
    let mut affected = BTreeSet::new();
    let mut queue = VecDeque::new();

    for node in changed_nodes {
        if affected.insert(node.clone()) {
            queue.push_back(node.clone());
        }
    }

    while let Some(upstream) = queue.pop_front() {
        for edge in edges.iter().filter(|edge| edge.upstream == upstream) {
            if affected.insert(edge.downstream.clone()) {
                queue.push_back(edge.downstream.clone());
            }
        }
    }

    affected.into_iter().collect()
}

fn require_nonempty(value: &str, field: &'static str, errors: &mut Vec<ValidationError>) {
    if value.trim().is_empty() {
        errors.push(ValidationError::EmptyField(field));
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn candidate() -> CandidateInput {
        let candidate_id = "candidate-a".to_string();
        let candidate_revision = "design-r7".to_string();
        let service_metric = ServiceMetric::NetElectricExportPowerMegawatt;
        let mut evidence: Vec<EvidenceRef> = HardGateKind::REQUIRED
            .iter()
            .enumerate()
            .map(|(index, kind)| EvidenceRef {
                id: format!("evidence-{index}"),
                subject_ref: candidate_id.clone(),
                subject_revision: candidate_revision.clone(),
                scope_ref: kind.scope_ref().to_string(),
                content_sha256: sha('a'),
                class: EvidenceClass::SyntheticFixture,
            })
            .collect();

        evidence.push(EvidenceRef {
            id: "grid-export-boundary-v1".to_string(),
            subject_ref: candidate_id.clone(),
            subject_revision: candidate_revision.clone(),
            scope_ref: service_metric.boundary_scope_ref().to_string(),
            content_sha256: sha('f'),
            class: EvidenceClass::SyntheticFixture,
        });

        let hard_gates = HardGateKind::REQUIRED
            .iter()
            .enumerate()
            .map(|(index, kind)| HardGate {
                kind: *kind,
                state: GateState::Pass,
                reason_code: None,
                evidence_ids: vec![format!("evidence-{index}")],
            })
            .collect();

        CandidateInput {
            candidate_id,
            candidate_revision,
            geometry_sha256: sha('b'),
            objective_id: "service-objective-v1".to_string(),
            service_metric,
            service_boundary_ref: Some("grid-export-boundary-v1".to_string()),
            required_regime: "tokamak-equilibrium".to_string(),
            required_model_use: ModelUse::SingleStepPrediction,
            required_observables: vec!["psi".to_string(), "q95".to_string()],
            evidence,
            claims: Vec::new(),
            hard_gates,
            solver_runs: vec![solver_run("run-a", 'c', 'd')],
        }
    }

    fn solver_run(run_id: &str, input_digest: char, output_digest: char) -> SolverRun {
        SolverRun {
            run_id: run_id.to_string(),
            solver_id: "reference-solver".to_string(),
            solver_revision: "1.2.3+commit".to_string(),
            candidate_revision: "design-r7".to_string(),
            geometry_sha256: sha('b'),
            supported_regimes: vec!["tokamak-equilibrium".to_string()],
            supported_observables: vec!["psi".to_string(), "q95".to_string()],
            supported_uses: vec![ModelUse::SingleStepPrediction],
            applicability: Applicability::InDeclaredDomain,
            exit_code: Some(0),
            convergence: Convergence::Converged,
            output_observables: vec!["psi".to_string(), "q95".to_string()],
            input_sha256: sha(input_digest),
            output_sha256: sha(output_digest),
            dependencies: DeclaredModelDependencies {
                dataset_group: Some(format!("dataset-{run_id}")),
                code_group: Some(format!("code-{run_id}")),
                assumption_group: Some(format!("assumptions-{run_id}")),
            },
        }
    }

    fn has_error(errors: &[ValidationError], expected: ValidationError) -> bool {
        errors.contains(&expected)
    }

    #[test]
    fn complete_candidate_is_only_eligible_for_further_review() {
        let build = compile_candidate(&candidate()).unwrap();
        assert_eq!(
            build.disposition,
            CandidateDisposition::EligibleForFurtherEngineeringReview
        );
        assert_eq!(
            build.claim_ceiling,
            ClaimCeiling::FurtherEngineeringReviewOnly
        );
    }

    #[test]
    fn decision_build_preserves_typed_service_metric() {
        let mut source = candidate();
        source.service_metric = ServiceMetric::LifetimeNetElectricEnergyMegawattHour;
        let service_evidence = source
            .evidence
            .iter_mut()
            .find(|evidence| evidence.id == "grid-export-boundary-v1")
            .unwrap();
        service_evidence.scope_ref = source.service_metric.boundary_scope_ref().to_string();
        let build = compile_candidate(&source).unwrap();
        assert_eq!(
            build.service_metric,
            ServiceMetric::LifetimeNetElectricEnergyMegawattHour
        );
    }

    #[test]
    fn missing_service_boundary_is_indeterminate_not_zero() {
        let mut source = candidate();
        source.service_boundary_ref = None;
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Indeterminate);
    }

    #[test]
    fn service_boundary_reference_must_resolve_to_evidence() {
        let mut source = candidate();
        source.service_boundary_ref = Some("missing-boundary-record".to_string());
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::MissingServiceBoundaryEvidence("missing-boundary-record".to_string())
        ));
    }

    #[test]
    fn service_boundary_evidence_must_match_typed_metric_scope() {
        let mut source = candidate();
        let service_evidence = source
            .evidence
            .iter_mut()
            .find(|evidence| evidence.id == "grid-export-boundary-v1")
            .unwrap();
        service_evidence.scope_ref = "gross-thermal-boundary".to_string();
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::ServiceBoundaryScopeMismatch("grid-export-boundary-v1".to_string())
        ));
    }

    #[test]
    fn service_metric_cannot_reuse_a_different_metric_boundary() {
        let mut source = candidate();
        source.service_metric = ServiceMetric::LifetimeUsefulThermalEnergyMegawattHour;
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::ServiceBoundaryScopeMismatch("grid-export-boundary-v1".to_string())
        ));
    }

    #[test]
    fn missing_required_gate_is_indeterminate() {
        let mut source = candidate();
        source
            .hard_gates
            .retain(|gate| gate.kind != HardGateKind::EnergyBoundary);
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Indeterminate);
    }

    #[test]
    fn failed_hard_gate_blocks_candidate() {
        let mut source = candidate();
        let gate = source
            .hard_gates
            .iter_mut()
            .find(|gate| gate.kind == HardGateKind::ReplacementAccess)
            .unwrap();
        gate.state = GateState::Fail;
        gate.reason_code = Some("replacement-envelope-failed".to_string());
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Blocked);
    }

    #[test]
    fn unknown_material_gate_is_indeterminate() {
        let mut source = candidate();
        let gate = source
            .hard_gates
            .iter_mut()
            .find(|gate| gate.kind == HardGateKind::MaterialApplicability)
            .unwrap();
        gate.state = GateState::Unknown;
        gate.reason_code = Some("material-domain-unknown".to_string());
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Indeterminate);
    }

    #[test]
    fn unsupported_regime_blocks_candidate() {
        let mut source = candidate();
        source.solver_runs[0].supported_regimes = vec!["thermal-arc".to_string()];
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Blocked);
    }

    #[test]
    fn unsupported_required_observable_blocks_candidate() {
        let mut source = candidate();
        source.solver_runs[0].supported_observables = vec!["psi".to_string()];
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Blocked);
    }

    #[test]
    fn one_step_model_is_not_implicitly_qualified_for_open_loop_rollout() {
        let mut source = candidate();
        source.required_model_use = ModelUse::OpenLoopRollout;
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Blocked);
        assert_eq!(
            build.solver_findings[0].state,
            SolverFindingState::UnsupportedUse
        );
    }

    #[test]
    fn out_of_domain_solver_blocks_candidate() {
        let mut source = candidate();
        source.solver_runs[0].applicability = Applicability::OutOfDomain;
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Blocked);
    }

    #[test]
    fn unknown_solver_applicability_is_indeterminate() {
        let mut source = candidate();
        source.solver_runs[0].applicability = Applicability::Unknown;
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Indeterminate);
    }

    #[test]
    fn zero_exit_with_nonconvergence_is_still_blocked() {
        let mut source = candidate();
        source.solver_runs[0].exit_code = Some(0);
        source.solver_runs[0].convergence = Convergence::NonConverged;
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Blocked);
    }

    #[test]
    fn nonzero_process_exit_blocks_candidate() {
        let mut source = candidate();
        source.solver_runs[0].exit_code = Some(2);
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Blocked);
    }

    #[test]
    fn missing_required_output_is_indeterminate() {
        let mut source = candidate();
        source.solver_runs[0].output_observables = vec!["psi".to_string()];
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Indeterminate);
    }

    #[test]
    fn no_solver_run_is_indeterminate() {
        let mut source = candidate();
        source.solver_runs.clear();
        let build = compile_candidate(&source).unwrap();
        assert_eq!(build.disposition, CandidateDisposition::Indeterminate);
    }

    #[test]
    fn solver_run_must_bind_exact_candidate_revision() {
        let mut source = candidate();
        source.solver_runs[0].candidate_revision = "design-r6".to_string();
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::SolverCandidateRevisionMismatch("run-a".to_string())
        ));
    }

    #[test]
    fn solver_run_must_bind_exact_geometry_identity() {
        let mut source = candidate();
        source.solver_runs[0].geometry_sha256 = sha('e');
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::SolverGeometryMismatch("run-a".to_string())
        ));
    }

    #[test]
    fn synthetic_fixture_cannot_support_measured_claim() {
        let mut source = candidate();
        source.claims.push(ClaimRequest {
            claim_id: "claim-measured-1".to_string(),
            subject_ref: source.candidate_id.clone(),
            subject_revision: source.candidate_revision.clone(),
            evidence_id: "evidence-1".to_string(),
            requested_class: ClaimClass::Measured,
        });
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::ClaimClassMismatch("claim-measured-1".to_string())
        ));
    }

    #[test]
    fn claim_must_bind_exact_subject_revision() {
        let mut source = candidate();
        source.claims.push(ClaimRequest {
            claim_id: "claim-synthetic-1".to_string(),
            subject_ref: source.candidate_id.clone(),
            subject_revision: "stale-design-r6".to_string(),
            evidence_id: "evidence-1".to_string(),
            requested_class: ClaimClass::Synthetic,
        });
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::ClaimRevisionMismatch("claim-synthetic-1".to_string())
        ));
    }

    #[test]
    fn malformed_geometry_digest_is_rejected() {
        let mut source = candidate();
        source.geometry_sha256 = "not-a-digest".to_string();
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::InvalidSha256("geometry_sha256")
        ));
    }

    #[test]
    fn dangling_gate_evidence_is_rejected() {
        let mut source = candidate();
        source.hard_gates[0].evidence_ids = vec!["missing-evidence".to_string()];
        let errors = compile_candidate(&source).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationError::DanglingGateEvidence {
                gate: HardGateKind::EnergyBoundary,
                evidence_id
            } if evidence_id == "missing-evidence"
        )));
    }

    #[test]
    fn pass_with_no_evidence_is_rejected() {
        let mut source = candidate();
        source.hard_gates[0].evidence_ids.clear();
        let errors = compile_candidate(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationError::PassingGateWithoutEvidence(HardGateKind::EnergyBoundary)
        ));
    }

    #[test]
    fn shared_dataset_dependency_is_common_mode_not_independent() {
        let mut source = candidate();
        let mut second = solver_run("run-b", 'e', 'f');
        second.dependencies.dataset_group =
            source.solver_runs[0].dependencies.dataset_group.clone();
        source.solver_runs.push(second);
        let build = compile_candidate(&source).unwrap();
        assert_eq!(
            build.dependence_findings[0].disposition,
            PairwiseDependence::CommonModeSharedDependency
        );
    }

    #[test]
    fn missing_dependency_identity_is_unknown_not_independent() {
        let mut source = candidate();
        let mut second = solver_run("run-b", 'e', 'f');
        second.dependencies.assumption_group = None;
        source.solver_runs.push(second);
        let build = compile_candidate(&source).unwrap();
        assert_eq!(
            build.dependence_findings[0].disposition,
            PairwiseDependence::UnknownDependencyIdentity
        );
    }

    #[test]
    fn distinct_declared_dependency_groups_are_only_conditionally_independent() {
        let mut source = candidate();
        source.solver_runs.push(solver_run("run-b", 'e', 'f'));
        let build = compile_candidate(&source).unwrap();
        assert_eq!(
            build.dependence_findings[0].disposition,
            PairwiseDependence::IndependentUnderDeclaredFactors
        );
    }

    #[test]
    fn empty_dependency_group_is_unknown_not_independent() {
        let mut source = candidate();
        let mut second = solver_run("run-b", 'e', 'f');
        second.dependencies.dataset_group = Some(String::new());
        source.solver_runs.push(second);
        let build = compile_candidate(&source).unwrap();
        assert_eq!(
            build.dependence_findings[0].disposition,
            PairwiseDependence::UnknownDependencyIdentity
        );
    }

    #[test]
    fn dependency_change_invalidates_transitive_dependants() {
        let edges = vec![
            DependencyEdge {
                upstream: "material-lot-1".to_string(),
                downstream: "component-design-7".to_string(),
            },
            DependencyEdge {
                upstream: "component-design-7".to_string(),
                downstream: "candidate-42".to_string(),
            },
            DependencyEdge {
                upstream: "candidate-42".to_string(),
                downstream: "decision-build-9".to_string(),
            },
        ];
        let affected = affected_by_change(&["material-lot-1".to_string()], &edges);
        assert_eq!(
            affected,
            vec![
                "candidate-42".to_string(),
                "component-design-7".to_string(),
                "decision-build-9".to_string(),
                "material-lot-1".to_string()
            ]
        );
    }

    #[test]
    fn frozen_feccomp_corpus_matches_actual_compiler_dispositions() {
        const CORPUS: &str = include_str!("../fixtures/fecomp-cases-v1.tsv");

        let mut checked = 0usize;
        for line in CORPUS.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') || line == "case_id|expected|mutation" {
                continue;
            }

            let fields: Vec<&str> = line.split('|').collect();
            assert_eq!(fields.len(), 3, "malformed fixture row: {line}");
            let [case_id, expected, mutation] = [fields[0], fields[1], fields[2]];
            let mut source = candidate();

            match mutation {
                "none" => {}
                "missing_service_boundary" => source.service_boundary_ref = None,
                "unsupported_regime" => {
                    source.solver_runs[0].supported_regimes = vec!["thermal-arc".to_string()];
                }
                "out_of_domain" => {
                    source.solver_runs[0].applicability = Applicability::OutOfDomain;
                }
                "unknown_applicability" => {
                    source.solver_runs[0].applicability = Applicability::Unknown;
                }
                "nonconverged" => {
                    source.solver_runs[0].convergence = Convergence::NonConverged;
                }
                "nonzero_exit" => source.solver_runs[0].exit_code = Some(2),
                "missing_observable" => {
                    source.solver_runs[0].output_observables = vec!["psi".to_string()];
                }
                "missing_energy_gate" => {
                    source
                        .hard_gates
                        .retain(|gate| gate.kind != HardGateKind::EnergyBoundary);
                }
                "replacement_failed" => {
                    let gate = source
                        .hard_gates
                        .iter_mut()
                        .find(|gate| gate.kind == HardGateKind::ReplacementAccess)
                        .unwrap();
                    gate.state = GateState::Fail;
                    gate.reason_code = Some("replacement-envelope-failed".to_string());
                }
                "material_unknown" => {
                    let gate = source
                        .hard_gates
                        .iter_mut()
                        .find(|gate| gate.kind == HardGateKind::MaterialApplicability)
                        .unwrap();
                    gate.state = GateState::Unknown;
                    gate.reason_code = Some("material-domain-unknown".to_string());
                }
                "claim_class_mismatch" => source.claims.push(ClaimRequest {
                    claim_id: "fixture-claim".to_string(),
                    subject_ref: source.candidate_id.clone(),
                    subject_revision: source.candidate_revision.clone(),
                    evidence_id: "evidence-1".to_string(),
                    requested_class: ClaimClass::Measured,
                }),
                "claim_revision_mismatch" => source.claims.push(ClaimRequest {
                    claim_id: "fixture-claim".to_string(),
                    subject_ref: source.candidate_id.clone(),
                    subject_revision: "stale-design-r6".to_string(),
                    evidence_id: "evidence-1".to_string(),
                    requested_class: ClaimClass::Synthetic,
                }),
                "shared_dependency" => {
                    let mut second = solver_run("run-b", 'e', 'f');
                    second.dependencies.dataset_group =
                        source.solver_runs[0].dependencies.dataset_group.clone();
                    source.solver_runs.push(second);
                }
                "unknown_dependency" => {
                    let mut second = solver_run("run-b", 'e', 'f');
                    second.dependencies.assumption_group = None;
                    source.solver_runs.push(second);
                }
                "no_solver_runs" => source.solver_runs.clear(),
                "bad_geometry_digest" => source.geometry_sha256 = "not-a-digest".to_string(),
                "dangling_evidence" => {
                    source.hard_gates[0].evidence_ids = vec!["missing-evidence".to_string()];
                }
                "dependency_invalidation" => {
                    let edges = vec![
                        DependencyEdge {
                            upstream: "material".to_string(),
                            downstream: "component".to_string(),
                        },
                        DependencyEdge {
                            upstream: "component".to_string(),
                            downstream: "decision".to_string(),
                        },
                    ];
                    let affected = affected_by_change(&["material".to_string()], &edges);
                    assert_eq!(
                        affected,
                        vec![
                            "component".to_string(),
                            "decision".to_string(),
                            "material".to_string()
                        ]
                    );
                }
                "dependency_cycle" => {
                    let edges = vec![
                        DependencyEdge {
                            upstream: "a".to_string(),
                            downstream: "b".to_string(),
                        },
                        DependencyEdge {
                            upstream: "b".to_string(),
                            downstream: "c".to_string(),
                        },
                        DependencyEdge {
                            upstream: "c".to_string(),
                            downstream: "a".to_string(),
                        },
                    ];
                    let affected = affected_by_change(&["a".to_string()], &edges);
                    assert_eq!(
                        affected,
                        vec!["a".to_string(), "b".to_string(), "c".to_string()]
                    );
                }
                "solver_revision_mismatch" => {
                    source.solver_runs[0].candidate_revision = "design-r6".to_string();
                }
                "solver_geometry_mismatch" => {
                    source.solver_runs[0].geometry_sha256 = sha('e');
                }
                "unsupported_observable" => {
                    source.solver_runs[0].supported_observables = vec!["psi".to_string()];
                }
                "unsupported_model_use" => {
                    source.required_model_use = ModelUse::OpenLoopRollout;
                }
                "service_boundary_unresolved" => {
                    source.service_boundary_ref = Some("missing-boundary-record".to_string());
                }
                "service_boundary_scope" => {
                    source.evidence[0].scope_ref = "component-geometry".to_string();
                }
                other => panic!("unknown fixture mutation {other} in {case_id}"),
            }

            let actual = match compile_candidate(&source) {
                Ok(build) => match build.disposition {
                    CandidateDisposition::EligibleForFurtherEngineeringReview => "REVIEW",
                    CandidateDisposition::Blocked => "BLOCKED",
                    CandidateDisposition::Indeterminate => "INDETERMINATE",
                },
                Err(_) => "REJECT",
            };

            assert_eq!(actual, expected, "fixture case {case_id} ({mutation})");
            checked += 1;
        }

        assert_eq!(checked, 26, "all frozen FECOMP-0 cases must execute");
    }

    #[test]
    fn dependency_cycles_terminate_and_remain_deterministic() {
        let edges = vec![
            DependencyEdge {
                upstream: "a".to_string(),
                downstream: "b".to_string(),
            },
            DependencyEdge {
                upstream: "b".to_string(),
                downstream: "c".to_string(),
            },
            DependencyEdge {
                upstream: "c".to_string(),
                downstream: "a".to_string(),
            },
        ];
        let affected = affected_by_change(&["a".to_string()], &edges);
        assert_eq!(
            affected,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }
}
