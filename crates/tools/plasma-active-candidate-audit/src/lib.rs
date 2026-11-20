use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsLane {
    #[serde(rename = "landau_damping_1d_electrostatic")]
    LandauDamping1dElectrostatic,
    #[serde(rename = "magnetosheath_2d3v_pressure_tensor")]
    Magnetosheath2d3vPressureTensor,
    #[serde(rename = "icf_nonlocal_electron_heat_transport")]
    IcfNonlocalElectronHeatTransport,
    #[serde(rename = "pic_field_solver_warm_start")]
    PicFieldSolverWarmStart,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KineticMethod {
    #[serde(rename = "PIC")]
    Pic,
    #[serde(rename = "Vlasov")]
    Vlasov,
    #[serde(rename = "VFP")]
    Vfp,
    #[serde(rename = "other_declared_kinetic")]
    OtherDeclaredKinetic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitRole {
    Training,
    DevelopmentValidation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsState {
    Verified,
    PermissionPending,
    Restricted,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStatus {
    Proposed,
    Selected,
    Rejected,
    Unaffordable,
    Cancelled,
    Executed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionReason {
    Random,
    SpaceFilling,
    CommitteeDisagreement,
    CalibratedUncertainty,
    PhysicalResidual,
    AdmissibilityFailure,
    SupportDistance,
    MultifidelityValue,
    ParetoDiversity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    pub geometry_id: String,
    pub dimension: usize,
    pub grid_size: Vec<usize>,
    pub cell_spacing: Option<Vec<f64>>,
    pub domain_size: Option<Vec<f64>>,
    pub coordinate_frame_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resolution {
    pub spatial_resolution_id: String,
    pub time_step_seconds: f64,
    pub output_cadence_seconds: Option<f64>,
    pub downsampling: Option<BTreeMap<String, Value>>,
    pub phase_space_resolution: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KineticFidelity {
    pub method: KineticMethod,
    pub fidelity_id: String,
    pub particles_per_cell: Option<u64>,
    pub phase_space_resolution: Option<BTreeMap<String, Value>>,
    pub noise_or_discretisation_notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverIdentity {
    pub repository: String,
    pub revision: String,
    pub solver_name: String,
    pub build_id: String,
    pub compiler_runtime: Option<String>,
    pub numerical_options: BTreeMap<String, Value>,
    pub random_seed_policy: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostEstimate {
    pub measurement_basis: String,
    pub cpu_core_hours: f64,
    pub gpu_hours: f64,
    pub wall_seconds: f64,
    pub peak_memory_bytes: u64,
    pub data_volume_bytes: u64,
    pub cost_estimator_revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceReceipt {
    pub receipt_id: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Acquisition {
    pub policy_id: String,
    pub policy_revision: String,
    pub reason_codes: Vec<AcquisitionReason>,
    pub evidence_receipts: Vec<EvidenceReceipt>,
    pub signal_values: BTreeMap<String, f64>,
    pub selection_rationale: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRun {
    pub schema_version: String,
    pub candidate_id: String,
    pub parent_experiment_id: String,
    pub physics_lane: PhysicsLane,
    pub governing_equation_version: String,
    pub physical_parameters: BTreeMap<String, Value>,
    pub initial_condition_id: String,
    pub boundary_condition_id: String,
    pub geometry: Geometry,
    pub resolution: Resolution,
    pub kinetic_fidelity: KineticFidelity,
    pub solver: SolverIdentity,
    pub random_seed: u64,
    pub target_observables: Vec<String>,
    pub cost_estimate: CostEstimate,
    pub acquisition: Acquisition,
    pub split_role: SplitRole,
    pub rights_state: RightsState,
    pub selection_status: SelectionStatus,
    pub execution_receipt_id: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditDisposition {
    Admissible,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditFinding {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateAuditBody {
    pub schema_version: String,
    pub candidate_id: String,
    pub disposition: AuditDisposition,
    pub request_sha256: String,
    pub selection_status: SelectionStatus,
    pub split_role: SplitRole,
    pub rights_state: RightsState,
    pub findings: Vec<AuditFinding>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateAuditReport {
    pub body: CandidateAuditBody,
    pub receipt_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn numeric_values_positive(values: &BTreeMap<String, Value>) -> bool {
    !values.is_empty()
        && values.values().all(|value| match value {
            Value::Number(number) => number.as_f64().is_some_and(positive),
            Value::Array(array) => !array.is_empty()
                && array.iter().all(|item| {
                    item.as_f64().is_some_and(positive)
                        || item.as_u64().is_some_and(|number| number > 0)
                }),
            _ => false,
        })
}

fn candidate_findings(candidate: &CandidateRun) -> Vec<AuditFinding> {
    let mut findings = Vec::new();
    let mut add = |code: &str, detail: &str| {
        findings.push(AuditFinding {
            code: code.to_owned(),
            detail: detail.to_owned(),
        });
    };

    if candidate.schema_version != SCHEMA_VERSION {
        add("unsupported_schema_version", "Candidate schema version is not 0.1.0.");
    }
    for (name, value) in [
        ("candidate_id", candidate.candidate_id.as_str()),
        ("parent_experiment_id", candidate.parent_experiment_id.as_str()),
        ("governing_equation_version", candidate.governing_equation_version.as_str()),
        ("initial_condition_id", candidate.initial_condition_id.as_str()),
        ("boundary_condition_id", candidate.boundary_condition_id.as_str()),
        ("geometry_id", candidate.geometry.geometry_id.as_str()),
        ("spatial_resolution_id", candidate.resolution.spatial_resolution_id.as_str()),
        ("fidelity_id", candidate.kinetic_fidelity.fidelity_id.as_str()),
        ("solver_name", candidate.solver.solver_name.as_str()),
        ("solver_build_id", candidate.solver.build_id.as_str()),
        ("acquisition_policy_id", candidate.acquisition.policy_id.as_str()),
    ] {
        if !non_empty(value) {
            add("empty_required_identity", &format!("{name} must be non-empty."));
        }
    }

    if candidate.physical_parameters.is_empty() {
        add("missing_physical_parameters", "At least one physical parameter is required.");
    }
    if candidate.geometry.dimension == 0 || candidate.geometry.dimension > 3 {
        add("invalid_geometry_dimension", "Spatial dimension must be between 1 and 3.");
    } else {
        let dimension = candidate.geometry.dimension;
        if candidate.geometry.grid_size.len() != dimension
            || candidate.geometry.grid_size.iter().any(|size| *size < 2)
        {
            add(
                "grid_shape_dimension_mismatch",
                "grid_size length must equal spatial dimension and each axis must have at least two cells.",
            );
        }
        if candidate.geometry.cell_spacing.as_ref().is_some_and(|values| {
            values.len() != dimension || values.iter().any(|value| !positive(*value))
        }) {
            add("invalid_cell_spacing", "cell_spacing must have one positive finite value per dimension.");
        }
        if candidate.geometry.domain_size.as_ref().is_some_and(|values| {
            values.len() != dimension || values.iter().any(|value| !positive(*value))
        }) {
            add("invalid_domain_size", "domain_size must have one positive finite value per dimension.");
        }
    }
    if candidate.geometry.coordinate_frame_id.as_ref().is_some_and(|value| !non_empty(value)) {
        add("empty_coordinate_frame", "coordinate_frame_id cannot be empty when supplied.");
    }

    if !positive(candidate.resolution.time_step_seconds) {
        add("invalid_time_step", "time_step_seconds must be finite and positive.");
    }
    if candidate.resolution.output_cadence_seconds.is_some_and(|value| !positive(value)) {
        add("invalid_output_cadence", "output_cadence_seconds must be finite and positive.");
    }

    match &candidate.kinetic_fidelity.method {
        KineticMethod::Pic => {
            if candidate.kinetic_fidelity.particles_per_cell.is_none_or(|number| number == 0) {
                add("pic_particle_count_missing", "PIC candidates require particles_per_cell >= 1.");
            }
        }
        KineticMethod::Vlasov | KineticMethod::Vfp => {
            if candidate
                .kinetic_fidelity
                .phase_space_resolution
                .as_ref()
                .is_none_or(|values| !numeric_values_positive(values))
            {
                add(
                    "phase_space_resolution_missing",
                    "Vlasov/VFP candidates require a non-empty positive phase_space_resolution map.",
                );
            }
        }
        KineticMethod::OtherDeclaredKinetic => {
            if candidate.kinetic_fidelity.particles_per_cell.is_none()
                && candidate
                    .kinetic_fidelity
                    .phase_space_resolution
                    .as_ref()
                    .is_none_or(|values| !numeric_values_positive(values))
            {
                add(
                    "kinetic_fidelity_underspecified",
                    "Other declared kinetic methods require particles_per_cell or phase_space_resolution.",
                );
            }
        }
    }

    if !(candidate.solver.repository.starts_with("https://")
        || candidate.solver.repository.starts_with("http://"))
        || !non_empty(&candidate.solver.repository)
    {
        add("invalid_solver_repository", "Solver repository must be a non-empty HTTP(S) URI.");
    }
    if !is_hex(&candidate.solver.revision, 40) {
        add("invalid_solver_revision", "Solver revision must be a 40-character hexadecimal commit SHA.");
    }
    if candidate.solver.numerical_options.is_empty() {
        add("missing_solver_options", "At least one numerical solver option must be pinned.");
    }
    if candidate.solver.compiler_runtime.as_ref().is_some_and(|value| !non_empty(value)) {
        add("empty_solver_runtime", "compiler_runtime cannot be empty when supplied.");
    }

    if candidate.target_observables.is_empty()
        || candidate.target_observables.iter().any(|value| !non_empty(value))
    {
        add("invalid_target_observables", "At least one non-empty target observable is required.");
    }

    let unique_targets = candidate
        .target_observables
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    if unique_targets.len() != candidate.target_observables.len() {
        add(
            "duplicate_target_observable",
            "target_observables must be unique.",
        );
    }

    let cost = &candidate.cost_estimate;
    if cost.measurement_basis != "raw_resource_units" {
        add(
            "invalid_cost_basis",
            "Cost estimates must use raw_resource_units; monetary or weighted scores belong in separate fields.",
        );
    }
    if !finite_nonnegative(cost.cpu_core_hours) || !finite_nonnegative(cost.gpu_hours) {
        add("invalid_compute_cost", "CPU core-hours and GPU-hours must be finite and non-negative.");
    }
    if !positive(cost.wall_seconds) {
        add("invalid_wall_cost", "Estimated wall_seconds must be finite and positive.");
    }
    if cost.peak_memory_bytes == 0 || cost.data_volume_bytes == 0 {
        add("invalid_resource_capacity", "Peak memory and expected data volume must be positive.");
    }
    if cost.cost_estimator_revision.as_ref().is_some_and(|value| !is_hex(value, 40)) {
        add(
            "invalid_cost_estimator_revision",
            "Cost estimator revision must be a 40-character hexadecimal SHA when supplied.",
        );
    }

    if !is_hex(&candidate.acquisition.policy_revision, 40) {
        add(
            "invalid_acquisition_revision",
            "Acquisition policy revision must be a 40-character hexadecimal commit SHA.",
        );
    }
    if candidate.acquisition.reason_codes.is_empty() {
        add("missing_acquisition_reason", "At least one acquisition reason code is required.");
    }
    let unique_reasons = candidate
        .acquisition
        .reason_codes
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    if unique_reasons.len() != candidate.acquisition.reason_codes.len() {
        add(
            "duplicate_acquisition_reason",
            "Acquisition reason codes must be unique.",
        );
    }
    if candidate.acquisition.evidence_receipts.is_empty() {
        add("missing_acquisition_evidence", "At least one evidence receipt is required.");
    }
    for receipt in &candidate.acquisition.evidence_receipts {
        if !non_empty(&receipt.receipt_id) || !is_hex(&receipt.sha256, 64) {
            add(
                "invalid_acquisition_receipt",
                "Every acquisition evidence receipt requires a non-empty ID and a 64-character SHA-256.",
            );
        }
    }
    if candidate.acquisition.signal_values.is_empty()
        || candidate.acquisition.signal_values.values().any(|value| !value.is_finite())
    {
        add("invalid_acquisition_signals", "At least one finite raw acquisition signal is required.");
    }

    match &candidate.selection_status {
        SelectionStatus::Selected | SelectionStatus::Executed => {
            if candidate.rights_state != RightsState::Verified {
                add("rights_not_verified", "A selected or executed candidate must have rights_state=verified.");
            }
        }
        _ => {}
    }
    if candidate.selection_status == SelectionStatus::Executed
        && candidate.execution_receipt_id.as_ref().is_none_or(|value| !non_empty(value))
    {
        add("execution_receipt_missing", "An executed candidate must include an execution_receipt_id.");
    }
    if candidate.split_role != SplitRole::Training
        && candidate.split_role != SplitRole::DevelopmentValidation
    {
        add(
            "forbidden_split_role",
            "Only training and development_validation can be proposed for acquisition; sealed test is never selectable.",
        );
    }

    if candidate.notes.as_ref().is_some_and(|value| value.trim().is_empty()) {
        add("empty_notes", "notes must be omitted rather than supplied empty.");
    }

    findings
}

pub fn audit_candidate(candidate: &CandidateRun) -> Result<CandidateAuditReport, String> {
    let request_bytes = serde_json::to_vec(candidate)
        .map_err(|error| format!("candidate serialization failed: {error}"))?;
    let request_sha256 = sha256_hex(&request_bytes);
    let findings = candidate_findings(candidate);
    let body = CandidateAuditBody {
        schema_version: SCHEMA_VERSION.to_owned(),
        candidate_id: candidate.candidate_id.clone(),
        disposition: if findings.is_empty() {
            AuditDisposition::Admissible
        } else {
            AuditDisposition::Rejected
        },
        request_sha256,
        selection_status: candidate.selection_status.clone(),
        split_role: candidate.split_role.clone(),
        rights_state: candidate.rights_state.clone(),
        findings,
        claim_ceiling: "PROPOSAL_CONTRACT_ONLY_NO_KINETIC_SIMULATION_SELECTED_OR_EXECUTED".to_owned(),
    };
    let report_bytes = serde_json::to_vec(&body)
        .map_err(|error| format!("audit report serialization failed: {error}"))?;
    Ok(CandidateAuditReport {
        body,
        receipt_sha256: sha256_hex(&report_bytes),
    })
}

pub fn verify_candidate_report(
    candidate: &CandidateRun,
    report: &CandidateAuditReport,
) -> Result<(), String> {
    let expected = audit_candidate(candidate)?;
    if expected != *report {
        return Err("replayed candidate audit differs from supplied report".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA40: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHA64: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn valid_pic_candidate() -> CandidateRun {
        CandidateRun {
            schema_version: SCHEMA_VERSION.to_owned(),
            candidate_id: "landau-pic-candidate-001".to_owned(),
            parent_experiment_id: "landau-development-v0".to_owned(),
            physics_lane: PhysicsLane::LandauDamping1dElectrostatic,
            governing_equation_version: "vlasov-poisson-1d-v1".to_owned(),
            physical_parameters: BTreeMap::from([
                ("amplitude".to_owned(), Value::from(0.05)),
                ("wavenumber".to_owned(), Value::from(0.5)),
            ]),
            initial_condition_id: "weak-landau-alpha-0.05".to_owned(),
            boundary_condition_id: "periodic".to_owned(),
            geometry: Geometry {
                geometry_id: "periodic-1d".to_owned(),
                dimension: 1,
                grid_size: vec![256],
                cell_spacing: Some(vec![0.02454369260617026]),
                domain_size: Some(vec![6.283185307179586]),
                coordinate_frame_id: Some("x-periodic".to_owned()),
            },
            resolution: Resolution {
                spatial_resolution_id: "nx-256".to_owned(),
                time_step_seconds: 0.05,
                output_cadence_seconds: Some(0.05),
                downsampling: None,
                phase_space_resolution: None,
            },
            kinetic_fidelity: KineticFidelity {
                method: KineticMethod::Pic,
                fidelity_id: "pic-100k-ppc-v1".to_owned(),
                particles_per_cell: Some(100_000),
                phase_space_resolution: None,
                noise_or_discretisation_notes: Some("Synthetic test fixture only.".to_owned()),
            },
            solver: SolverIdentity {
                repository: "https://github.com/example/kinetic-solver".to_owned(),
                revision: SHA40.to_owned(),
                solver_name: "vlasov-poisson-pic".to_owned(),
                build_id: "fixture-build-v1".to_owned(),
                compiler_runtime: Some("rust-1.96".to_owned()),
                numerical_options: BTreeMap::from([
                    ("integrator".to_owned(), Value::from("leapfrog")),
                    ("shape_function".to_owned(), Value::from("CIC")),
                ]),
                random_seed_policy: Some("fixed_seed".to_owned()),
            },
            random_seed: 7,
            target_observables: vec!["electric_field".to_owned(), "field_energy".to_owned()],
            cost_estimate: CostEstimate {
                measurement_basis: "raw_resource_units".to_owned(),
                cpu_core_hours: 2.0,
                gpu_hours: 0.5,
                wall_seconds: 7200.0,
                peak_memory_bytes: 8_000_000_000,
                data_volume_bytes: 1_000_000_000,
                cost_estimator_revision: Some(SHA40.to_owned()),
            },
            acquisition: Acquisition {
                policy_id: "random-baseline-v1".to_owned(),
                policy_revision: SHA40.to_owned(),
                reason_codes: vec![AcquisitionReason::Random],
                evidence_receipts: vec![EvidenceReceipt {
                    receipt_id: "dev-envelope-receipt-1".to_owned(),
                    sha256: SHA64.to_owned(),
                }],
                signal_values: BTreeMap::from([("diversity_score".to_owned(), 0.5)]),
                selection_rationale: Some("Random baseline proposal.".to_owned()),
            },
            split_role: SplitRole::Training,
            rights_state: RightsState::Verified,
            selection_status: SelectionStatus::Proposed,
            execution_receipt_id: None,
            notes: None,
        }
    }

    #[test]
    fn valid_pic_proposal_is_admissible() {
        let report = audit_candidate(&valid_pic_candidate()).expect("audit");
        assert_eq!(report.body.disposition, AuditDisposition::Admissible);
        assert!(report.body.findings.is_empty());
    }

    #[test]
    fn selected_candidate_requires_verified_rights() {
        let mut candidate = valid_pic_candidate();
        candidate.selection_status = SelectionStatus::Selected;
        candidate.rights_state = RightsState::PermissionPending;
        let report = audit_candidate(&candidate).expect("audit");
        assert_eq!(report.body.disposition, AuditDisposition::Rejected);
        assert!(report.body.findings.iter().any(|item| item.code == "rights_not_verified"));
    }

    #[test]
    fn executed_candidate_requires_execution_receipt() {
        let mut candidate = valid_pic_candidate();
        candidate.selection_status = SelectionStatus::Executed;
        candidate.rights_state = RightsState::Verified;
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "execution_receipt_missing"));
    }

    #[test]
    fn vfp_candidate_requires_phase_space_resolution() {
        let mut candidate = valid_pic_candidate();
        candidate.kinetic_fidelity.method = KineticMethod::Vfp;
        candidate.kinetic_fidelity.particles_per_cell = None;
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "phase_space_resolution_missing"));
    }

    #[test]
    fn pic_candidate_requires_particle_count() {
        let mut candidate = valid_pic_candidate();
        candidate.kinetic_fidelity.particles_per_cell = None;
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "pic_particle_count_missing"));
    }

    #[test]
    fn geometry_dimension_must_match_grid_shape() {
        let mut candidate = valid_pic_candidate();
        candidate.geometry.dimension = 2;
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "grid_shape_dimension_mismatch"));
    }

    #[test]
    fn solver_and_policy_revisions_must_be_immutable_sha() {
        let mut candidate = valid_pic_candidate();
        candidate.solver.revision = "main".to_owned();
        candidate.acquisition.policy_revision = "latest".to_owned();
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "invalid_solver_revision"));
        assert!(report.body.findings.iter().any(|item| item.code == "invalid_acquisition_revision"));
    }

    #[test]
    fn sealed_test_split_is_not_deserializable() {
        let mut value = serde_json::to_value(valid_pic_candidate()).expect("serialize");
        value["split_role"] = Value::from("sealed_test");
        assert!(serde_json::from_value::<CandidateRun>(value).is_err());
    }

    #[test]
    fn nonfinite_cost_or_signal_is_rejected() {
        let mut candidate = valid_pic_candidate();
        candidate.cost_estimate.gpu_hours = f64::NAN;
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "invalid_compute_cost"));

        let mut candidate = valid_pic_candidate();
        candidate.acquisition.signal_values.insert("uncertainty".to_owned(), f64::INFINITY);
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "invalid_acquisition_signals"));
    }

    #[test]
    fn acquisition_receipts_must_be_sha256_bound() {
        let mut candidate = valid_pic_candidate();
        candidate.acquisition.evidence_receipts[0].sha256 = "not-a-digest".to_owned();
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "invalid_acquisition_receipt"));
    }

    #[test]
    fn candidate_audit_receipt_is_deterministic_and_replayable() {
        let candidate = valid_pic_candidate();
        let first = audit_candidate(&candidate).expect("first");
        let second = audit_candidate(&candidate).expect("second");
        assert_eq!(first, second);
        verify_candidate_report(&candidate, &first).expect("verify");
    }

    #[test]
    fn verifier_rejects_rehashed_report_tampering() {
        let candidate = valid_pic_candidate();
        let mut report = audit_candidate(&candidate).expect("audit");
        report.body.claim_ceiling = "SELECTED_AND_EXECUTED".to_owned();
        let bytes = serde_json::to_vec(&report.body).expect("serialize");
        report.receipt_sha256 = sha256_hex(&bytes);
        assert!(verify_candidate_report(&candidate, &report).is_err());
    }

    #[test]
    fn frozen_valid_candidate_fixture_parses_and_passes() {
        let input = include_str!("../../../../benchmarks/plasma/active-kinetic-candidate-valid-v0.json");
        let candidate: CandidateRun = serde_json::from_str(input).expect("candidate fixture parses");
        let report = audit_candidate(&candidate).expect("audit");
        assert_eq!(report.body.disposition, AuditDisposition::Admissible);
        assert!(report.body.findings.is_empty());
    }

    #[test]
    fn frozen_rights_negative_control_is_rejected() {
        let input =
            include_str!("../../../../benchmarks/plasma/active-kinetic-candidate-rights-negative-v0.json");
        let candidate: CandidateRun =
            serde_json::from_str(input).expect("negative fixture parses");
        let report = audit_candidate(&candidate).expect("audit");
        assert_eq!(report.body.disposition, AuditDisposition::Rejected);
        assert!(report.body.findings.iter().any(|finding| {
            finding.code == "rights_not_verified"
        }));
    }

    #[test]
    fn unknown_fields_are_rejected_at_top_and_nested_levels() {
        let mut value = serde_json::to_value(valid_pic_candidate()).expect("serialize");
        value["unregistered"] = Value::Bool(true);
        assert!(serde_json::from_value::<CandidateRun>(value).is_err());

        let mut value = serde_json::to_value(valid_pic_candidate()).expect("serialize");
        value["solver"]["unregistered"] = Value::Bool(true);
        assert!(serde_json::from_value::<CandidateRun>(value).is_err());
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let mut candidate = valid_pic_candidate();
        candidate.schema_version = "9.9.9".to_owned();
        let report = audit_candidate(&candidate).expect("audit");
        assert!(report.body.findings.iter().any(|item| item.code == "unsupported_schema_version"));
    }
}
