//! Fail-closed comparability contracts for kinetic-plasma benchmark runs.
//!
//! This crate decides whether two recorded runs are eligible for a paired
//! comparison. It does not validate the numerical correctness of either
//! solver. Physical diagnostic thresholds and source-specific adapters remain
//! the responsibility of separately versioned benchmark code.

#![forbid(unsafe_code)]

/// Governing-equation cohorts. Physically distinct lanes are never pooled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsLane {
    CollisionlessElectrostaticVlasovPoisson,
    ElectrostaticVlasovAmpere,
    CollisionalVlasovPoissonLandau,
    ElectromagneticVlasovMaxwell,
    QuasineutralVlasovAmpere,
    RelativisticVlasovMaxwell,
}

/// Numeric precision used by the executed solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericPrecision {
    Fp64,
    Fp32,
    Fp16,
    Bf16,
    Mixed,
}

/// Comparison purpose. Accuracy comparisons can span precision/hardware;
/// strict timing comparisons cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonPurpose {
    Accuracy,
    StrictPerformance,
}

/// Data split recorded for the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitRole {
    Training,
    Validation,
    Test,
}

/// Immutable identity of the physical task and numerical experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskIdentity {
    pub physics_lane: PhysicsLane,
    pub spatial_dimension: u8,
    pub velocity_dimension: u8,
    pub geometry_id: String,
    pub boundary_condition_id: String,
    pub normalization_sha256: String,
    pub physical_parameters_sha256: String,
    pub initial_condition_cohort_sha256: String,
    pub initial_state_sha256: String,
    pub reference_corpus_sha256: String,
    pub timestep_seconds: f64,
    pub final_time_seconds: f64,
    pub output_schedule_sha256: String,
    pub metric_profile_id: String,
    pub metric_profile_sha256: String,
    pub precision: NumericPrecision,
    /// Hardware/software platform fingerprint; mandatory for performance.
    pub platform_fingerprint_sha256: Option<String>,
}

/// Solver provenance. Solver identity is intentionally not part of the
/// physical task key: different methods are the competitors being compared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverIdentity {
    pub solver_id: String,
    pub solver_family: String,
    /// Git commit SHA-1 or SHA-256.
    pub source_commit: String,
    pub build_sha256: String,
    pub configuration_sha256: String,
    pub learned: bool,
}

/// Data split and model-training provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataProvenance {
    pub split_id: String,
    pub split_role: SplitRole,
    pub test_data_sha256: String,
    pub training_data_sha256: Option<String>,
    pub checkpoint_sha256: Option<String>,
    /// Per-case training initial-state digests; checked against the test case.
    pub training_initial_state_sha256: Vec<String>,
}

/// Receipts binding execution and output artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceReceipts {
    pub execution_receipt_sha256: String,
    pub output_bundle_sha256: String,
}

/// One diagnostic observation and its versioned mathematical definition.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticValue {
    pub metric_id: String,
    pub definition_sha256: String,
    pub value: f64,
}

/// Complete run manifest required for benchmark admission.
#[derive(Debug, Clone, PartialEq)]
pub struct RunManifest {
    pub run_id: String,
    pub seed: u64,
    pub task: TaskIdentity,
    pub solver: SolverIdentity,
    pub data: DataProvenance,
    pub receipts: EvidenceReceipts,
    pub diagnostics: Vec<DiagnosticValue>,
}

/// A machine-readable finding from validation or comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub detail: String,
}

/// Result of a paired-comparison eligibility check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonReport {
    pub eligible: bool,
    pub performance_comparable: bool,
    pub findings: Vec<Finding>,
}

/// Required diagnostic identifiers for each physical lane.
///
/// These are presence gates, not numerical acceptance thresholds. Tolerance
/// policies must be pinned separately to a benchmark task and metric profile.
pub fn required_diagnostics(lane: PhysicsLane) -> &'static [&'static str] {
    match lane {
        PhysicsLane::CollisionlessElectrostaticVlasovPoisson => &[
            "poisson_gauss_residual",
            "charge_continuity_residual",
            "particle_field_energy_drift",
            "momentum_drift",
            "distribution_error",
            "linear_mode_rate_error",
        ],
        PhysicsLane::ElectrostaticVlasovAmpere => &[
            "gauss_law_residual",
            "charge_continuity_residual",
            "curl_free_residual",
            "mass_drift",
            "momentum_drift",
            "total_energy_drift",
        ],
        PhysicsLane::CollisionalVlasovPoissonLandau => &[
            "mass_drift",
            "momentum_drift",
            "total_energy_drift",
            "entropy_production",
            "collision_operator_degeneracy",
        ],
        PhysicsLane::ElectromagneticVlasovMaxwell => &[
            "gauss_e_residual",
            "div_b_residual",
            "charge_continuity_residual",
            "maxwell_residual",
            "particle_field_energy_drift",
            "momentum_drift",
        ],
        PhysicsLane::QuasineutralVlasovAmpere => &[
            "gauss_law_residual",
            "mass_drift",
            "momentum_drift",
            "total_energy_drift",
            "distribution_minimum",
            "quasineutral_asymptotic_error",
        ],
        PhysicsLane::RelativisticVlasovMaxwell => &[
            "gauss_e_residual",
            "div_b_residual",
            "charge_continuity_residual",
            "maxwell_residual",
            "particle_field_energy_drift",
            "momentum_drift",
            "relativistic_invariant_error",
            "energy_correction_failure_count",
        ],
    }
}

/// Validate a run independently of any comparison partner.
pub fn validate_run(run: &RunManifest) -> Vec<Finding> {
    let mut findings = Vec::new();

    for (field, value) in [
        ("run_id", run.run_id.as_str()),
        ("geometry_id", run.task.geometry_id.as_str()),
        ("boundary_condition_id", run.task.boundary_condition_id.as_str()),
        ("metric_profile_id", run.task.metric_profile_id.as_str()),
        ("solver_id", run.solver.solver_id.as_str()),
        ("solver_family", run.solver.solver_family.as_str()),
        ("split_id", run.data.split_id.as_str()),
    ] {
        if value.trim().is_empty() {
            findings.push(Finding {
                code: "empty_required_identity",
                detail: format!("{field} must be non-empty"),
            });
        }
    }

    if !(1..=3).contains(&run.task.spatial_dimension)
        || !(1..=3).contains(&run.task.velocity_dimension)
    {
        findings.push(Finding {
            code: "invalid_phase_space_dimension",
            detail: "spatial and velocity dimensions must each be in 1..=3".to_owned(),
        });
    }

    if !run.task.timestep_seconds.is_finite() || run.task.timestep_seconds <= 0.0 {
        findings.push(Finding {
            code: "invalid_timestep",
            detail: "timestep must be finite and positive".to_owned(),
        });
    }
    if !run.task.final_time_seconds.is_finite()
        || run.task.final_time_seconds <= run.task.timestep_seconds
    {
        findings.push(Finding {
            code: "invalid_final_time",
            detail: "final time must be finite and greater than one timestep".to_owned(),
        });
    }

    for (field, value) in [
        ("normalization_sha256", run.task.normalization_sha256.as_str()),
        (
            "physical_parameters_sha256",
            run.task.physical_parameters_sha256.as_str(),
        ),
        (
            "initial_condition_cohort_sha256",
            run.task.initial_condition_cohort_sha256.as_str(),
        ),
        ("initial_state_sha256", run.task.initial_state_sha256.as_str()),
        ("reference_corpus_sha256", run.task.reference_corpus_sha256.as_str()),
        ("output_schedule_sha256", run.task.output_schedule_sha256.as_str()),
        ("metric_profile_sha256", run.task.metric_profile_sha256.as_str()),
        ("build_sha256", run.solver.build_sha256.as_str()),
        (
            "configuration_sha256",
            run.solver.configuration_sha256.as_str(),
        ),
        ("test_data_sha256", run.data.test_data_sha256.as_str()),
        (
            "execution_receipt_sha256",
            run.receipts.execution_receipt_sha256.as_str(),
        ),
        (
            "output_bundle_sha256",
            run.receipts.output_bundle_sha256.as_str(),
        ),
    ] {
        if !is_hex(value, 64) {
            findings.push(Finding {
                code: "invalid_sha256",
                detail: format!("{field} must be 64 lowercase hexadecimal characters"),
            });
        }
    }

    if !is_object_id(&run.solver.source_commit) {
        findings.push(Finding {
            code: "invalid_source_commit",
            detail: "source_commit must be a 40- or 64-character hexadecimal object id".to_owned(),
        });
    }

    if run.data.test_data_sha256 != run.task.initial_state_sha256 {
        findings.push(Finding {
            code: "test_data_identity_mismatch",
            detail: "test_data_sha256 must equal the task initial_state_sha256".to_owned(),
        });
    }
    if run.data.split_role != SplitRole::Test {
        findings.push(Finding {
            code: "not_a_test_run",
            detail: "benchmark qualification requires an independently held-out test run".to_owned(),
        });
    }
    if run
        .data
        .training_initial_state_sha256
        .iter()
        .any(|digest| !is_hex(digest, 64))
    {
        findings.push(Finding {
            code: "invalid_training_case_digest",
            detail: "every training initial-state digest must be SHA-256".to_owned(),
        });
    }
    if run
        .data
        .training_initial_state_sha256
        .iter()
        .any(|digest| digest == &run.task.initial_state_sha256)
    {
        findings.push(Finding {
            code: "train_test_case_leakage",
            detail: "the test initial state occurs in the recorded training cases".to_owned(),
        });
    }

    if run.solver.learned {
        match run.data.training_data_sha256.as_deref() {
            Some(digest) if is_hex(digest, 64) => {}
            _ => findings.push(Finding {
                code: "missing_training_data_digest",
                detail: "learned solvers require a training-data SHA-256 digest".to_owned(),
            }),
        }
        match run.data.checkpoint_sha256.as_deref() {
            Some(digest) if is_hex(digest, 64) => {}
            _ => findings.push(Finding {
                code: "missing_checkpoint_digest",
                detail: "learned solvers require a model-checkpoint SHA-256 digest".to_owned(),
            }),
        }
    }

    if let Some(platform) = run.task.platform_fingerprint_sha256.as_deref() {
        if !is_hex(platform, 64) {
            findings.push(Finding {
                code: "invalid_platform_fingerprint",
                detail: "platform fingerprint must be SHA-256".to_owned(),
            });
        }
    }

    let mut seen_metrics = Vec::new();
    for metric in &run.diagnostics {
        if metric.metric_id.trim().is_empty() {
            findings.push(Finding {
                code: "empty_metric_id",
                detail: "diagnostic metric_id must be non-empty".to_owned(),
            });
        }
        if seen_metrics.iter().any(|seen| seen == &metric.metric_id) {
            findings.push(Finding {
                code: "duplicate_metric",
                detail: format!("diagnostic {} occurs more than once", metric.metric_id),
            });
        }
        seen_metrics.push(metric.metric_id.as_str());

        if !is_hex(&metric.definition_sha256, 64) {
            findings.push(Finding {
                code: "invalid_metric_definition_digest",
                detail: format!("diagnostic {} has an invalid definition digest", metric.metric_id),
            });
        }
        if !metric.value.is_finite() {
            findings.push(Finding {
                code: "non_finite_metric",
                detail: format!("diagnostic {} is non-finite", metric.metric_id),
            });
        }
    }

    for required in required_diagnostics(run.task.physics_lane) {
        if !run.diagnostics.iter().any(|metric| metric.metric_id == *required) {
            findings.push(Finding {
                code: "missing_required_diagnostic",
                detail: format!("{} requires diagnostic {}", lane_name(run.task.physics_lane), required),
            });
        }
    }

    findings
}

/// Decide whether two runs can be compared under a declared purpose.
///
/// Accuracy comparison requires the same physical task, initial state, held-out
/// split, seed, reference corpus, timestep, output schedule and metric profile.
/// Strict performance comparison additionally requires exact numeric
/// precision and platform identity. Solver family/source/build are recorded
/// but are expected to differ between competitors.
pub fn compare_runs(
    left: &RunManifest,
    right: &RunManifest,
    purpose: ComparisonPurpose,
) -> ComparisonReport {
    let mut findings = validate_run(left);
    findings.extend(validate_run(right));

    compare_task_identity(&left.task, &right.task, &mut findings);

    if left.data.split_id != right.data.split_id {
        add_mismatch(&mut findings, "split_mismatch", "held-out split ids differ");
    }
    if left.seed != right.seed {
        add_mismatch(&mut findings, "seed_mismatch", "paired runs must use the same seed");
    }

    compare_diagnostic_definitions(left, right, &mut findings);

    if purpose == ComparisonPurpose::StrictPerformance {
        if left.task.precision != right.task.precision {
            add_mismatch(
                &mut findings,
                "precision_mismatch",
                "strict timing comparisons require identical numeric precision",
            );
        }
        match (
            left.task.platform_fingerprint_sha256.as_deref(),
            right.task.platform_fingerprint_sha256.as_deref(),
        ) {
            (Some(a), Some(b)) if is_hex(a, 64) && a == b => {}
            _ => add_mismatch(
                &mut findings,
                "platform_mismatch",
                "strict timing comparisons require the same valid platform fingerprint",
            ),
        }
    }

    let eligible = findings.is_empty();
    ComparisonReport {
        eligible,
        performance_comparable: eligible && purpose == ComparisonPurpose::StrictPerformance,
        findings,
    }
}

fn compare_task_identity(left: &TaskIdentity, right: &TaskIdentity, findings: &mut Vec<Finding>) {
    if left.physics_lane != right.physics_lane {
        add_mismatch(findings, "physics_lane_mismatch", "governing-equation cohorts differ");
    }
    if left.spatial_dimension != right.spatial_dimension
        || left.velocity_dimension != right.velocity_dimension
    {
        add_mismatch(findings, "phase_space_dimension_mismatch", "phase-space dimensions differ");
    }
    if left.geometry_id != right.geometry_id {
        add_mismatch(findings, "geometry_mismatch", "geometry or topology identifiers differ");
    }
    if left.boundary_condition_id != right.boundary_condition_id {
        add_mismatch(findings, "boundary_mismatch", "boundary-condition identifiers differ");
    }
    if left.normalization_sha256 != right.normalization_sha256 {
        add_mismatch(findings, "normalization_mismatch", "unit/coordinate normalization differs");
    }
    if left.physical_parameters_sha256 != right.physical_parameters_sha256 {
        add_mismatch(findings, "physical_parameters_mismatch", "physical parameter digests differ");
    }
    if left.initial_condition_cohort_sha256 != right.initial_condition_cohort_sha256 {
        add_mismatch(findings, "initial_condition_cohort_mismatch", "initial-condition cohort differs");
    }
    if left.initial_state_sha256 != right.initial_state_sha256 {
        add_mismatch(findings, "initial_state_mismatch", "paired runs do not share the same initial state");
    }
    if left.reference_corpus_sha256 != right.reference_corpus_sha256 {
        add_mismatch(findings, "reference_corpus_mismatch", "reference corpus digests differ");
    }
    if left.timestep_seconds.to_bits() != right.timestep_seconds.to_bits() {
        add_mismatch(findings, "timestep_mismatch", "timestep differs; compare in a separate convergence cohort");
    }
    if left.final_time_seconds.to_bits() != right.final_time_seconds.to_bits() {
        add_mismatch(findings, "final_time_mismatch", "simulation horizons differ");
    }
    if left.output_schedule_sha256 != right.output_schedule_sha256 {
        add_mismatch(findings, "output_schedule_mismatch", "diagnostic output schedules differ");
    }
    if left.metric_profile_id != right.metric_profile_id
        || left.metric_profile_sha256 != right.metric_profile_sha256
    {
        add_mismatch(findings, "metric_profile_mismatch", "metric profile identity differs");
    }
}

fn compare_diagnostic_definitions(
    left: &RunManifest,
    right: &RunManifest,
    findings: &mut Vec<Finding>,
) {
    if left.diagnostics.len() != right.diagnostics.len() {
        add_mismatch(findings, "diagnostic_set_mismatch", "diagnostic counts differ");
        return;
    }

    for metric in &left.diagnostics {
        match right
            .diagnostics
            .iter()
            .find(|candidate| candidate.metric_id == metric.metric_id)
        {
            Some(candidate) if candidate.definition_sha256 == metric.definition_sha256 => {}
            _ => add_mismatch(
                findings,
                "diagnostic_definition_mismatch",
                &format!("diagnostic {} is absent or uses a different definition", metric.metric_id),
            ),
        }
    }
}

fn add_mismatch(findings: &mut Vec<Finding>, code: &'static str, detail: &str) {
    findings.push(Finding {
        code,
        detail: detail.to_owned(),
    });
}

fn lane_name(lane: PhysicsLane) -> &'static str {
    match lane {
        PhysicsLane::CollisionlessElectrostaticVlasovPoisson => "collisionless Vlasov-Poisson",
        PhysicsLane::ElectrostaticVlasovAmpere => "electrostatic Vlasov-Ampere",
        PhysicsLane::CollisionalVlasovPoissonLandau => "collisional Vlasov-Poisson-Landau",
        PhysicsLane::ElectromagneticVlasovMaxwell => "electromagnetic Vlasov-Maxwell",
        PhysicsLane::QuasineutralVlasovAmpere => "quasi-neutral Vlasov-Ampere",
        PhysicsLane::RelativisticVlasovMaxwell => "relativistic Vlasov-Maxwell",
    }
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_object_id(value: &str) -> bool {
    is_hex(value, 40) || is_hex(value, 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const COMMIT: &str = "cccccccccccccccccccccccccccccccccccccccc";

    fn task() -> TaskIdentity {
        TaskIdentity {
            physics_lane: PhysicsLane::CollisionlessElectrostaticVlasovPoisson,
            spatial_dimension: 1,
            velocity_dimension: 1,
            geometry_id: "periodic-1d".to_owned(),
            boundary_condition_id: "periodic".to_owned(),
            normalization_sha256: DIGEST_A.to_owned(),
            physical_parameters_sha256: DIGEST_B.to_owned(),
            initial_condition_cohort_sha256: DIGEST_A.to_owned(),
            initial_state_sha256: DIGEST_B.to_owned(),
            reference_corpus_sha256: DIGEST_A.to_owned(),
            timestep_seconds: 0.01,
            final_time_seconds: 10.0,
            output_schedule_sha256: DIGEST_B.to_owned(),
            metric_profile_id: "landau-1d-v1".to_owned(),
            metric_profile_sha256: DIGEST_A.to_owned(),
            precision: NumericPrecision::Fp64,
            platform_fingerprint_sha256: Some(DIGEST_B.to_owned()),
        }
    }

    fn diagnostic(metric_id: &str) -> DiagnosticValue {
        DiagnosticValue {
            metric_id: metric_id.to_owned(),
            definition_sha256: DIGEST_A.to_owned(),
            value: 0.0,
        }
    }

    fn run(solver_id: &str, solver_family: &str) -> RunManifest {
        RunManifest {
            run_id: format!("run-{solver_id}"),
            seed: 17,
            task: task(),
            solver: SolverIdentity {
                solver_id: solver_id.to_owned(),
                solver_family: solver_family.to_owned(),
                source_commit: COMMIT.to_owned(),
                build_sha256: DIGEST_A.to_owned(),
                configuration_sha256: DIGEST_B.to_owned(),
                learned: false,
            },
            data: DataProvenance {
                split_id: "heldout-test-2026-01".to_owned(),
                split_role: SplitRole::Test,
                test_data_sha256: DIGEST_B.to_owned(),
                training_data_sha256: None,
                checkpoint_sha256: None,
                training_initial_state_sha256: vec![DIGEST_A.to_owned()],
            },
            receipts: EvidenceReceipts {
                execution_receipt_sha256: DIGEST_A.to_owned(),
                output_bundle_sha256: DIGEST_B.to_owned(),
            },
            diagnostics: required_diagnostics(PhysicsLane::CollisionlessElectrostaticVlasovPoisson)
                .iter()
                .map(|metric| diagnostic(metric))
                .collect(),
        }
    }

    fn has_code(findings: &[Finding], code: &str) -> bool {
        findings.iter().any(|finding| finding.code == code)
    }

    #[test]
    fn compatible_classical_and_learned_solvers_are_eligible() {
        let classical = run("pif-reference", "variational-fourier-particle");
        let mut learned = run("neopic-model", "particle-measure-neural-operator");
        learned.solver.learned = true;
        learned.data.training_data_sha256 = Some(DIGEST_A.to_owned());
        learned.data.checkpoint_sha256 = Some(DIGEST_B.to_owned());
        let report = compare_runs(&classical, &learned, ComparisonPurpose::StrictPerformance);

        assert!(report.eligible, "{:?}", report.findings);
        assert!(report.performance_comparable);
    }

    #[test]
    fn rejects_cross_physics_lane_comparison() {
        let left = run("vp", "pic");
        let mut right = run("vm", "pic");
        right.task.physics_lane = PhysicsLane::ElectromagneticVlasovMaxwell;
        let report = compare_runs(&left, &right, ComparisonPurpose::Accuracy);
        assert!(!report.eligible);
        assert!(has_code(&report.findings, "physics_lane_mismatch"));
    }

    #[test]
    fn rejects_boundary_normalization_and_initial_state_mismatch() {
        let left = run("solver-a", "pic");
        let mut right = run("solver-b", "neural");
        right.task.boundary_condition_id = "open".to_owned();
        right.task.normalization_sha256 = DIGEST_B.to_owned();
        right.task.initial_state_sha256 = DIGEST_A.to_owned();
        right.data.test_data_sha256 = DIGEST_A.to_owned();
        let report = compare_runs(&left, &right, ComparisonPurpose::Accuracy);
        assert!(!report.eligible);
        assert!(has_code(&report.findings, "boundary_mismatch"));
        assert!(has_code(&report.findings, "normalization_mismatch"));
        assert!(has_code(&report.findings, "initial_state_mismatch"));
    }

    #[test]
    fn rejects_missing_required_physics_diagnostic() {
        let mut candidate = run("candidate", "pic");
        candidate.diagnostics.retain(|metric| metric.metric_id != "particle_field_energy_drift");
        let findings = validate_run(&candidate);
        assert!(has_code(&findings, "missing_required_diagnostic"));
    }

    #[test]
    fn rejects_training_test_initial_state_leakage() {
        let mut candidate = run("candidate", "learned");
        candidate.solver.learned = true;
        candidate.data.training_data_sha256 = Some(DIGEST_A.to_owned());
        candidate.data.checkpoint_sha256 = Some(DIGEST_B.to_owned());
        candidate.data.training_initial_state_sha256.push(DIGEST_B.to_owned());
        let findings = validate_run(&candidate);
        assert!(has_code(&findings, "train_test_case_leakage"));
    }

    #[test]
    fn rejects_missing_learned_model_provenance() {
        let mut candidate = run("candidate", "learned");
        candidate.solver.learned = true;
        let findings = validate_run(&candidate);
        assert!(has_code(&findings, "missing_training_data_digest"));
        assert!(has_code(&findings, "missing_checkpoint_digest"));
    }

    #[test]
    fn accuracy_can_compare_different_precision_but_timing_cannot() {
        let left = run("fp64", "pif");
        let mut right = run("fp32", "neopic");
        right.task.precision = NumericPrecision::Fp32;
        assert!(compare_runs(&left, &right, ComparisonPurpose::Accuracy).eligible);
        let timed = compare_runs(&left, &right, ComparisonPurpose::StrictPerformance);
        assert!(!timed.eligible);
        assert!(has_code(&timed.findings, "precision_mismatch"));
    }

    #[test]
    fn strict_performance_requires_identical_platform_fingerprint() {
        let left = run("solver-a", "pic");
        let mut right = run("solver-b", "neural");
        right.task.platform_fingerprint_sha256 = Some(DIGEST_A.to_owned());
        let report = compare_runs(&left, &right, ComparisonPurpose::StrictPerformance);
        assert!(!report.eligible);
        assert!(has_code(&report.findings, "platform_mismatch"));
    }

    #[test]
    fn rejects_timestep_mismatch_as_a_separate_convergence_cohort() {
        let left = run("solver-a", "pic");
        let mut right = run("solver-b", "neural");
        right.task.timestep_seconds = 0.005;
        let report = compare_runs(&left, &right, ComparisonPurpose::Accuracy);
        assert!(!report.eligible);
        assert!(has_code(&report.findings, "timestep_mismatch"));
    }

    #[test]
    fn rejects_invalid_receipt_or_metric_values() {
        let mut candidate = run("candidate", "pic");
        candidate.receipts.execution_receipt_sha256 = "invalid".to_owned();
        candidate.diagnostics[0].value = f64::NAN;
        let findings = validate_run(&candidate);
        assert!(has_code(&findings, "invalid_sha256"));
        assert!(has_code(&findings, "non_finite_metric"));
    }

    #[test]
    fn requires_distinct_diagnostic_definitions_to_match() {
        let left = run("solver-a", "pic");
        let mut right = run("solver-b", "neural");
        right.diagnostics[0].definition_sha256 = DIGEST_B.to_owned();
        let report = compare_runs(&left, &right, ComparisonPurpose::Accuracy);
        assert!(!report.eligible);
        assert!(has_code(&report.findings, "diagnostic_definition_mismatch"));
    }

    #[test]
    fn collisionless_and_collisional_require_different_diagnostic_profiles() {
        let collisionless = required_diagnostics(PhysicsLane::CollisionlessElectrostaticVlasovPoisson);
        let collisional = required_diagnostics(PhysicsLane::CollisionalVlasovPoissonLandau);
        assert!(collisionless.contains(&"poisson_gauss_residual"));
        assert!(!collisionless.contains(&"entropy_production"));
        assert!(collisional.contains(&"entropy_production"));
        assert!(collisional.contains(&"collision_operator_degeneracy"));
    }

    #[test]
    fn quasi_neutral_lane_has_asymptotic_and_positivity_diagnostics() {
        let required = required_diagnostics(PhysicsLane::QuasineutralVlasovAmpere);
        assert!(required.contains(&"distribution_minimum"));
        assert!(required.contains(&"quasineutral_asymptotic_error"));
    }
}
