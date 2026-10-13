// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end release gate for the Humanoid control substrate.
//!
//! Individual certificates are useful but insufficient when they can be mixed
//! across models, solvers, or incomplete campaigns. This release artifact
//! requires one solver identity and simultaneous oracle, QP, embodied-control,
//! and adversarial-recovery evidence before a build can claim a qualified
//! Humanoid control release.

use serde::{Deserialize, Serialize};

use crate::dynamics_oracle::{
    certify_dynamics_oracle_snapshots, DynamicsOracleCertificate, DynamicsOracleInputCase,
};
use crate::oracle_dataset::DynamicsOracleDataset;
use crate::oracle_generator_certification::{
    recompute_independent_oracle_generation_certificate, IndependentOracleGenerationCertificate,
};
use crate::embodied_certification::{
    certify_embodied_control_against_manifest, EmbodiedCertificationCriteria, EmbodiedControlCase,
    EmbodiedControlCertificate, EmbodiedScenarioManifest,
};
use crate::inprocess_solver_certification::{
    certify_inprocess_sparse_qp_cases, InProcessSparseQpQualificationCertificate,
    InProcessSparseQpQualificationCriteria,
};
use crate::qp_certification::{
    certify_floating_base_qp, FloatingBaseQpCase, FloatingBaseQpCertificate,
    FloatingBaseQpCriteria,
};
use crate::recovery_certification::{
    certify_adversarial_recovery_cases, AdversarialRecoveryCertificationConfig,
    AdversarialRecoveryCertificate,
};
use crate::recovery_benchmark::PushRecoveryProtocol;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidControlReleaseEvidence {
    pub release_id: String,
    pub candidate_build_id: String,
    pub model_artifact_sha256: String,
    pub generated_unix_millis: u64,
    pub oracle_generation: IndependentOracleGenerationCertificate,
    /// Exact generated dataset bound to the oracle-generation receipt.
    #[serde(default)]
    pub oracle_dataset: Option<DynamicsOracleDataset>,
    pub dynamics_oracle: DynamicsOracleCertificate,
    /// Raw candidate/oracle snapshots; the report is recomputed by the release gate.
    #[serde(default)]
    pub dynamics_oracle_inputs: Vec<DynamicsOracleInputCase>,
    pub inprocess_solver: InProcessSparseQpQualificationCertificate,
    pub floating_base_qp: FloatingBaseQpCertificate,
    /// Raw floating-base report cases used to independently recompute the summary.
    #[serde(default)]
    pub floating_base_qp_cases: Vec<FloatingBaseQpCase>,
    pub embodied_control: EmbodiedControlCertificate,
    /// Raw inputs are re-evaluated by the release gate; the serialized pass bit is not trusted.
    #[serde(default)]
    pub embodied_scenario_manifest: EmbodiedScenarioManifest,
    #[serde(default)]
    pub embodied_control_cases: Vec<EmbodiedControlCase>,
    pub adversarial_recovery: AdversarialRecoveryCertificate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidControlReleaseCertificate {
    pub schema_version: u32,
    pub release_id: String,
    pub candidate_build_id: String,
    pub model_artifact_sha256: String,
    /// SHA-256 of the exact evidence JSON bytes verified against an external trust anchor.
    #[serde(default)]
    pub evidence_bundle_sha256: String,
    #[serde(default)]
    pub oracle_generator_executable_sha256: Option<String>,
    #[serde(default)]
    pub controller_artifact_sha256: String,
    #[serde(default)]
    pub simulator_id: String,
    #[serde(default)]
    pub simulator_version: String,
    #[serde(default)]
    pub test_harness_sha256: String,
    #[serde(default)]
    pub scenario_config_sha256: String,
    /// Evidence digests copied from the candidate-bound schema-v5 certificate.
    #[serde(default)]
    pub embodied_corpus_digest: String,
    #[serde(default)]
    pub embodied_scenario_manifest_digest: String,
    pub solver_backend_id: Option<String>,
    pub oracle_dataset_id: String,
    pub generated_unix_millis: u64,
    pub accepted: bool,
    pub failures: Vec<String>,
}

/// Recompute release evidence against the exact evidence-bundle and artifact hashes.
///
/// The evidence-bundle SHA-256, scenario-manifest BLAKE3, and oracle-worker SHA-256
/// trust anchors must be provisioned independently of the evidence JSON. The bundle
/// hash binds the bytes the caller parsed; it does not itself validate nested receipts.
pub fn certify_humanoid_control_release(
    evidence: &HumanoidControlReleaseEvidence,
    actual_evidence_bundle_sha256: &str,
    trusted_evidence_bundle_sha256: &str,
    actual_model_artifact_sha256: &str,
    actual_controller_artifact_sha256: &str,
    actual_test_harness_sha256: &str,
    actual_scenario_config_sha256: &str,
    actual_oracle_executable_sha256: &str,
    trusted_scenario_manifest_digest: &str,
    trusted_oracle_executable_sha256: &str,
) -> HumanoidControlReleaseCertificate {
    let mut failures = Vec::new();
    if !artifact_sha256_matches(
        actual_evidence_bundle_sha256,
        trusted_evidence_bundle_sha256,
    ) {
        failures.push(
            "evidence JSON bytes do not match the separately trusted evidence-bundle SHA-256"
                .to_string(),
        );
    }
    if evidence.release_id.trim().is_empty() {
        failures.push("release identity is empty".to_string());
    }
    if evidence.candidate_build_id.trim().is_empty() {
        failures.push("candidate build identity is empty".to_string());
    }
    if !is_lower_hex_sha256(&evidence.model_artifact_sha256) {
        failures.push("model artifact identity is not canonical SHA-256 hex".to_string());
    }
    if !is_lower_hex_sha256(actual_model_artifact_sha256)
        || actual_model_artifact_sha256 != evidence.model_artifact_sha256
    {
        failures.push("model artifact hash does not match the bytes supplied to the release gate".to_string());
    }
    if !is_lower_hex_blake3(trusted_scenario_manifest_digest) {
        failures.push("trusted scenario-manifest anchor is not canonical BLAKE3 hex".to_string());
    }
    if evidence.oracle_generation.schema_version != 2 {
        failures.push("independent oracle generation certificate schema is not v2".to_string());
    }
    if !oracle_executable_identity_matches(
        actual_oracle_executable_sha256,
        trusted_oracle_executable_sha256,
        evidence.oracle_generation.generator_executable_sha256.as_deref(),
    ) {
        failures.push(
            "oracle worker bytes, generation receipt, and trusted release anchor do not agree"
                .to_string(),
        );
    }
    if evidence.generated_unix_millis == 0 {
        failures.push("release evidence has no generation timestamp".to_string());
    }
    if !oracle_generation_certificate_matches_recomputation(
        &evidence.oracle_generation,
        evidence.oracle_dataset.as_ref(),
        &evidence.candidate_build_id,
        &evidence.model_artifact_sha256,
    ) {
        failures.push(
            "oracle-generation certificate does not match the exact stored dataset and receipt records under pinned release criteria"
                .to_string(),
        );
    }
    // Recompute numerical comparisons from raw candidate/oracle snapshots. The
    // serialized report totals and pass bit are claims until this result agrees.
    if !dynamics_oracle_certificate_matches_recomputation(
        &evidence.dynamics_oracle,
        &evidence.dynamics_oracle_inputs,
        evidence.oracle_dataset.as_ref(),
    ) {
        failures.push(
            "dynamics-oracle certificate does not match recomputation from raw snapshots under pinned tolerances"
                .to_string(),
        );
    }
    // Recompute solver qualification from every embedded case; missing or invalid
    // records and altered aggregate fields fail closed under the pinned release policy.
    if !inprocess_solver_certificate_matches_recomputation(&evidence.inprocess_solver) {
        failures.push(
            "in-process sparse solver certificate does not match pinned release recomputation or contains invalid/omitted cases"
                .to_string(),
        );
    }
    // Recompute floating-base QP summary metrics from raw report cases under
    // the release-pinned criteria. Legacy summary-only evidence fails closed.
    if !floating_base_qp_certificate_matches_recomputation(
        &evidence.floating_base_qp,
        &evidence.floating_base_qp_cases,
    ) {
        failures.push(
            "floating-base QP certificate does not match recomputation from raw cases under pinned release criteria"
                .to_string(),
        );
    }
    // Recompute embodied qualification from the raw cases and frozen manifest.
    // A serialized "accepted": true flag is only a claim until these inputs pass again.
    let recomputed_embodied = certify_embodied_control_against_manifest(
        &evidence.embodied_control_cases,
        EmbodiedCertificationCriteria::default(),
        &evidence.embodied_scenario_manifest,
    );
    let supplied_embodied_bytes = serde_json::to_vec(&evidence.embodied_control).ok();
    let recomputed_embodied_bytes = serde_json::to_vec(&recomputed_embodied).ok();
    if supplied_embodied_bytes.is_none()
        || supplied_embodied_bytes != recomputed_embodied_bytes
    {
        failures.push("serialized embodied certificate does not match release-gate recomputation".to_string());
    }
    if !recomputed_embodied.accepted {
        failures.push("release-gate recomputation did not qualify embodied control".to_string());
    }
    for (label, actual, expected) in [
        (
            "controller",
            actual_controller_artifact_sha256,
            recomputed_embodied.controller_artifact_sha256.as_str(),
        ),
        (
            "test harness",
            actual_test_harness_sha256,
            recomputed_embodied.test_harness_sha256.as_str(),
        ),
        (
            "scenario configuration",
            actual_scenario_config_sha256,
            recomputed_embodied.scenario_config_sha256.as_str(),
        ),
    ] {
        if !artifact_sha256_matches(actual, expected) {
            failures.push(format!(
                "actual {label} artifact bytes do not match the frozen scenario manifest"
            ));
        }
    }
    if recomputed_embodied.schema_version != 5 {
        failures.push("embodied-control certificate schema is not v5".to_string());
    }
    if recomputed_embodied.scenario_manifest_digest != trusted_scenario_manifest_digest {
        failures.push("scenario manifest differs from the trusted external manifest anchor".to_string());
    }
    if !is_lower_hex_blake3(&recomputed_embodied.corpus_digest) {
        failures.push("embodied-control certificate omitted a canonical corpus digest".to_string());
    }
    if !recomputed_embodied.manifest_verified
        || !is_lower_hex_blake3(&recomputed_embodied.scenario_manifest_digest)
        || recomputed_embodied.scenario_manifest_id.trim().is_empty()
    {
        failures.push("embodied-control evidence is not bound to a verified frozen scenario manifest".to_string());
    }
    if recomputed_embodied.total_cases == 0 {
        failures.push("embodied-control certificate contains no cases".to_string());
    }
    if recomputed_embodied.candidate_build_id != evidence.candidate_build_id {
        failures.push("embodied-control certificate belongs to a different candidate build".to_string());
    }
    if recomputed_embodied.model_artifact_sha256 != evidence.model_artifact_sha256 {
        failures.push("embodied-control certificate belongs to a different model artifact".to_string());
    }
    // Recompute recovery metrics from the complete recorded matrix under the
    // release-pinned protocol/configuration; do not trust supplied rates or pass flag.
    if !adversarial_recovery_certificate_matches_recomputation(
        &evidence.adversarial_recovery,
    ) {
        failures.push(
            "adversarial recovery certificate does not match recomputation from the exact case matrix under pinned release policy"
                .to_string(),
        );
    }

    let solver_backend_id = evidence.inprocess_solver.backend_id.clone();
    if let Some(expected) = solver_backend_id.as_deref() {
        if evidence.floating_base_qp.backend_ids.len() != 1
            || evidence.floating_base_qp.backend_ids[0] != expected
        {
            failures.push(
                "solver qualification and floating-base campaign name different backends"
                    .to_string(),
            );
        }
    } else {
        failures.push("solver qualification omitted a stable backend identity".to_string());
    }
    if evidence.oracle_generation.dataset_id.trim().is_empty() {
        failures.push("oracle generation omitted a dataset identity".to_string());
    }
    if evidence.dynamics_oracle.valid_cases < evidence.oracle_generation.admitted_cases {
        failures.push(
            "dynamics-oracle comparison covers fewer cases than the admitted dataset".to_string(),
        );
    }
    if evidence.embodied_control.scenario_fingerprint == 0
        || evidence.adversarial_recovery.scenario_fingerprint == 0
    {
        failures.push("scenario evidence contains an empty fingerprint".to_string());
    }

    HumanoidControlReleaseCertificate {
        schema_version: 7,
        release_id: evidence.release_id.clone(),
        candidate_build_id: evidence.candidate_build_id.clone(),
        model_artifact_sha256: actual_model_artifact_sha256.to_string(),
        evidence_bundle_sha256: is_lower_hex_sha256(actual_evidence_bundle_sha256)
            .then(|| actual_evidence_bundle_sha256.to_string())
            .unwrap_or_default(),
        oracle_generator_executable_sha256: is_lower_hex_sha256(actual_oracle_executable_sha256)
            .then(|| actual_oracle_executable_sha256.to_string()),
        controller_artifact_sha256: recomputed_embodied.controller_artifact_sha256.clone(),
        simulator_id: recomputed_embodied.simulator_id.clone(),
        simulator_version: recomputed_embodied.simulator_version.clone(),
        test_harness_sha256: recomputed_embodied.test_harness_sha256.clone(),
        scenario_config_sha256: recomputed_embodied.scenario_config_sha256.clone(),
        embodied_corpus_digest: recomputed_embodied.corpus_digest.clone(),
        embodied_scenario_manifest_digest: recomputed_embodied.scenario_manifest_digest.clone(),
        solver_backend_id,
        oracle_dataset_id: evidence.oracle_generation.dataset_id.clone(),
        generated_unix_millis: evidence.generated_unix_millis,
        accepted: failures.is_empty(),
        failures,
    }
}

fn oracle_generation_certificate_matches_recomputation(
    supplied: &IndependentOracleGenerationCertificate,
    dataset: Option<&DynamicsOracleDataset>,
    candidate_build_id: &str,
    model_artifact_sha256: &str,
) -> bool {
    let Some(dataset) = dataset else {
        return false;
    };
    let recomputed = recompute_independent_oracle_generation_certificate(
        supplied,
        dataset,
        candidate_build_id,
        model_artifact_sha256,
    );
    recomputed.passed
        && supplied.passed
        && serde_json::to_vec(supplied).ok() == serde_json::to_vec(&recomputed).ok()
}

fn dynamics_oracle_certificate_matches_recomputation(
    supplied: &DynamicsOracleCertificate,
    inputs: &[DynamicsOracleInputCase],
    dataset: Option<&DynamicsOracleDataset>,
) -> bool {
    let Some(dataset) = dataset else {
        return false;
    };
    if !dynamics_oracle_inputs_match_dataset(inputs, dataset) {
        return false;
    }
    let recomputed = certify_dynamics_oracle_snapshots(inputs);
    recomputed.passed
        && recomputed.valid_cases == inputs.len()
        && supplied.passed
        && serde_json::to_vec(supplied).ok() == serde_json::to_vec(&recomputed).ok()
}

fn dynamics_oracle_inputs_match_dataset(
    inputs: &[DynamicsOracleInputCase],
    dataset: &DynamicsOracleDataset,
) -> bool {
    dataset.cases.len() == inputs.len()
        && dataset.cases.iter().all(|dataset_case| {
            let matches = inputs
                .iter()
                .filter(|input| {
                    input.case_id == dataset_case.case_id
                        && input.sampled_state_fingerprint == dataset_case.state_fingerprint
                })
                .collect::<Vec<_>>();
            matches.len() == 1
                && serde_json::to_vec(&matches[0].oracle_snapshot).ok()
                    == serde_json::to_vec(&dataset_case.oracle).ok()
        })
}

fn adversarial_recovery_certificate_matches_recomputation(
    supplied: &AdversarialRecoveryCertificate,
) -> bool {
    let recomputed = certify_adversarial_recovery_cases(
        supplied.cases.clone(),
        &PushRecoveryProtocol::default(),
        &AdversarialRecoveryCertificationConfig::default(),
    );
    recomputed.passed
        && supplied.passed
        && serde_json::to_vec(supplied).ok() == serde_json::to_vec(&recomputed).ok()
}

fn floating_base_qp_certificate_matches_recomputation(
    supplied: &FloatingBaseQpCertificate,
    cases: &[FloatingBaseQpCase],
) -> bool {
    let recomputed = certify_floating_base_qp(cases, FloatingBaseQpCriteria::default());
    recomputed.passed
        && recomputed.valid_cases == cases.len()
        && supplied.passed
        && serde_json::to_vec(supplied).ok() == serde_json::to_vec(&recomputed).ok()
}

fn inprocess_solver_certificate_matches_recomputation(
    supplied: &InProcessSparseQpQualificationCertificate,
) -> bool {
    // The release bundle must contain one result for every attempted case, with
    // no invalid records silently filtered out by the summary calculator.
    if supplied.total_cases != supplied.cases.len() {
        return false;
    }
    let recomputed = certify_inprocess_sparse_qp_cases(
        supplied.cases.len(),
        supplied.cases.clone(),
        InProcessSparseQpQualificationCriteria::default(),
    );
    recomputed.passed
        && recomputed.valid_cases == supplied.cases.len()
        && supplied.passed
        && serde_json::to_vec(supplied).ok() == serde_json::to_vec(&recomputed).ok()
}

fn oracle_executable_identity_matches(
    actual_sha256: &str,
    trusted_sha256: &str,
    generation_receipt_sha256: Option<&str>,
) -> bool {
    is_lower_hex_sha256(actual_sha256)
        && is_lower_hex_sha256(trusted_sha256)
        && actual_sha256 == trusted_sha256
        && generation_receipt_sha256 == Some(trusted_sha256)
}

fn is_lower_hex_blake3(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn artifact_sha256_matches(actual: &str, expected: &str) -> bool {
    is_lower_hex_sha256(actual)
        && is_lower_hex_sha256(expected)
        && actual == expected
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_artifact_hashes_must_match_actual_bytes_and_frozen_manifest() {
        let expected = "a".repeat(64);
        let other = "b".repeat(64);
        assert!(artifact_sha256_matches(&expected, &expected));
        assert!(!artifact_sha256_matches(&other, &expected));
        assert!(!artifact_sha256_matches("not-a-digest", &expected));
        assert!(!artifact_sha256_matches(&expected, "not-a-digest"));
        assert!(!artifact_sha256_matches(&expected, &other));
    }

    #[test]
    fn inprocess_solver_release_verifier_recomputes_case_aggregates_and_pass_bit() {
        use crate::inprocess_solver_certification::{
            certify_inprocess_sparse_qp_cases, InProcessSparseQpQualificationCase,
        };

        let cases = (0..512)
            .map(|index| InProcessSparseQpQualificationCase {
                case_id: format!("case-{index}"),
                backend_id: "release-test-backend".to_string(),
                state_fingerprint: index as u64 + 1,
                structure_fingerprint: index as u64 % 3 + 1,
                admitted: true,
                converged: true,
                warm_start_requested: true,
                warm_start_used: true,
                symbolic_pattern_reused: true,
                iterations: 10,
                maximum_equality_residual: 1.0e-8,
                maximum_bound_violation: 1.0e-8,
                elapsed_micros: 100,
            })
            .collect::<Vec<_>>();
        let certificate = certify_inprocess_sparse_qp_cases(
            cases.len(),
            cases,
            InProcessSparseQpQualificationCriteria::default(),
        );
        assert!(certificate.passed, "{:?}", certificate.failures);
        assert!(inprocess_solver_certificate_matches_recomputation(&certificate));

        let mut forged_pass = certificate.clone();
        forged_pass.passed = false;
        assert!(!inprocess_solver_certificate_matches_recomputation(
            &forged_pass
        ));

        let mut forged_summary = certificate.clone();
        forged_summary.admitted_cases += 1;
        assert!(!inprocess_solver_certificate_matches_recomputation(
            &forged_summary
        ));

        let mut omitted_case_claim = certificate.clone();
        omitted_case_claim.total_cases += 1;
        assert!(!inprocess_solver_certificate_matches_recomputation(
            &omitted_case_claim
        ));

        let mut invalid_case_bundle = certificate.clone();
        invalid_case_bundle.cases[0].state_fingerprint = 0;
        assert!(!inprocess_solver_certificate_matches_recomputation(
            &invalid_case_bundle
        ));
    }

    #[test]
    fn floating_base_qp_release_verifier_recomputes_raw_case_summary() {
        use crate::control_budget::SolverBudgetEvidence;
        use crate::floating_base_inverse_dynamics::FloatingBaseInverseDynamicsReport;

        let cases = (0..100)
            .map(|index| FloatingBaseQpCase {
                case_id: format!("qp-case-{index}"),
                state_fingerprint: index as u64 + 1,
                report: FloatingBaseInverseDynamicsReport {
                    solver_derived_model: true,
                    converged: true,
                    used_fallback: false,
                    active_contacts: 2,
                    active_set_iterations: 3,
                    warm_start_used: true,
                    warm_start_active_bounds: 2,
                    qp_structure_fingerprint: 7,
                    solver_backend_id: Some("release-test-backend".to_string()),
                    symbolic_pattern_reused: true,
                    maximum_dynamics_residual: 1.0e-6,
                    maximum_contact_acceleration_residual: 1.0e-6,
                    maximum_friction_utilization: 0.5,
                    objective: 1.0,
                    budget: SolverBudgetEvidence {
                        admitted: true,
                        elapsed_micros: 100,
                        deadline_missed: false,
                        estimated_operations: 1_000,
                        variables: 10,
                        constraints: 5,
                    },
                    model_id: Some("test-model".to_string()),
                },
            })
            .collect::<Vec<_>>();
        let certificate = certify_floating_base_qp(&cases, FloatingBaseQpCriteria::default());
        assert!(certificate.passed, "{:?}", certificate.failures);
        assert!(floating_base_qp_certificate_matches_recomputation(
            &certificate, &cases
        ));

        let mut forged_pass = certificate.clone();
        forged_pass.passed = false;
        assert!(!floating_base_qp_certificate_matches_recomputation(
            &forged_pass, &cases
        ));

        let mut forged_summary = certificate.clone();
        forged_summary.valid_cases += 1;
        assert!(!floating_base_qp_certificate_matches_recomputation(
            &forged_summary, &cases
        ));

        assert!(!floating_base_qp_certificate_matches_recomputation(
            &certificate, &cases[..99]
        ));
    }

    #[test]
    fn recovery_release_verifier_recomputes_metrics_and_requires_exact_matrix() {
        use crate::recovery_certification::{certify_adversarial_recovery_cases, TimedRecoveryCase};
        use crate::recovery_benchmark::{PushDirection, PushRecoveryResult};

        let config = AdversarialRecoveryCertificationConfig::default();
        let cases = config
            .disturbance_times_s
            .iter()
            .flat_map(|time| {
                config.force_levels_n.iter().flat_map(move |force| {
                    PushDirection::ALL.into_iter().map(move |direction| TimedRecoveryCase {
                        disturbance_time_s: *time,
                        direction,
                        force_n: *force,
                        result: PushRecoveryResult {
                            recovered: true,
                            fell: false,
                            recovery_time_s: Some(0.5),
                            min_uprightness: 0.95,
                            min_capture_margin_m: 0.05,
                            peak_recovery_effort: 0.1,
                            recovery_interventions: 1,
                        },
                    })
                })
            })
            .collect::<Vec<_>>();
        let certificate = certify_adversarial_recovery_cases(
            cases,
            &PushRecoveryProtocol::default(),
            &config,
        );
        assert!(certificate.passed, "{:?}", certificate.failure_reasons);
        assert!(adversarial_recovery_certificate_matches_recomputation(
            &certificate
        ));

        let mut forged_summary = certificate.clone();
        forged_summary.recovery_rate = 0.0;
        assert!(!adversarial_recovery_certificate_matches_recomputation(
            &forged_summary
        ));

        let mut omitted_case = certificate.clone();
        omitted_case.cases.pop();
        assert!(!adversarial_recovery_certificate_matches_recomputation(
            &omitted_case
        ));
    }

    #[test]
    fn oracle_worker_trust_anchor_requires_all_three_digests_to_match() {
        let digest = "a".repeat(64);
        assert!(oracle_executable_identity_matches(
            &digest,
            &digest,
            Some(digest.as_str()),
        ));
        assert!(!oracle_executable_identity_matches(
            &"b".repeat(64),
            &digest,
            Some(digest.as_str()),
        ));
        assert!(!oracle_executable_identity_matches(&digest, &digest, None));
        assert!(!oracle_executable_identity_matches(
            &digest,
            &"A".repeat(64),
            Some(digest.as_str()),
        ));
    }

    #[test]
    fn corpus_digest_requires_canonical_lowercase_256_bit_hex() {
        assert!(is_lower_hex_blake3(&"a".repeat(64)));
        assert!(!is_lower_hex_blake3(""));
        assert!(!is_lower_hex_blake3(&"A".repeat(64)));
        assert!(!is_lower_hex_blake3(&"g".repeat(64)));
    }

    #[test]
    fn malformed_model_identity_fails_before_any_claim() {
        let certificate = HumanoidControlReleaseCertificate {
            schema_version: 7,
            release_id: "release".to_string(),
            candidate_build_id: "candidate".to_string(),
            model_artifact_sha256: "not-a-sha".to_string(),
            evidence_bundle_sha256: String::new(),
            oracle_generator_executable_sha256: None,
            controller_artifact_sha256: String::new(),
            simulator_id: String::new(),
            simulator_version: String::new(),
            test_harness_sha256: String::new(),
            scenario_config_sha256: String::new(),
            embodied_corpus_digest: String::new(),
            embodied_scenario_manifest_digest: String::new(),
            solver_backend_id: None,
            oracle_dataset_id: "dataset".to_string(),
            generated_unix_millis: 1,
            accepted: false,
            failures: vec!["model artifact identity is not canonical SHA-256 hex".to_string()],
        };
        assert!(!certificate.accepted);
        assert!(!is_lower_hex_sha256(&certificate.model_artifact_sha256));
    }
}
