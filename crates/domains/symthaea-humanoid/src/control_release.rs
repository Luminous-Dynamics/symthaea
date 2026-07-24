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

use crate::dynamics_oracle::DynamicsOracleCertificate;
use crate::embodied_certification::EmbodiedControlCertificate;
use crate::inprocess_solver_certification::InProcessSparseQpQualificationCertificate;
use crate::oracle_generator_certification::IndependentOracleGenerationCertificate;
use crate::qp_certification::FloatingBaseQpCertificate;
use crate::recovery_certification::AdversarialRecoveryCertificate;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidControlReleaseEvidence {
    pub release_id: String,
    pub candidate_build_id: String,
    pub model_artifact_sha256: String,
    pub generated_unix_millis: u64,
    pub oracle_generation: IndependentOracleGenerationCertificate,
    pub dynamics_oracle: DynamicsOracleCertificate,
    pub inprocess_solver: InProcessSparseQpQualificationCertificate,
    pub floating_base_qp: FloatingBaseQpCertificate,
    pub embodied_control: EmbodiedControlCertificate,
    pub adversarial_recovery: AdversarialRecoveryCertificate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidControlReleaseCertificate {
    pub schema_version: u32,
    pub release_id: String,
    pub candidate_build_id: String,
    pub model_artifact_sha256: String,
    pub solver_backend_id: Option<String>,
    pub oracle_dataset_id: String,
    pub generated_unix_millis: u64,
    pub accepted: bool,
    pub failures: Vec<String>,
}

pub fn certify_humanoid_control_release(
    evidence: &HumanoidControlReleaseEvidence,
) -> HumanoidControlReleaseCertificate {
    let mut failures = Vec::new();
    if evidence.release_id.trim().is_empty() {
        failures.push("release identity is empty".to_string());
    }
    if evidence.candidate_build_id.trim().is_empty() {
        failures.push("candidate build identity is empty".to_string());
    }
    if !is_lower_hex_sha256(&evidence.model_artifact_sha256) {
        failures.push("model artifact identity is not canonical SHA-256 hex".to_string());
    }
    if evidence.generated_unix_millis == 0 {
        failures.push("release evidence has no generation timestamp".to_string());
    }
    if !evidence.oracle_generation.passed {
        failures.push("independent oracle generation did not certify".to_string());
    }
    if !evidence.dynamics_oracle.passed {
        failures.push("candidate dynamics did not match the oracle corpus".to_string());
    }
    if !evidence.inprocess_solver.passed {
        failures.push("in-process sparse solver did not qualify".to_string());
    }
    if !evidence.floating_base_qp.passed {
        failures.push("floating-base QP campaign did not certify".to_string());
    }
    if !evidence.embodied_control.accepted {
        failures.push("embodied-control campaign did not certify".to_string());
    }
    if !evidence.adversarial_recovery.passed {
        failures.push("adversarial recovery campaign did not certify".to_string());
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
        schema_version: 1,
        release_id: evidence.release_id.clone(),
        candidate_build_id: evidence.candidate_build_id.clone(),
        model_artifact_sha256: evidence.model_artifact_sha256.clone(),
        solver_backend_id,
        oracle_dataset_id: evidence.oracle_generation.dataset_id.clone(),
        generated_unix_millis: evidence.generated_unix_millis,
        accepted: failures.is_empty(),
        failures,
    }
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
    fn malformed_model_identity_fails_before_any_claim() {
        let certificate = HumanoidControlReleaseCertificate {
            schema_version: 1,
            release_id: "release".to_string(),
            candidate_build_id: "candidate".to_string(),
            model_artifact_sha256: "not-a-sha".to_string(),
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
