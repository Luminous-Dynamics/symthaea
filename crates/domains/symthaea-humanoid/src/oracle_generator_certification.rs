// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chain-of-custody evidence for independently generated MuJoCo oracle data.
//!
//! A dataset is admitted only when every requested state is answered by one
//! stable generator/build/engine identity, all response fingerprints match,
//! contact ordering remains stable, and the generator is distinct from the
//! candidate build under test.

use std::collections::BTreeSet;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::mujoco_oracle_protocol::{MujocoOracleWorkerRequest, ProcessMujocoOracleGenerator};
use crate::oracle_dataset::{
    DynamicsOracleDataset, DynamicsOracleDatasetBuilder, DynamicsOracleDatasetManifest,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentOracleGenerationCase {
    pub request_id: u64,
    pub state_fingerprint: u64,
    pub generator_id: String,
    pub generator_build_id: String,
    pub engine_id: String,
    pub model_id: Option<String>,
    pub admitted: bool,
    pub elapsed_micros: u64,
    pub failure: Option<String>,
}

impl IndependentOracleGenerationCase {
    pub fn validate(&self) -> bool {
        self.request_id != 0
            && self.state_fingerprint != 0
            && !self.generator_id.trim().is_empty()
            && !self.generator_build_id.trim().is_empty()
            && !self.engine_id.trim().is_empty()
            && self.elapsed_micros > 0
            && self.admitted == (self.failure.is_none() && self.model_id.is_some())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IndependentOracleGenerationCriteria {
    pub minimum_cases: usize,
    pub minimum_distinct_states: usize,
    pub maximum_case_elapsed_micros: u64,
    pub require_all_cases: bool,
}

impl Default for IndependentOracleGenerationCriteria {
    fn default() -> Self {
        Self {
            minimum_cases: 64,
            minimum_distinct_states: 64,
            maximum_case_elapsed_micros: 5_000_000,
            require_all_cases: true,
        }
    }
}

impl IndependentOracleGenerationCriteria {
    pub fn validate(&self) -> bool {
        self.minimum_cases > 0
            && self.minimum_distinct_states > 0
            && self.maximum_case_elapsed_micros > 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentOracleGenerationCertificate {
    pub schema_version: u32,
    pub dataset_id: String,
    pub generator_id: Option<String>,
    pub generator_build_id: Option<String>,
    /// SHA-256 from the frozen dataset manifest, checked against the configured worker.
    pub generator_executable_sha256: Option<String>,
    pub engine_id: Option<String>,
    pub total_requests: usize,
    pub valid_cases: usize,
    pub admitted_cases: usize,
    pub distinct_states: usize,
    pub distinct_models: usize,
    pub maximum_case_elapsed_micros: u64,
    pub passed: bool,
    pub failures: Vec<String>,
    pub cases: Vec<IndependentOracleGenerationCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentOracleGenerationResult {
    pub certificate: IndependentOracleGenerationCertificate,
    pub dataset: Option<DynamicsOracleDataset>,
}

pub fn generate_independent_oracle_dataset(
    generator: &ProcessMujocoOracleGenerator,
    manifest: DynamicsOracleDatasetManifest,
    candidate_build_id: &str,
    requests: &[MujocoOracleWorkerRequest],
    criteria: IndependentOracleGenerationCriteria,
) -> IndependentOracleGenerationResult {
    let worker_digest_matches_manifest = generator.expected_executable_sha256()
        == Some(manifest.generator_executable_sha256.as_str());
    let mut builder = DynamicsOracleDatasetBuilder::new(manifest.clone());
    let mut cases = Vec::new();
    for request in requests {
        let request_identity_matches = request.validate()
            && request.generator_id == manifest.generator_id
            && request.generator_build_id == manifest.generator_build_id
            && request.engine_id == manifest.engine_id
            && request.candidate_build_id == candidate_build_id;
        let start = Instant::now();
        let response = if worker_digest_matches_manifest && request_identity_matches {
            generator.generate(request)
        } else {
            None
        };
        let elapsed_micros = start.elapsed().as_micros().max(1) as u64;
        let mut evidence = IndependentOracleGenerationCase {
            request_id: request.request_id,
            state_fingerprint: request.state_fingerprint,
            generator_id: request.generator_id.clone(),
            generator_build_id: request.generator_build_id.clone(),
            engine_id: request.engine_id.clone(),
            model_id: None,
            admitted: false,
            elapsed_micros,
            failure: None,
        };
        let failure = if !request.validate() {
            Some("oracle request failed validation".to_string())
        } else if request.generator_id != manifest.generator_id
            || request.generator_build_id != manifest.generator_build_id
            || request.engine_id != manifest.engine_id
            || request.candidate_build_id != candidate_build_id
        {
            Some("oracle request identity differs from the dataset manifest".to_string())
        } else if !worker_digest_matches_manifest {
            Some("oracle executable digest differs from the frozen dataset manifest".to_string())
        } else if let Some(response) = response {
            if let Some(oracle) = response.oracle {
                evidence.model_id = Some(oracle.model_id.clone());
                match builder.as_mut() {
                    Some(builder) => {
                        if builder.push_case(
                            format!("oracle-{}", request.request_id),
                            request.generalized_position.clone(),
                            request.generalized_velocity.clone(),
                            request.actuator_command.clone(),
                            oracle,
                        ) {
                            evidence.admitted = true;
                            None
                        } else {
                            Some("oracle case was rejected by the dataset builder".to_string())
                        }
                    }
                    None => Some("oracle case was rejected by the dataset builder".to_string()),
                }
            } else {
                Some("oracle response omitted a dynamics snapshot".to_string())
            }
        } else {
            Some("independent oracle worker failed or returned invalid evidence".to_string())
        };
        evidence.failure = failure;
        cases.push(evidence);
    }

    let dataset = builder.and_then(|builder| builder.finish(candidate_build_id));
    let certificate = certify_generation_cases(
        &manifest,
        candidate_build_id,
        requests.len(),
        cases,
        criteria,
        dataset.is_some(),
    );
    IndependentOracleGenerationResult {
        dataset: certificate.passed.then_some(()).and(dataset),
        certificate,
    }
}

/// Recompute an oracle-generation receipt against its exact stored dataset.
/// The dataset must contain exactly the admitted receipt rows, with matching
/// request-derived case IDs, state fingerprints, model IDs, manifest identity,
/// and the independently pinned model/worker artifacts.
pub fn recompute_independent_oracle_generation_certificate(
    supplied: &IndependentOracleGenerationCertificate,
    dataset: &DynamicsOracleDataset,
    candidate_build_id: &str,
    expected_model_artifact_sha256: &str,
) -> IndependentOracleGenerationCertificate {
    let manifest = &dataset.manifest;
    let dataset_valid = supplied.schema_version == 2
        && dataset.validate_for_candidate(candidate_build_id)
        && is_lower_hex_sha256(expected_model_artifact_sha256)
        && manifest.model_artifact_sha256 == expected_model_artifact_sha256
        && manifest.dataset_id == supplied.dataset_id
        && supplied.generator_executable_sha256.as_deref()
            == Some(manifest.generator_executable_sha256.as_str())
        && generation_receipt_matches_dataset(&supplied.cases, dataset);
    certify_generation_cases(
        manifest,
        candidate_build_id,
        supplied.total_requests,
        supplied.cases.clone(),
        IndependentOracleGenerationCriteria::default(),
        dataset_valid,
    )
}

fn generation_receipt_matches_dataset(
    cases: &[IndependentOracleGenerationCase],
    dataset: &DynamicsOracleDataset,
) -> bool {
    if cases.iter().map(|case| case.request_id).collect::<BTreeSet<_>>().len() != cases.len()
        || cases
            .iter()
            .map(|case| case.state_fingerprint)
            .collect::<BTreeSet<_>>()
            .len()
            != cases.len()
        || dataset.cases.len() != cases.iter().filter(|case| case.admitted).count()
    {
        return false;
    }
    cases.iter().all(|case| {
        let expected_case_id = format!("oracle-{}", case.request_id);
        let matching_case = dataset.cases.iter().find(|dataset_case| {
            dataset_case.case_id == expected_case_id
                || dataset_case.state_fingerprint == case.state_fingerprint
        });
        if case.admitted {
            matching_case.is_some_and(|dataset_case| {
                dataset_case.case_id == expected_case_id
                    && dataset_case.state_fingerprint == case.state_fingerprint
                    && case.model_id.as_deref() == Some(dataset_case.oracle.model_id.as_str())
            })
        } else {
            matching_case.is_none()
        }
    })
}

fn certify_generation_cases(
    manifest: &DynamicsOracleDatasetManifest,
    candidate_build_id: &str,
    total_requests: usize,
    cases: Vec<IndependentOracleGenerationCase>,
    criteria: IndependentOracleGenerationCriteria,
    dataset_valid: bool,
) -> IndependentOracleGenerationCertificate {
    let valid = cases
        .iter()
        .filter(|case| case.validate())
        .collect::<Vec<_>>();
    let generator_ids = valid
        .iter()
        .map(|case| case.generator_id.as_str())
        .collect::<BTreeSet<_>>();
    let generator_build_ids = valid
        .iter()
        .map(|case| case.generator_build_id.as_str())
        .collect::<BTreeSet<_>>();
    let engine_ids = valid
        .iter()
        .map(|case| case.engine_id.as_str())
        .collect::<BTreeSet<_>>();
    let generator_id = exactly_one(&generator_ids);
    let generator_build_id = exactly_one(&generator_build_ids);
    let engine_id = exactly_one(&engine_ids);
    let admitted_cases = valid.iter().filter(|case| case.admitted).count();
    let distinct_request_ids = valid
        .iter()
        .map(|case| case.request_id)
        .collect::<BTreeSet<_>>()
        .len();
    let distinct_states = valid
        .iter()
        .map(|case| case.state_fingerprint)
        .collect::<BTreeSet<_>>()
        .len();
    let distinct_models = valid
        .iter()
        .filter_map(|case| case.model_id.as_deref())
        .collect::<BTreeSet<_>>()
        .len();
    let maximum_case_elapsed_micros = valid
        .iter()
        .map(|case| case.elapsed_micros)
        .max()
        .unwrap_or(0);

    let mut failures = Vec::new();
    if !criteria.validate() {
        failures.push("independent oracle generation criteria are invalid".to_string());
    }
    if !manifest.validate_for_candidate(candidate_build_id) {
        failures.push("oracle manifest is invalid or self-generated".to_string());
    }
    if generator_build_id.is_some()
        && !is_lower_hex_sha256(&manifest.generator_executable_sha256)
    {
        failures.push("oracle manifest omitted a canonical worker executable digest".to_string());
    }
    if generator_id.as_deref() != Some(manifest.generator_id.as_str())
        || generator_build_id.as_deref() != Some(manifest.generator_build_id.as_str())
        || engine_id.as_deref() != Some(manifest.engine_id.as_str())
    {
        failures.push("oracle cases do not preserve one manifest identity".to_string());
    }
    if total_requests != cases.len() || valid.len() != cases.len() {
        failures.push("oracle receipt omitted requests or included invalid case records".to_string());
    }
    if distinct_request_ids != valid.len() {
        failures.push("oracle receipt contains duplicate request IDs".to_string());
    }
    if valid.iter().map(|case| case.state_fingerprint).collect::<BTreeSet<_>>().len()
        != valid.len()
    {
        failures.push("oracle receipt contains duplicate state fingerprints".to_string());
    }
    if valid.len() < criteria.minimum_cases {
        failures.push(format!(
            "only {} valid oracle cases were produced; {} required",
            valid.len(),
            criteria.minimum_cases
        ));
    }
    if distinct_states < criteria.minimum_distinct_states {
        failures.push(format!(
            "only {distinct_states} distinct states were produced; {} required",
            criteria.minimum_distinct_states
        ));
    }
    if criteria.require_all_cases && admitted_cases != valid.len() {
        failures.push(format!(
            "{} valid oracle cases were not admitted",
            valid.len().saturating_sub(admitted_cases)
        ));
    }
    if maximum_case_elapsed_micros > criteria.maximum_case_elapsed_micros {
        failures.push(format!(
            "oracle case latency {maximum_case_elapsed_micros}us exceeds {}us",
            criteria.maximum_case_elapsed_micros
        ));
    }
    if !dataset_valid {
        failures.push("the assembled oracle dataset failed final admission".to_string());
    }

    IndependentOracleGenerationCertificate {
        schema_version: 2,
        dataset_id: manifest.dataset_id.clone(),
        generator_id,
        generator_build_id,
        generator_executable_sha256: is_lower_hex_sha256(&manifest.generator_executable_sha256)
            .then(|| manifest.generator_executable_sha256.clone()),
        engine_id,
        total_requests,
        valid_cases: valid.len(),
        admitted_cases,
        distinct_states,
        distinct_models,
        maximum_case_elapsed_micros,
        passed: failures.is_empty(),
        failures,
        cases,
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn exactly_one(values: &BTreeSet<&str>) -> Option<String> {
    (values.len() == 1)
        .then(|| values.first().map(|value| (*value).to_string()))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle_snapshot() -> crate::floating_base::FloatingBaseDynamicsSnapshot {
        use crate::floating_base::{FLOATING_BASE_DOF, FloatingBaseDynamicsSnapshot};
        use crate::full_dynamics::{DynamicsProvenance, SpatialContactJacobian};
        use crate::morphology::HumanoidMorphology;

        let morphology = HumanoidMorphology::Dmc21;
        let nv = FLOATING_BASE_DOF + morphology.num_actuators();
        let mut mass_matrix = vec![0.0; nv * nv];
        for index in 0..nv {
            mass_matrix[index * nv + index] = 1.0 + index as f64 * 0.1;
        }
        FloatingBaseDynamicsSnapshot {
            morphology,
            sampled_at_s: 1.0,
            total_mass_kg: 70.0,
            gravity_world_mps2: [0.0, 0.0, -9.81],
            generalized_velocity_count: nv,
            mass_matrix,
            bias_force: vec![0.25; nv],
            actuator_velocity_indices: (FLOATING_BASE_DOF..nv).collect(),
            torque_limits_nm: vec![100.0; morphology.num_actuators()],
            centroidal_momentum_matrix: vec![0.5; 6 * nv],
            contacts: vec![SpatialContactJacobian {
                site_id: "r_foot_site".to_string(),
                rows: std::array::from_fn(|row| vec![row as f64 * 0.01; nv]),
                confidence: 1.0,
            }],
            provenance: DynamicsProvenance::mujoco_solver_with_morphology_limits(),
            model_id: "oracle-model".to_string(),
        }
    }

    fn dataset_manifest() -> DynamicsOracleDatasetManifest {
        let morphology = crate::morphology::HumanoidMorphology::Dmc21;
        DynamicsOracleDatasetManifest {
            schema_version: 2,
            dataset_id: "release-dataset".to_string(),
            generator_id: "independent-generator".to_string(),
            generator_build_id: "generator-build".to_string(),
            generator_executable_sha256: "e".repeat(64),
            engine_id: "mujoco".to_string(),
            model_artifact_sha256: "a".repeat(64),
            generalized_coordinate_order: (0..6 + morphology.num_actuators())
                .map(|index| format!("v{index}"))
                .collect(),
            morphology,
            contact_site_order: vec!["r_foot_site".to_string()],
            generated_unix_millis: 1,
        }
    }

    #[test]
    fn release_receipt_must_match_exact_dataset_rows_and_hashes() {
        use crate::oracle_dataset::DynamicsOracleDatasetBuilder;

        let manifest = dataset_manifest();
        let mut builder = DynamicsOracleDatasetBuilder::new(manifest.clone()).unwrap();
        let nv = 6 + manifest.morphology.num_actuators();
        let actuators = manifest.morphology.num_actuators();
        for index in 0..64 {
            assert!(builder.push_case(
                format!("oracle-{}", index + 1),
                vec![index as f64 + 1.0; nv + 1],
                vec![index as f64 * 0.01; nv],
                vec![0.1; actuators],
                oracle_snapshot(),
            ));
        }
        let dataset = builder.finish("candidate-build").unwrap();
        let receipt_cases = dataset
            .cases
            .iter()
            .enumerate()
            .map(|(index, case)| IndependentOracleGenerationCase {
                request_id: index as u64 + 1,
                state_fingerprint: case.state_fingerprint,
                generator_id: manifest.generator_id.clone(),
                generator_build_id: manifest.generator_build_id.clone(),
                engine_id: manifest.engine_id.clone(),
                model_id: Some(case.oracle.model_id.clone()),
                admitted: true,
                elapsed_micros: 100,
                failure: None,
            })
            .collect::<Vec<_>>();
        let certificate = certify_generation_cases(
            &manifest,
            "candidate-build",
            receipt_cases.len(),
            receipt_cases,
            IndependentOracleGenerationCriteria::default(),
            true,
        );
        assert!(certificate.passed, "{:?}", certificate.failures);

        let recomputed = recompute_independent_oracle_generation_certificate(
            &certificate,
            &dataset,
            "candidate-build",
            &manifest.model_artifact_sha256,
        );
        assert!(recomputed.passed, "{:?}", recomputed.failures);
        assert_eq!(
            serde_json::to_vec(&certificate).unwrap(),
            serde_json::to_vec(&recomputed).unwrap(),
        );

        let mut altered_receipt = certificate.clone();
        altered_receipt.cases[0].state_fingerprint ^= 1;
        let rejected = recompute_independent_oracle_generation_certificate(
            &altered_receipt,
            &dataset,
            "candidate-build",
            &manifest.model_artifact_sha256,
        );
        assert!(!rejected.passed);

        let mut truncated_dataset = dataset.clone();
        truncated_dataset.cases.pop();
        let rejected = recompute_independent_oracle_generation_certificate(
            &certificate,
            &truncated_dataset,
            "candidate-build",
            &manifest.model_artifact_sha256,
        );
        assert!(!rejected.passed);

        let rejected = recompute_independent_oracle_generation_certificate(
            &certificate,
            &dataset,
            "candidate-build",
            &"b".repeat(64),
        );
        assert!(!rejected.passed);
    }

    #[test]
    fn empty_generation_never_certifies() {
        let manifest = DynamicsOracleDatasetManifest {
            schema_version: 2,
            dataset_id: "dataset".to_string(),
            generator_id: "generator".to_string(),
            generator_build_id: "generator-build".to_string(),
            generator_executable_sha256: "e".repeat(64),
            engine_id: "mujoco".to_string(),
            model_artifact_sha256: "a".repeat(64),
            morphology: crate::morphology::HumanoidMorphology::Dmc21,
            generalized_coordinate_order: (0..27).map(|index| format!("v{index}")).collect(),
            contact_site_order: vec!["r_foot_site".to_string(), "l_foot_site".to_string()],
            generated_unix_millis: 1,
        };
        let certificate = certify_generation_cases(
            &manifest,
            "candidate-build",
            0,
            Vec::new(),
            IndependentOracleGenerationCriteria {
                minimum_cases: 1,
                minimum_distinct_states: 1,
                maximum_case_elapsed_micros: 1,
                require_all_cases: true,
            },
            false,
        );
        assert!(!certificate.passed);
    }
}
