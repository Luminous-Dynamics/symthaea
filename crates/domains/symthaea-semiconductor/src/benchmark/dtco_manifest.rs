// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-fidelity DTCO evidence-lineage validation for SEMI-BENCH-001.
//!
//! This benchmark-specific contract binds each candidate evaluation to exact
//! input/output artifact digests and parent runs. It does not create a generic
//! provenance system; production identity and custody remain with Mycelix.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.dtco-lineage-manifest.v1";
pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.dtco-lineage-validation.v1";

const STAGES: [&str; 10] = [
    "analytical",
    "compact_model",
    "tcad",
    "spice",
    "synthesis",
    "placement",
    "global_route",
    "detailed_route",
    "package_simulation",
    "physical_measurement",
];

const EVIDENCE_KINDS: [&str; 4] = [
    "analytical_estimate",
    "model_prediction",
    "tool_simulation",
    "physical_measurement",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DtcoRunV1 {
    pub run_id: String,
    pub stage: String,
    pub evidence_kind: String,
    pub parent_run_ids: Vec<String>,
    pub input_artifact_sha256: Vec<String>,
    pub output_artifact_sha256: Vec<String>,
    pub tool_name: String,
    pub tool_version: String,
    pub configuration_sha256: String,
    #[serde(default)]
    pub model_sha256: Option<String>,
    #[serde(default)]
    pub prediction_uncertainty_artifact_sha256: Option<String>,
    #[serde(default)]
    pub instrument_ref: Option<String>,
    #[serde(default)]
    pub calibration_ref: Option<String>,
    #[serde(default)]
    pub uncertainty_artifact_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DtcoLineageManifestV1 {
    pub schema: String,
    pub design_id: String,
    pub candidate_id: String,
    pub design_input_sha256: String,
    pub pdk_or_process_profile_id: String,
    pub flow_profile_id: String,
    pub constraints_sha256: String,
    pub runs: Vec<DtcoRunV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct DtcoLineageError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DtcoLineageGuaranteesV1 {
    pub unique_run_identity: bool,
    pub all_parent_runs_exist: bool,
    pub lineage_is_acyclic: bool,
    pub each_parent_output_is_bound_to_child_input: bool,
    pub physical_measurements_have_instrument_calibration_uncertainty: bool,
    pub modeled_evidence_is_not_mislabeled_as_physical_measurement: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DtcoLineageClaimLimitsV1 {
    pub validates_evidence_lineage_only: bool,
    pub does_not_validate_physics_or_model_accuracy: bool,
    pub does_not_establish_pdk_or_process_qualification: bool,
    pub does_not_establish_fabricated_silicon_performance_or_yield: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DtcoLineageReportV1 {
    pub schema: String,
    pub design_id: String,
    pub candidate_id: String,
    pub run_count: usize,
    pub unique_artifact_count: usize,
    pub parent_edge_count: usize,
    pub runs_by_stage: BTreeMap<String, usize>,
    pub runs_by_evidence_kind: BTreeMap<String, usize>,
    pub physical_measurement_count: usize,
    pub guarantees: DtcoLineageGuaranteesV1,
    pub claim_limits: DtcoLineageClaimLimitsV1,
}

fn require_text(value: &str, field: &str) -> Result<(), DtcoLineageError> {
    if value.trim().is_empty() {
        return Err(DtcoLineageError(format!(
            "{field}: expected non-empty string"
        )));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &str) -> Result<(), DtcoLineageError> {
    require_text(value, field)?;
    if !is_sha256(value) {
        return Err(DtcoLineageError(format!(
            "{field}: expected lowercase SHA-256 hex"
        )));
    }
    Ok(())
}

fn validate_digest_list(values: &[String], field: &str) -> Result<(), DtcoLineageError> {
    if values.is_empty() {
        return Err(DtcoLineageError(format!(
            "{field}: expected non-empty digest array"
        )));
    }
    let mut seen = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        require_sha256(value, &format!("{field}[{index}]"))?;
        if !seen.insert(value.as_str()) {
            return Err(DtcoLineageError(format!(
                "{field}[{index}]: duplicate artifact digest"
            )));
        }
    }
    Ok(())
}

/// Validates a benchmark-specific multi-fidelity run graph.
///
/// Parent/child relationships are explicit rather than inferred from a single
/// scalar fidelity ordering. A child must name at least one exact output
/// artifact digest from every parent in its input artifact set.
pub fn validate_dtco_lineage_v1(
    manifest: &DtcoLineageManifestV1,
) -> Result<DtcoLineageReportV1, DtcoLineageError> {
    if manifest.schema != INPUT_SCHEMA_V1 {
        return Err(DtcoLineageError(
            "schema: unsupported or missing DTCO lineage manifest".to_string(),
        ));
    }
    require_text(&manifest.design_id, "design_id")?;
    require_text(&manifest.candidate_id, "candidate_id")?;
    require_sha256(&manifest.design_input_sha256, "design_input_sha256")?;
    require_text(
        &manifest.pdk_or_process_profile_id,
        "pdk_or_process_profile_id",
    )?;
    require_text(&manifest.flow_profile_id, "flow_profile_id")?;
    require_sha256(&manifest.constraints_sha256, "constraints_sha256")?;
    if manifest.runs.is_empty() {
        return Err(DtcoLineageError(
            "runs: expected non-empty array".to_string(),
        ));
    }

    let mut run_by_id = BTreeMap::new();
    let mut artifacts = BTreeSet::new();
    let mut runs_by_stage: BTreeMap<String, usize> = BTreeMap::new();
    let mut runs_by_evidence_kind: BTreeMap<String, usize> = BTreeMap::new();
    let mut physical_measurement_count = 0usize;
    let mut parent_edge_count = 0usize;

    for (index, run) in manifest.runs.iter().enumerate() {
        require_text(&run.run_id, &format!("runs[{index}].run_id"))?;
        if run_by_id.insert(run.run_id.as_str(), run).is_some() {
            return Err(DtcoLineageError(format!(
                "runs[{index}].run_id: duplicate run identity"
            )));
        }
        if !STAGES.contains(&run.stage.as_str()) {
            return Err(DtcoLineageError(format!(
                "runs[{index}].stage: unsupported stage"
            )));
        }
        if !EVIDENCE_KINDS.contains(&run.evidence_kind.as_str()) {
            return Err(DtcoLineageError(format!(
                "runs[{index}].evidence_kind: unsupported evidence kind"
            )));
        }
        let physical_stage = run.stage == "physical_measurement";
        let physical_kind = run.evidence_kind == "physical_measurement";
        if physical_stage != physical_kind {
            return Err(DtcoLineageError(format!(
                "runs[{index}]: stage/evidence_kind mismatch; only physical_measurement stage may assert physical_measurement evidence"
            )));
        }
        if run.evidence_kind == "model_prediction" && run.model_sha256.is_none() {
            return Err(DtcoLineageError(format!(
                "runs[{index}].model_sha256: required for model_prediction"
            )));
        }
        if let Some(model_sha256) = run.model_sha256.as_deref() {
            require_sha256(model_sha256, &format!("runs[{index}].model_sha256"))?;
            artifacts.insert(model_sha256.to_string());
        }
        if run.evidence_kind == "model_prediction" {
            let prediction_uncertainty = run
                .prediction_uncertainty_artifact_sha256
                .as_deref()
                .unwrap_or_default();
            require_sha256(
                prediction_uncertainty,
                &format!("runs[{index}].prediction_uncertainty_artifact_sha256"),
            )?;
            artifacts.insert(prediction_uncertainty.to_string());
        } else if run.prediction_uncertainty_artifact_sha256.is_some() {
            return Err(DtcoLineageError(format!(
                "runs[{index}].prediction_uncertainty_artifact_sha256: only permitted for model_prediction evidence"
            )));
        }

        require_text(&run.tool_name, &format!("runs[{index}].tool_name"))?;
        require_text(&run.tool_version, &format!("runs[{index}].tool_version"))?;
        require_sha256(
            &run.configuration_sha256,
            &format!("runs[{index}].configuration_sha256"),
        )?;
        validate_digest_list(
            &run.input_artifact_sha256,
            &format!("runs[{index}].input_artifact_sha256"),
        )?;
        validate_digest_list(
            &run.output_artifact_sha256,
            &format!("runs[{index}].output_artifact_sha256"),
        )?;
        for digest in run
            .input_artifact_sha256
            .iter()
            .chain(run.output_artifact_sha256.iter())
        {
            artifacts.insert(digest.clone());
        }

        let mut parent_ids = BTreeSet::new();
        for parent_id in &run.parent_run_ids {
            require_text(parent_id, &format!("runs[{index}].parent_run_ids"))?;
            if parent_id == &run.run_id {
                return Err(DtcoLineageError(format!(
                    "runs[{index}]: run cannot parent itself"
                )));
            }
            if !parent_ids.insert(parent_id.as_str()) {
                return Err(DtcoLineageError(format!(
                    "runs[{index}]: duplicate parent run identity"
                )));
            }
        }
        parent_edge_count += run.parent_run_ids.len();

        if physical_stage {
            require_text(
                run.instrument_ref.as_deref().unwrap_or_default(),
                &format!("runs[{index}].instrument_ref"),
            )?;
            require_text(
                run.calibration_ref.as_deref().unwrap_or_default(),
                &format!("runs[{index}].calibration_ref"),
            )?;
            require_sha256(
                run.uncertainty_artifact_sha256
                    .as_deref()
                    .unwrap_or_default(),
                &format!("runs[{index}].uncertainty_artifact_sha256"),
            )?;
            artifacts.insert(
                run.uncertainty_artifact_sha256
                    .as_ref()
                    .expect("validated uncertainty digest")
                    .clone(),
            );
            physical_measurement_count += 1;
        } else if run.instrument_ref.is_some()
            || run.calibration_ref.is_some()
            || run.uncertainty_artifact_sha256.is_some()
        {
            return Err(DtcoLineageError(format!(
                "runs[{index}]: physical instrument/calibration/uncertainty fields are only permitted on physical_measurement evidence"
            )));
        }

        *runs_by_stage.entry(run.stage.clone()).or_default() += 1;
        *runs_by_evidence_kind
            .entry(run.evidence_kind.clone())
            .or_default() += 1;
    }

    // Resolve parent references and require every edge to be backed by an
    // exact parent output digest present in the child's input set.
    let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut indegree: BTreeMap<&str, usize> = run_by_id.keys().map(|id| (*id, 0)).collect();
    for run in &manifest.runs {
        for parent_id in &run.parent_run_ids {
            let parent = run_by_id.get(parent_id.as_str()).ok_or_else(|| {
                DtcoLineageError(format!(
                    "run {:?}: missing parent run {:?}",
                    run.run_id, parent_id
                ))
            })?;
            let parent_stage = STAGES
                .iter()
                .position(|stage| *stage == parent.stage)
                .expect("parent stage was validated");
            let child_stage = STAGES
                .iter()
                .position(|stage| *stage == run.stage)
                .expect("child stage was validated");
            if parent_stage > child_stage {
                return Err(DtcoLineageError(format!(
                    "run {:?}: parent fidelity stage cannot be later than child stage",
                    run.run_id
                )));
            }
            let child_inputs: BTreeSet<&str> = run
                .input_artifact_sha256
                .iter()
                .map(String::as_str)
                .collect();
            if !parent
                .output_artifact_sha256
                .iter()
                .any(|digest| child_inputs.contains(digest.as_str()))
            {
                return Err(DtcoLineageError(format!(
                    "run {:?}: parent {:?} output digest is not bound to child inputs",
                    run.run_id, parent_id
                )));
            }
            children
                .entry(parent.run_id.as_str())
                .or_default()
                .push(run.run_id.as_str());
            *indegree
                .get_mut(run.run_id.as_str())
                .expect("run IDs were indexed") += 1;
        }
    }

    // Kahn topological traversal rejects cycles without relying on recursion.
    let mut ready: BTreeSet<&str> = indegree
        .iter()
        .filter_map(|(run_id, degree)| (*degree == 0).then_some(*run_id))
        .collect();
    let mut visited = 0usize;
    while let Some(run_id) = ready.pop_first() {
        visited += 1;
        if let Some(descendants) = children.get(run_id) {
            for child_id in descendants {
                let degree = indegree.get_mut(child_id).expect("child ID was indexed");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(child_id);
                }
            }
        }
    }
    if visited != run_by_id.len() {
        return Err(DtcoLineageError(
            "runs: parent relationships contain a cycle".to_string(),
        ));
    }

    // Bind declared top-level candidate identity into actual evaluation inputs.
    // Every root run begins from the immutable design identity, and declared
    // constraints must be consumed by at least one run in the candidate graph.
    for (run_id, run) in &run_by_id {
        if run.parent_run_ids.is_empty()
            && !run
                .input_artifact_sha256
                .iter()
                .any(|digest| digest == &manifest.design_input_sha256)
        {
            return Err(DtcoLineageError(format!(
                "root run {:?}: design_input_sha256 is not bound to root inputs",
                run_id
            )));
        }
    }
    if !manifest.runs.iter().any(|run| {
        run.input_artifact_sha256
            .iter()
            .any(|digest| digest == &manifest.constraints_sha256)
    }) {
        return Err(DtcoLineageError(
            "constraints_sha256: not bound to any run input artifact".to_string(),
        ));
    }

    Ok(DtcoLineageReportV1 {
        schema: OUTPUT_SCHEMA_V1.to_string(),
        design_id: manifest.design_id.clone(),
        candidate_id: manifest.candidate_id.clone(),
        run_count: manifest.runs.len(),
        unique_artifact_count: artifacts.len(),
        parent_edge_count,
        runs_by_stage,
        runs_by_evidence_kind,
        physical_measurement_count,
        guarantees: DtcoLineageGuaranteesV1 {
            unique_run_identity: true,
            all_parent_runs_exist: true,
            lineage_is_acyclic: true,
            each_parent_output_is_bound_to_child_input: true,
            physical_measurements_have_instrument_calibration_uncertainty: true,
            modeled_evidence_is_not_mislabeled_as_physical_measurement: true,
        },
        claim_limits: DtcoLineageClaimLimitsV1 {
            validates_evidence_lineage_only: true,
            does_not_validate_physics_or_model_accuracy: true,
            does_not_establish_pdk_or_process_qualification: true,
            does_not_establish_fabricated_silicon_performance_or_yield: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const INPUT: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/dtco-lineage-known-answer-v1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/dtco-lineage-known-answer-v1.expected.json"
    );

    fn known_answer() -> DtcoLineageManifestV1 {
        serde_json::from_str(INPUT).expect("known-answer DTCO manifest must deserialize")
    }

    #[test]
    fn rust_lineage_report_matches_frozen_known_answer() {
        let report = validate_dtco_lineage_v1(&known_answer()).expect("manifest validates");
        let actual = serde_json::to_value(report).expect("report serializes");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected report parses");
        assert_eq!(actual, expected);
    }

    #[test]
    fn missing_parent_is_rejected() {
        let mut manifest = known_answer();
        manifest.runs[1].parent_run_ids = vec!["absent-parent".to_string()];
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("missing parent run")
        );
    }

    #[test]
    fn parent_output_must_be_bound_to_child_input() {
        let mut manifest = known_answer();
        manifest.runs[1].input_artifact_sha256[0] =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string();
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("not bound to child inputs")
        );
    }

    #[test]
    fn cyclic_parent_graph_is_rejected() {
        let mut manifest = known_answer();
        manifest.runs[0].parent_run_ids = vec!["run-spice".to_string()];
        manifest.runs[0].input_artifact_sha256 = manifest.runs[1].output_artifact_sha256.clone();
        manifest.runs[1].parent_run_ids = vec!["run-analytical".to_string()];
        manifest.runs[1].input_artifact_sha256 = manifest.runs[0].output_artifact_sha256.clone();
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("cycle")
        );
    }

    #[test]
    fn physical_measurement_requires_instrument_calibration_and_uncertainty() {
        let mut manifest = known_answer();
        manifest.runs[2].calibration_ref = None;
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("calibration_ref")
        );
    }

    #[test]
    fn simulated_run_cannot_claim_physical_measurement_evidence() {
        let mut manifest = known_answer();
        manifest.runs[1].evidence_kind = "physical_measurement".to_string();
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("stage/evidence_kind mismatch")
        );
    }

    #[test]
    fn model_prediction_requires_exact_model_digest() {
        let mut manifest = known_answer();
        manifest.runs[0].evidence_kind = "model_prediction".to_string();
        manifest.runs[0].model_sha256 = None;
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("model_sha256")
        );
    }

    #[test]
    fn model_prediction_requires_uncertainty_artifact() {
        let mut manifest = known_answer();
        manifest.runs[0].evidence_kind = "model_prediction".to_string();
        manifest.runs[0].model_sha256 = Some(
            "5555555555555555555555555555555555555555555555555555555555555555".to_string(),
        );
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("prediction_uncertainty_artifact_sha256")
        );
    }

    #[test]
    fn later_fidelity_stage_cannot_be_parent_of_earlier_stage() {
        let mut manifest = known_answer();
        manifest.runs[0].parent_run_ids = vec!["run-measurement".to_string()];
        manifest.runs[0].input_artifact_sha256 =
            manifest.runs[2].output_artifact_sha256.clone();
        manifest.runs[2].parent_run_ids.clear();
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("parent fidelity stage cannot be later")
        );
    }

    #[test]
    fn root_run_must_bind_design_and_constraints_must_be_consumed() {
        let mut manifest = known_answer();
        manifest.runs[0].input_artifact_sha256 = vec![
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
        ];
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("design_input_sha256 is not bound")
        );

        let mut manifest = known_answer();
        manifest.runs[0].input_artifact_sha256 =
            vec![manifest.design_input_sha256.clone()];
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("constraints_sha256: not bound")
        );
    }

    #[test]
    fn invalid_artifact_digest_is_rejected() {
        let mut manifest = known_answer();
        manifest.runs[0].output_artifact_sha256[0] = "not-a-digest".to_string();
        assert!(
            validate_dtco_lineage_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("lowercase SHA-256")
        );
    }
}
