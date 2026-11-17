// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-gated qualification contracts for Atelier robotic Lightshell.
//!
//! This module is deliberately an admission and benchmark layer. It does not
//! model textile mechanics, make safety decisions, or authorize robot motion.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

const CANDIDATE_IDS: [&str; 6] = ["B0", "B1", "B2", "L1", "L2", "L3"];
const REQUIRED_METRICS: [&str; 4] = [
    "added_mass",
    "rom_delta",
    "joint_energy_delta",
    "peak_temperature_delta",
];
const REQUIRED_GATES: [&str; 9] = [
    "pinch_shear_clearance",
    "required_rom",
    "cable_hose_bend",
    "thermal_cooling",
    "sensor_fov_calibration",
    "retention_detachment_snag",
    "emergency_service_access",
    "actuation_task_nonregression",
    "evidence_integrity",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkManifest {
    pub schema_version: String,
    pub benchmark_id: String,
    pub benchmark_state: String,
    pub claim_scope: String,
    pub platform: Platform,
    pub application_profile: ApplicationProfile,
    pub matched_candidates: Vec<Candidate>,
    pub metric_registry: Vec<Metric>,
    pub hard_gates: Vec<Gate>,
    pub test_runs: Vec<TestRun>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Platform {
    pub platform_id: String,
    pub article_revision: Option<String>,
    pub geometry_state: String,
    pub geometry_digest: Option<String>,
    pub joint_or_zone_refs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationProfile {
    pub operating_domain: Vec<String>,
    pub task_profile: Option<String>,
    pub hazard_profile: Vec<String>,
    pub coverage_boundary: Option<String>,
    pub human_proximity: String,
    pub cleaning_profile: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub candidate_id: String,
    pub architecture: String,
    pub availability: String,
    pub comparability: String,
    pub description: String,
    pub material_lot_refs: Vec<String>,
    pub article_revision: Option<String>,
    pub coverage_boundary_ref: Option<String>,
    pub deviation_notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    pub metric_id: String,
    pub candidate_id: String,
    pub unit: String,
    pub measurement_boundary: String,
    pub comparison_direction: String,
    pub target_value: Option<f64>,
    pub baseline_value: Option<f64>,
    pub observed_value: Option<f64>,
    pub hard_gate: bool,
    pub method_ref: Option<String>,
    pub uncertainty_model_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    pub gate_id: String,
    pub candidate_id: String,
    pub condition: String,
    pub state: String,
    pub acceptance_rule_ref: Option<String>,
    pub reviewer_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestRun {
    pub run_id: String,
    pub candidate_id: String,
    pub stage: String,
    pub status: String,
    pub protocol_revision: Option<String>,
    pub instrument_refs: Vec<String>,
    pub calibration_refs: Vec<String>,
    pub raw_data_digest: Option<String>,
    pub acceptance_rule_ref: Option<String>,
    pub deviation_notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub raw_data_retained: bool,
    pub configuration_bound: bool,
    pub calibration_required: bool,
    pub uncertainty_required: bool,
    pub simulation_not_physical_qualification: bool,
    pub supplier_claims_not_independent_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    /// Only candidates with complete candidate-scoped gates, a matched B0
    /// baseline, a physically passed module/integrated run, and the required
    /// observed metric values are returned here.
    pub rankable_candidates: Vec<String>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn can_rank(&self, candidate_id: &str) -> bool {
        self.rankable_candidates
            .iter()
            .any(|id| id == candidate_id)
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: BenchmarkManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl BenchmarkManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut candidates: HashMap<&str, &Candidate> = HashMap::new();

        if self.schema_version != "0.1.0" {
            push(
                &mut findings,
                "unsupported_schema_version",
                "schema_version",
                "Only schema version 0.1.0 is admitted by this validator.",
            );
        }
        if self.benchmark_id.trim().is_empty() || self.claim_scope.trim().is_empty() {
            push(
                &mut findings,
                "missing_identity_or_scope",
                "benchmark_id/claim_scope",
                "Benchmark identity and claim scope must be non-empty.",
            );
        }
        if !matches!(
            self.benchmark_state.as_str(),
            "proposed" | "inputs_pending" | "frozen" | "running" | "complete" | "invalidated"
        ) {
            push(
                &mut findings,
                "invalid_benchmark_state",
                "benchmark_state",
                "Unknown benchmark lifecycle state.",
            );
        }
        if !matches!(
            self.platform.geometry_state.as_str(),
            "unknown" | "simplified_fixture" | "cad_revision_bound" | "as_built_metrology_bound"
        ) {
            push(
                &mut findings,
                "invalid_geometry_state",
                "platform.geometry_state",
                "Unknown geometry evidence state.",
            );
        }

        for (index, candidate) in self.matched_candidates.iter().enumerate() {
            let path = format!("matched_candidates[{index}]");
            if !CANDIDATE_IDS.contains(&candidate.candidate_id.as_str()) {
                push(
                    &mut findings,
                    "unknown_candidate_id",
                    &path,
                    "Candidate ID must be one of B0, B1, B2, L1, L2 or L3.",
                );
            }
            if candidates.insert(candidate.candidate_id.as_str(), candidate).is_some() {
                push(
                    &mut findings,
                    "duplicate_candidate",
                    &path,
                    "Candidate IDs must be unique.",
                );
            }
            if !matches!(
                candidate.availability.as_str(),
                "unknown" | "planned" | "available" | "unavailable" | "not_applicable"
            ) {
                push(
                    &mut findings,
                    "invalid_candidate_availability",
                    format!("{path}.availability"),
                    "Unknown availability state.",
                );
            }
            if !matches!(
                candidate.comparability.as_str(),
                "unassessed" | "matched" | "partially_matched" | "not_comparable"
            ) {
                push(
                    &mut findings,
                    "invalid_comparability",
                    format!("{path}.comparability"),
                    "Unknown comparability state.",
                );
            }
        }

        for expected in CANDIDATE_IDS {
            if !candidates.contains_key(expected) {
                push(
                    &mut findings,
                    "missing_candidate",
                    "matched_candidates",
                    format!("Required candidate {expected} is absent."),
                );
            }
        }

        let mut metric_keys = HashSet::new();
        for (index, metric) in self.metric_registry.iter().enumerate() {
            let path = format!("metric_registry[{index}]");
            if !candidates.contains_key(metric.candidate_id.as_str()) {
                push(
                    &mut findings,
                    "metric_unknown_candidate",
                    format!("{path}.candidate_id"),
                    "Metric references a candidate that is absent from the manifest.",
                );
            }
            if !metric_keys.insert((metric.candidate_id.as_str(), metric.metric_id.as_str())) {
                push(
                    &mut findings,
                    "duplicate_candidate_metric",
                    &path,
                    "A candidate may have only one row per metric ID in this manifest revision.",
                );
            }
            if metric.unit.trim().is_empty() || metric.measurement_boundary.trim().is_empty() {
                push(
                    &mut findings,
                    "metric_missing_boundary",
                    &path,
                    "Metric unit and measurement boundary must be explicit.",
                );
            }
            if metric.observed_value.is_some_and(|value| !value.is_finite()) {
                push(
                    &mut findings,
                    "non_finite_metric",
                    &path,
                    "Observed metric values must be finite.",
                );
            }
        }

        for candidate_id in CANDIDATE_IDS {
            for metric_id in REQUIRED_METRICS {
                if !metric_keys.contains(&(candidate_id, metric_id)) {
                    push(
                        &mut findings,
                        "missing_required_metric",
                        "metric_registry",
                        format!("Candidate {candidate_id} is missing required metric {metric_id}."),
                    );
                }
            }
        }

        let mut gate_keys = HashSet::new();
        for (index, gate) in self.hard_gates.iter().enumerate() {
            let path = format!("hard_gates[{index}]");
            if !candidates.contains_key(gate.candidate_id.as_str()) {
                push(
                    &mut findings,
                    "gate_unknown_candidate",
                    format!("{path}.candidate_id"),
                    "Gate references a candidate that is absent from the manifest.",
                );
            }
            if !REQUIRED_GATES.contains(&gate.gate_id.as_str()) {
                push(
                    &mut findings,
                    "unknown_gate_id",
                    format!("{path}.gate_id"),
                    "Unknown hard-gate identifier.",
                );
            }
            if !gate_keys.insert((gate.candidate_id.as_str(), gate.gate_id.as_str())) {
                push(
                    &mut findings,
                    "duplicate_candidate_gate",
                    &path,
                    "A candidate may have only one row per hard-gate ID.",
                );
            }
            if !matches!(
                gate.state.as_str(),
                "unknown" | "not_run" | "pass" | "fail" | "not_applicable"
            ) {
                push(
                    &mut findings,
                    "invalid_gate_state",
                    format!("{path}.state"),
                    "Unknown gate state.",
                );
            }
            if gate.state == "pass"
                && (gate.acceptance_rule_ref.as_deref().unwrap_or("").trim().is_empty()
                    || gate.reviewer_ref.as_deref().unwrap_or("").trim().is_empty())
            {
                push(
                    &mut findings,
                    "gate_pass_without_acceptance",
                    &path,
                    "A passed gate requires an acceptance-rule reference and a reviewer reference.",
                );
            }
        }

        for candidate_id in CANDIDATE_IDS {
            for gate_id in REQUIRED_GATES {
                if !gate_keys.contains(&(candidate_id, gate_id)) {
                    push(
                        &mut findings,
                        "missing_candidate_gate",
                        "hard_gates",
                        format!("Candidate {candidate_id} is missing hard gate {gate_id}."),
                    );
                }
            }
        }

        let mut run_ids = HashSet::new();
        for (index, run) in self.test_runs.iter().enumerate() {
            let path = format!("test_runs[{index}]");
            if !run_ids.insert(run.run_id.as_str()) {
                push(
                    &mut findings,
                    "duplicate_test_run_id",
                    &path,
                    "Test-run IDs must be unique.",
                );
            }
            if !candidates.contains_key(run.candidate_id.as_str()) {
                push(
                    &mut findings,
                    "test_run_unknown_candidate",
                    format!("{path}.candidate_id"),
                    "Test run references a candidate that is absent from the manifest.",
                );
            }
            if !matches!(
                run.stage.as_str(),
                "coupon" | "fit_screen" | "module" | "integrated_robot" | "lifecycle"
            ) {
                push(
                    &mut findings,
                    "invalid_test_stage",
                    format!("{path}.stage"),
                    "Unknown test stage.",
                );
            }
            if !matches!(
                run.status.as_str(),
                "planned" | "not_run" | "inconclusive" | "failed" | "passed" | "invalidated"
            ) {
                push(
                    &mut findings,
                    "invalid_test_status",
                    format!("{path}.status"),
                    "Unknown test status.",
                );
            }
            if run.status == "passed" {
                let valid_digest = run
                    .raw_data_digest
                    .as_deref()
                    .is_some_and(is_sha256_digest);
                if run.protocol_revision.as_deref().unwrap_or("").trim().is_empty()
                    || run.instrument_refs.is_empty()
                    || run.calibration_refs.is_empty()
                    || !valid_digest
                    || run.acceptance_rule_ref.as_deref().unwrap_or("").trim().is_empty()
                {
                    push(
                        &mut findings,
                        "passed_test_without_evidence",
                        &path,
                        "A passed test requires a protocol revision, instruments, calibration references, SHA-256 raw-data digest and acceptance-rule reference.",
                    );
                }
                if candidates
                    .get(run.candidate_id.as_str())
                    .is_some_and(|candidate| candidate.availability != "available")
                {
                    push(
                        &mut findings,
                        "passed_test_candidate_not_available",
                        &path,
                        "A passed physical test requires the exact candidate article to be admitted as available.",
                    );
                }
            }
        }

        if !self.evidence_policy.raw_data_retained
            || !self.evidence_policy.configuration_bound
            || !self.evidence_policy.calibration_required
            || !self.evidence_policy.uncertainty_required
            || !self.evidence_policy.simulation_not_physical_qualification
            || !self.evidence_policy.supplier_claims_not_independent_qualification
        {
            push(
                &mut findings,
                "weakened_evidence_policy",
                "evidence_policy",
                "Lightshell benchmark admission requires raw-data retention, configuration binding, calibration and uncertainty, and strict separation of simulation/supplier claims from physical qualification.",
            );
        }

        if matches!(self.benchmark_state.as_str(), "frozen" | "running" | "complete") {
            if !matches!(
                self.platform.geometry_state.as_str(),
                "cad_revision_bound" | "as_built_metrology_bound"
            ) || self.platform.article_revision.as_deref().unwrap_or("").trim().is_empty()
            {
                push(
                    &mut findings,
                    "benchmark_geometry_not_frozen",
                    "platform",
                    "A frozen/running/complete benchmark requires revision-bound CAD or as-built metrology geometry.",
                );
            }
            if self.application_profile.operating_domain.is_empty()
                || self.application_profile.task_profile.as_deref().unwrap_or("").trim().is_empty()
                || self
                    .application_profile
                    .coverage_boundary
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty()
            {
                push(
                    &mut findings,
                    "benchmark_profile_not_frozen",
                    "application_profile",
                    "A frozen/running/complete benchmark requires an operating domain, task profile and coverage boundary.",
                );
            }
            if candidates.get("B0").is_none_or(|baseline| {
                baseline.availability != "available" || baseline.comparability != "matched"
            }) {
                push(
                    &mut findings,
                    "missing_matched_bare_baseline",
                    "matched_candidates",
                    "A frozen/running/complete benchmark requires an available, matched B0 bare-platform baseline.",
                );
            }
            for candidate in self.matched_candidates.iter() {
                if candidate.comparability == "unassessed" {
                    push(
                        &mut findings,
                        "candidate_comparability_unassessed",
                        "matched_candidates",
                        format!("Candidate {} must be marked matched, partially_matched or not_comparable before protocol freeze.", candidate.candidate_id),
                    );
                }
            }
        }

        if self.benchmark_state == "complete" {
            for candidate in self.matched_candidates.iter().filter(|candidate| {
                candidate.availability == "available" && candidate.comparability == "matched"
            }) {
                let has_integrated_pass = self.test_runs.iter().any(|run| {
                    run.candidate_id == candidate.candidate_id
                        && run.stage == "integrated_robot"
                        && run.status == "passed"
                });
                if !has_integrated_pass {
                    push(
                        &mut findings,
                        "complete_without_integrated_test",
                        "test_runs",
                        format!("Complete benchmark is missing a passed integrated-robot run for candidate {}.", candidate.candidate_id),
                    );
                }
                if !candidate_gates_pass(
                    &candidate.candidate_id,
                    self.hard_gates.as_slice(),
                    &gate_keys,
                ) {
                    push(
                        &mut findings,
                        "complete_with_unresolved_gates",
                        "hard_gates",
                        format!("Complete benchmark has unresolved or failed hard gates for candidate {}.", candidate.candidate_id),
                    );
                }
                if !has_required_observations(
                    &candidate.candidate_id,
                    self.metric_registry.as_slice(),
                ) {
                    push(
                        &mut findings,
                        "complete_without_required_measurements",
                        "metric_registry",
                        format!("Complete benchmark is missing observed core metrics with method and uncertainty for candidate {}.", candidate.candidate_id),
                    );
                }
            }
        }

        // Any malformed or contradictory manifest disables ranking globally.
        // Findings remain visible; the validator never "best-effort" ranks a
        // partially understood evidence package.
        let rankable_candidates = if findings.is_empty() {
            self.compute_rankable_candidates(&candidates, &gate_keys)
        } else {
            Vec::new()
        };
        ValidationReport {
            findings,
            rankable_candidates,
        }
    }

    fn compute_rankable_candidates(
        &self,
        candidates: &HashMap<&str, &Candidate>,
        gate_keys: &HashSet<(&str, &str)>,
    ) -> Vec<String> {
        let baseline = match candidates.get("B0") {
            Some(candidate)
                if candidate.availability == "available" && candidate.comparability == "matched" =>
            {
                *candidate
            }
            _ => return Vec::new(),
        };

        if !candidate_gates_pass("B0", self.hard_gates.as_slice(), gate_keys)
            || !has_passed_physical_stage("B0", self.test_runs.as_slice())
            || has_adverse_test_result("B0", self.test_runs.as_slice())
            || !has_required_observations("B0", self.metric_registry.as_slice())
        {
            return Vec::new();
        }

        self.matched_candidates
            .iter()
            .filter(|candidate| {
                candidate.candidate_id != "B0"
                    && candidate.availability == "available"
                    && candidate.comparability == "matched"
                    && candidate.coverage_boundary_ref.is_some()
                    && baseline.coverage_boundary_ref.is_some()
                    && candidate.coverage_boundary_ref == baseline.coverage_boundary_ref
                    && candidate_gates_pass(
                        &candidate.candidate_id,
                        self.hard_gates.as_slice(),
                        gate_keys,
                    )
                    && has_passed_physical_stage(
                        &candidate.candidate_id,
                        self.test_runs.as_slice(),
                    )
                    && !has_adverse_test_result(
                        &candidate.candidate_id,
                        self.test_runs.as_slice(),
                    )
                    && has_required_observations(
                        &candidate.candidate_id,
                        self.metric_registry.as_slice(),
                    )
            })
            .map(|candidate| candidate.candidate_id.clone())
            .collect()
    }
}

fn candidate_gates_pass(
    candidate_id: &str,
    gates: &[Gate],
    gate_keys: &HashSet<(&str, &str)>,
) -> bool {
    if !REQUIRED_GATES
        .iter()
        .all(|gate_id| gate_keys.contains(&(candidate_id, *gate_id)))
    {
        return false;
    }

    let candidate_gates: Vec<&Gate> = gates
        .iter()
        .filter(|gate| gate.candidate_id == candidate_id)
        .collect();

    !candidate_gates.is_empty()
        && candidate_gates.iter().all(|gate| {
            matches!(gate.state.as_str(), "pass" | "not_applicable")
                && (gate.state != "pass"
                    || (gate.acceptance_rule_ref.is_some() && gate.reviewer_ref.is_some()))
        })
        && candidate_gates.iter().any(|gate| gate.state == "pass")
}

fn has_passed_physical_stage(candidate_id: &str, runs: &[TestRun]) -> bool {
    runs.iter().any(|run| {
        run.candidate_id == candidate_id
            && matches!(run.stage.as_str(), "module" | "integrated_robot")
            && run.status == "passed"
            && run.protocol_revision.is_some()
            && !run.instrument_refs.is_empty()
            && !run.calibration_refs.is_empty()
            && run.raw_data_digest.as_deref().is_some_and(is_sha256_digest)
            && run.acceptance_rule_ref.is_some()
    })
}

fn has_adverse_test_result(candidate_id: &str, runs: &[TestRun]) -> bool {
    runs.iter().any(|run| {
        run.candidate_id == candidate_id
            && matches!(run.status.as_str(), "failed" | "inconclusive")
    })
}

fn has_required_observations(candidate_id: &str, metrics: &[Metric]) -> bool {
    REQUIRED_METRICS.iter().all(|metric_id| {
        metrics.iter().any(|metric| {
            metric.candidate_id == candidate_id
                && metric.metric_id == *metric_id
                && metric.observed_value.is_some()
                && metric.method_ref.is_some()
                && metric.uncertainty_model_ref.is_some()
        })
    })
}

fn is_sha256_digest(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn push(
    findings: &mut Vec<Finding>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(Finding {
        code,
        path: path.into(),
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: &str = include_str!(
        "../../../../docs/research/fixtures/atelier_robot_lightshell_v0_1/lightshell_benchmark.example.json"
    );

    fn manifest() -> BenchmarkManifest {
        serde_json::from_str(SEED).expect("seed fixture must deserialize")
    }

    fn qualify_candidate(manifest: &mut BenchmarkManifest, candidate_id: &str) {
        let candidate = manifest
            .matched_candidates
            .iter_mut()
            .find(|candidate| candidate.candidate_id == candidate_id)
            .expect("candidate exists");
        candidate.availability = "available".into();
        candidate.comparability = "matched".into();
        candidate.coverage_boundary_ref = Some("coverage:forearm-elbow-v1".into());

        for gate in manifest
            .hard_gates
            .iter_mut()
            .filter(|gate| gate.candidate_id == candidate_id)
        {
            gate.state = "pass".into();
            gate.acceptance_rule_ref = Some(format!("acceptance:{}:{}", candidate_id, gate.gate_id));
            gate.reviewer_ref = Some("reviewer:independent-engineering".into());
        }

        for metric in manifest
            .metric_registry
            .iter_mut()
            .filter(|metric| metric.candidate_id == candidate_id)
        {
            if REQUIRED_METRICS.contains(&metric.metric_id.as_str()) {
                metric.observed_value = Some(1.0);
                metric.method_ref = Some(format!("method:{}", metric.metric_id));
                metric.uncertainty_model_ref = Some(format!("uncertainty:{}", metric.metric_id));
            }
        }

        manifest.test_runs.push(TestRun {
            run_id: format!("run-{candidate_id}-module-001"),
            candidate_id: candidate_id.into(),
            stage: "module".into(),
            status: "passed".into(),
            protocol_revision: Some("lightshell-test-v0.1".into()),
            instrument_refs: vec!["instrument:reference-load-cell".into()],
            calibration_refs: vec!["calibration:current".into()],
            raw_data_digest: Some(format!("sha256:{}", "a".repeat(64))),
            acceptance_rule_ref: Some(format!("acceptance:{candidate_id}:module")),
            deviation_notes: None,
        });
    }

    #[test]
    fn seed_fixture_is_valid_but_not_rankable() {
        let report = manifest().validate();
        assert!(report.is_valid(), "{:#?}", report.findings);
        assert!(report.rankable_candidates.is_empty());
    }

    #[test]
    fn per_candidate_gates_prevent_cross_candidate_masking() {
        let mut manifest = manifest();
        qualify_candidate(&mut manifest, "B0");
        qualify_candidate(&mut manifest, "L1");

        let l1_gate = manifest
            .hard_gates
            .iter_mut()
            .find(|gate| gate.candidate_id == "L1" && gate.gate_id == "thermal_cooling")
            .unwrap();
        l1_gate.state = "fail".into();

        let report = manifest.validate();
        assert!(!report.can_rank("L1"));
        assert!(!report.is_valid() || report.rankable_candidates.is_empty());
    }

    #[test]
    fn ranking_requires_exact_matched_control_and_candidate_coverage() {
        let mut manifest = manifest();
        manifest.benchmark_state = "frozen".into();
        manifest.platform.article_revision = Some("jointlab-article-r1".into());
        manifest.platform.geometry_state = "cad_revision_bound".into();
        manifest.application_profile.operating_domain = vec!["bench-indoor".into()];
        manifest.application_profile.task_profile = Some("bounded-joint-motion".into());
        manifest.application_profile.coverage_boundary = Some("forearm-elbow-v1".into());

        for candidate in manifest.matched_candidates.iter_mut() {
            candidate.availability = "unavailable".into();
            candidate.comparability = "not_comparable".into();
        }
        qualify_candidate(&mut manifest, "B0");
        qualify_candidate(&mut manifest, "L1");

        let report = manifest.validate();
        assert!(report.is_valid(), "{:#?}", report.findings);
        assert_eq!(report.rankable_candidates, vec!["L1".to_string()]);
    }

    #[test]
    fn an_adverse_run_prevents_candidate_ranking_even_after_a_passed_run() {
        let mut manifest = manifest();
        qualify_candidate(&mut manifest, "B0");
        qualify_candidate(&mut manifest, "L1");
        manifest.test_runs.push(TestRun {
            run_id: "run-L1-failed-regression".into(),
            candidate_id: "L1".into(),
            stage: "module".into(),
            status: "failed".into(),
            protocol_revision: Some("lightshell-test-v0.1".into()),
            instrument_refs: vec!["instrument:reference-load-cell".into()],
            calibration_refs: vec!["calibration:current".into()],
            raw_data_digest: Some(format!("sha256:{}", "c".repeat(64))),
            acceptance_rule_ref: Some("acceptance:L1:failure".into()),
            deviation_notes: Some("Recorded failure remains part of the candidate evidence.".into()),
        });
        let report = manifest.validate();
        assert!(report.is_valid(), "{:#?}", report.findings);
        assert!(!report.can_rank("L1"));
    }

    #[test]
    fn passed_test_without_calibration_is_rejected() {
        let mut manifest = manifest();
        let mut run = TestRun {
            run_id: "bad-run".into(),
            candidate_id: "B0".into(),
            stage: "module".into(),
            status: "passed".into(),
            protocol_revision: Some("p1".into()),
            instrument_refs: vec!["instrument:1".into()],
            calibration_refs: Vec::new(),
            raw_data_digest: Some(format!("sha256:{}", "b".repeat(64))),
            acceptance_rule_ref: Some("acceptance:1".into()),
            deviation_notes: None,
        };
        // Keep the exact candidate status at planned; this specifically probes
        // the passed-test evidence contract rather than ranking eligibility.
        run.status = "passed".into();
        manifest.test_runs.push(run);
        let report = manifest.validate();
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "passed_test_without_evidence"));
    }

    #[test]
    fn frozen_benchmark_requires_cad_and_task_profile() {
        let mut manifest = manifest();
        manifest.benchmark_state = "frozen".into();
        let report = manifest.validate();
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "benchmark_geometry_not_frozen"));
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "benchmark_profile_not_frozen"));
    }
}
