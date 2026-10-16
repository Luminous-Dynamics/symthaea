//! Emit a deterministic synthetic-only Living City policy benchmark report.
//!
//! This command is a research harness smoke test. It does not run a physical
//! city model, assign real-world probabilities to the cases, or authorize an
//! infrastructure action. A valid report may still contain failed policies;
//! the report describes those failures rather than hiding them.

use serde::Serialize;
use std::error::Error;

use symthaea_scenarios::living_city_evidence::sha256_hex;
use symthaea_scenarios::living_city_policy_benchmark::{
    assess_policy_robustness, assess_policy_suite, built_in_synthetic_smoke_scenarios,
    evaluate_suite, BenchmarkConfig, PolicyKind, PolicyRobustnessAssessment,
    PolicyRun, PolicySuiteAssessment, PolicySuiteState, STANDARD_BASELINE_POLICIES,
    SYNTHETIC_SMOKE_CONFIG, LIVING_CITY_POLICY_BENCHMARK_VERSION,
};

#[derive(Debug, Serialize)]
struct PolicyReportPayload {
    report_schema_version: u32,
    suite_version: u32,
    suite_id: String,
    evidence_boundary: String,
    configuration_sha256: String,
    scenario_manifest_sha256: String,
    required_case_ids: Vec<String>,
    scenario_count: usize,
    run_count: usize,
    policy_assessments: Vec<PolicySuiteAssessment>,
    floor_first_vs_baselines: Vec<PolicyRobustnessAssessment>,
}

#[derive(Debug, Serialize)]
struct PolicyReportEnvelope {
    payload: PolicyReportPayload,
    /// Digest of compact serialized payload bytes, excluding this envelope.
    payload_sha256: String,
}

fn suite_config() -> BenchmarkConfig {
    SYNTHETIC_SMOKE_CONFIG
}

fn build_report() -> Result<PolicyReportEnvelope, Box<dyn Error>> {
    let config = suite_config();
    config.validate()?;

    let scenarios = built_in_synthetic_smoke_scenarios();
    let required_case_ids: Vec<String> = scenarios
        .iter()
        .map(|scenario| scenario.scenario_id.clone())
        .collect();

    let mut all_runs: Vec<(PolicyKind, Vec<PolicyRun>)> =
        Vec::with_capacity(STANDARD_BASELINE_POLICIES.len());
    let mut policy_assessments = Vec::with_capacity(STANDARD_BASELINE_POLICIES.len());

    for policy in STANDARD_BASELINE_POLICIES {
        let runs = evaluate_suite(&scenarios, &[policy], config)?;
        let assessment = assess_policy_suite(
            &required_case_ids,
            &scenarios,
            policy,
            &runs,
            config,
        )?;
        if assessment.state == PolicySuiteState::Blocked {
            return Err(format!(
                "synthetic policy suite blocked for {policy:?}: {:?}",
                assessment.blockers
            )
            .into());
        }
        policy_assessments.push(assessment);
        all_runs.push((policy, runs));
    }

    let candidate_policy = PolicyKind::FloorFirst;
    let candidate_runs = all_runs
        .iter()
        .find(|(policy, _)| *policy == candidate_policy)
        .map(|(_, runs)| runs)
        .ok_or("floor-first candidate runs missing")?;

    let mut floor_first_vs_baselines = Vec::new();
    for baseline_policy in STANDARD_BASELINE_POLICIES {
        if baseline_policy == candidate_policy {
            continue;
        }
        let baseline_runs = all_runs
            .iter()
            .find(|(policy, _)| *policy == baseline_policy)
            .map(|(_, runs)| runs)
            .ok_or("declared baseline runs missing")?;
        floor_first_vs_baselines.push(assess_policy_robustness(
            &required_case_ids,
            &scenarios,
            candidate_policy,
            candidate_runs,
            baseline_policy,
            baseline_runs,
            config,
        )?);
    }

    if floor_first_vs_baselines
        .iter()
        .any(|comparison| !comparison.comparison_ready)
    {
        return Err("paired policy comparison is not ready".into());
    }

    let config_bytes = config.canonical_bytes()?;
    let mut scenario_digests: Vec<String> = scenarios
        .iter()
        .map(|scenario| scenario.sha256())
        .collect::<Result<_, _>>()?;
    scenario_digests.sort();
    let scenario_manifest_bytes = serde_json::to_vec(&scenario_digests)?;
    let scenario_manifest_sha256 = sha256_hex(&scenario_manifest_bytes);

    let payload = PolicyReportPayload {
        report_schema_version: 1,
        suite_version: LIVING_CITY_POLICY_BENCHMARK_VERSION,
        suite_id: "living-city-synthetic-smoke-v1".into(),
        evidence_boundary: "synthetic_single_service_not_physical_model".into(),
        configuration_sha256: sha256_hex(&config_bytes),
        scenario_manifest_sha256,
        required_case_ids,
        scenario_count: scenarios.len(),
        run_count: all_runs.iter().map(|(_, runs)| runs.len()).sum(),
        policy_assessments,
        floor_first_vs_baselines,
    };
    let payload_bytes = serde_json::to_vec(&payload)?;
    let payload_sha256 = sha256_hex(&payload_bytes);

    Ok(PolicyReportEnvelope {
        payload,
        payload_sha256,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "living_city_policy_report\n\nRuns the deterministic synthetic-only Living City baseline suite and emits a SHA-256-bound JSON report. This is not a physical city model."
        );
        return Ok(());
    }
    if !args.is_empty() {
        return Err(format!("unsupported argument: {} (use --help)", args[0]).into());
    }

    let report = build_report()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_covers_all_cases_and_baselines() {
        let report = build_report().unwrap();
        assert_eq!(report.payload.scenario_count, 4);
        assert_eq!(report.payload.run_count, 16);
        assert_eq!(report.payload.policy_assessments.len(), 4);
        assert_eq!(report.payload.floor_first_vs_baselines.len(), 3);
        assert!(report
            .payload
            .policy_assessments
            .iter()
            .all(|assessment| assessment.state != PolicySuiteState::Blocked));
        assert!(report
            .payload
            .floor_first_vs_baselines
            .iter()
            .all(|comparison| comparison.comparison_ready));
    }

    #[test]
    fn payload_digest_is_reproducible() {
        let first = build_report().unwrap();
        let second = build_report().unwrap();
        let first_bytes = serde_json::to_vec(&first.payload).unwrap();
        let second_bytes = serde_json::to_vec(&second.payload).unwrap();
        assert_eq!(first.payload_sha256, sha256_hex(&first_bytes));
        assert_eq!(first.payload_sha256, second.payload_sha256);
        assert_eq!(first_bytes, second_bytes);
    }

    #[test]
    fn report_records_failures_without_calling_the_suite_blocked() {
        let report = build_report().unwrap();
        let no_intervention = report
            .payload
            .policy_assessments
            .iter()
            .find(|assessment| assessment.policy == PolicyKind::NoIntervention)
            .unwrap();
        assert_eq!(no_intervention.state, PolicySuiteState::Failed);
        assert!(report
            .payload
            .policy_assessments
            .iter()
            .all(|assessment| assessment.state != PolicySuiteState::Blocked));
    }
}
