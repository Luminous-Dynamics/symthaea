//! Synthetic negative controls for candidate-superiority decision rules.
//!
//! The oracle's known efficiency delta is deliberately separated from the
//! observed record delta. This exposes how unmodeled bias can create false
//! superiority claims even when a lower-bound rule is implemented correctly.
//! These generated records are software stress tests, not physical evidence.

use crate::{
    compare_candidates, BenchmarkRun, ComparisonDecision, EfficiencyComparison, ValidationError,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

const FIXTURE: &str =
    include_str!("../../../../docs/research/examples/energy-conversion-benchmark.synthetic.json");
const PROTOCOL_VERSION: u32 = 1;
const BASELINE_ID: &str = "registered-baseline";
const BASELINE_INPUT_J: f64 = 112.0;
const BASELINE_OUTPUT_J: f64 = 35.7;
const BASELINE_EFFICIENCY: f64 = BASELINE_OUTPUT_J / BASELINE_INPUT_J;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateDecisionScenario {
    pub scenario_id: String,
    /// Known oracle efficiency difference (candidate minus baseline).
    pub oracle_true_efficiency_difference: f64,
    /// Difference encoded in the candidate's observed record. May include
    /// deliberate synthetic bias and therefore differ from the oracle truth.
    pub observed_efficiency_difference: f64,
    /// Multiplier applied to the candidate's primitive standard uncertainties.
    pub candidate_uncertainty_scale: f64,
    /// Correlation declaration for the candidate, including "unknown".
    pub candidate_correlation_model: String,
    /// Absolute efficiency-fraction margin required by the decision policy.
    pub superiority_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateDecisionResult {
    pub scenario_id: String,
    pub oracle_true_efficiency_difference: f64,
    pub observed_efficiency_difference: Option<f64>,
    pub registered_margin: f64,
    pub comparator_decision: String,
    pub conservative_lower_bound: Option<f64>,
    pub false_superiority: bool,
    pub false_negative: bool,
    pub indeterminate: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateDecisionSummary {
    pub scenario_count: usize,
    pub truly_superior_cases: usize,
    pub truly_non_superior_cases: usize,
    pub affirmative_superiority_claims: usize,
    pub false_superiority_claims: usize,
    /// False-superiority claims / all oracle-non-superior cases.
    pub false_superiority_rate: Option<f64>,
    pub false_negative_cases: usize,
    /// Non-affirmative outcomes / all oracle-superior cases. Includes abstentions.
    pub false_negative_rate: Option<f64>,
    pub indeterminate_cases: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateDecisionBenchmarkReport {
    pub protocol_version: u32,
    pub fixture_sha256: String,
    pub suite_sha256: String,
    pub results_sha256: String,
    pub scenario_inputs: Vec<CandidateDecisionScenario>,
    pub results: Vec<CandidateDecisionResult>,
    pub summary: CandidateDecisionSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateDecisionBenchmarkError {
    pub reason: String,
}

impl fmt::Display for CandidateDecisionBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for CandidateDecisionBenchmarkError {}

impl From<ValidationError> for CandidateDecisionBenchmarkError {
    fn from(value: ValidationError) -> Self {
        Self {
            reason: value.to_string(),
        }
    }
}

/// Generate a deterministic grid with controlled bias, uncertainty, and correlation.
///
/// The grid intentionally contains observed wins whose oracle truth misses the
/// registered margin, plus true wins obscured by adverse observations or large
/// uncertainty. The resulting rates characterize this test grid only.
pub fn generate_candidate_decision_scenarios() -> Vec<CandidateDecisionScenario> {
    let truth_deltas = [-0.02, 0.0, 0.01, 0.019, 0.02, 0.021, 0.03, 0.05];
    let observed_deltas = [-0.03, 0.0, 0.01, 0.025, 0.05];
    let uncertainty_scales = [0.25, 1.0, 10.0];
    let correlation_models = ["independent", "unknown"];
    let mut cases = Vec::new();

    for (truth_index, truth) in truth_deltas.iter().enumerate() {
        for (observed_index, observed) in observed_deltas.iter().enumerate() {
            for (uncertainty_index, uncertainty_scale) in
                uncertainty_scales.iter().enumerate()
            {
                for (correlation_index, correlation_model) in
                    correlation_models.iter().enumerate()
                {
                    cases.push(CandidateDecisionScenario {
                        scenario_id: format!(
                            "candidate-decision-t{truth_index:02}-o{observed_index:02}-u{uncertainty_index:02}-c{correlation_index:02}"
                        ),
                        oracle_true_efficiency_difference: *truth,
                        observed_efficiency_difference: *observed,
                        candidate_uncertainty_scale: *uncertainty_scale,
                        candidate_correlation_model: (*correlation_model).into(),
                        superiority_margin: 0.02,
                    });
                }
            }
        }
    }
    cases
}

/// Evaluate a batch against its known synthetic efficiency-difference oracle.
pub fn run_candidate_decision_benchmark(
    scenarios: &[CandidateDecisionScenario],
) -> Result<CandidateDecisionBenchmarkReport, CandidateDecisionBenchmarkError> {
    if scenarios.is_empty() {
        return Err(CandidateDecisionBenchmarkError {
            reason: "candidate-decision benchmark must contain at least one case".into(),
        });
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in scenarios {
        validate_scenario(scenario)?;
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(CandidateDecisionBenchmarkError {
                reason: format!("duplicate candidate-decision scenario ID {:?}", scenario.scenario_id),
            });
        }
    }

    let fixture_sha256 = sha256_bytes(FIXTURE.as_bytes());
    let suite_sha256 = sha256_json(&(PROTOCOL_VERSION, &fixture_sha256, scenarios))?;
    let mut results = Vec::with_capacity(scenarios.len());

    for scenario in scenarios {
        let (candidate, baseline) = comparison_runs(scenario)?;
        let comparison = compare_candidates(&candidate, &baseline)?;
        let affirmed = comparison.decision == ComparisonDecision::LowerBoundClearsRegisteredMargin;
        let truly_superior =
            scenario.oracle_true_efficiency_difference > scenario.superiority_margin;
        results.push(CandidateDecisionResult {
            scenario_id: scenario.scenario_id.clone(),
            oracle_true_efficiency_difference: scenario.oracle_true_efficiency_difference,
            observed_efficiency_difference: comparison.estimated_efficiency_difference,
            registered_margin: scenario.superiority_margin,
            comparator_decision: decision_label(&comparison),
            conservative_lower_bound: comparison.conservative_lower_bound,
            false_superiority: affirmed && !truly_superior,
            false_negative: !affirmed && truly_superior,
            indeterminate: comparison.decision == ComparisonDecision::Indeterminate,
            reason: comparison.reason,
        });
    }

    let summary = summarize(&results);
    let results_sha256 = sha256_json(&(&results, &summary))?;
    Ok(CandidateDecisionBenchmarkReport {
        protocol_version: PROTOCOL_VERSION,
        fixture_sha256,
        suite_sha256,
        results_sha256,
        scenario_inputs: scenarios.to_vec(),
        results,
        summary,
    })
}

/// Serialize a candidate-decision benchmark receipt to portable JSON.
pub fn candidate_decision_report_json(
    report: &CandidateDecisionBenchmarkReport,
) -> Result<String, CandidateDecisionBenchmarkError> {
    serde_json::to_string_pretty(report).map_err(|error| CandidateDecisionBenchmarkError {
        reason: format!("cannot serialize candidate-decision receipt: {error}"),
    })
}

/// Parse, recompute, and verify a receipt against the embedded inputs and current fixture.
pub fn parse_and_verify_candidate_decision_report_json(
    json: &str,
) -> Result<CandidateDecisionBenchmarkReport, CandidateDecisionBenchmarkError> {
    let report: CandidateDecisionBenchmarkReport =
        serde_json::from_str(json).map_err(|error| CandidateDecisionBenchmarkError {
            reason: format!("invalid candidate-decision receipt JSON: {error}"),
        })?;
    if report.protocol_version != PROTOCOL_VERSION {
        return Err(CandidateDecisionBenchmarkError {
            reason: format!("unsupported protocol version {}", report.protocol_version),
        });
    }
    let expected = run_candidate_decision_benchmark(&report.scenario_inputs)?;
    if expected != report {
        return Err(CandidateDecisionBenchmarkError {
            reason: "candidate-decision receipt digest or replay mismatch".into(),
        });
    }
    Ok(report)
}

fn comparison_runs(
    scenario: &CandidateDecisionScenario,
) -> Result<(BenchmarkRun, BenchmarkRun), CandidateDecisionBenchmarkError> {
    let mut baseline: BenchmarkRun =
        serde_json::from_str(FIXTURE).map_err(|error| CandidateDecisionBenchmarkError {
            reason: format!("embedded test fixture could not be parsed: {error}"),
        })?;
    baseline.evidence_class = "simulated".into();
    baseline.not_for_decision = false;
    baseline.data_provenance[0].kind = "simulation_output".into();
    baseline.data_provenance[0].sha256 = Some(sha256_bytes(FIXTURE.as_bytes()));
    baseline.uncertainty.correlation_model = "independent".into();
    baseline.uncertainty.method = "synthetic candidate-decision negative control".into();
    baseline.candidate.candidate_id = BASELINE_ID.into();
    baseline.evaluation_policy.baseline_id = BASELINE_ID.into();
    baseline.evaluation_policy.comparison_metric_id = "net_cycle_efficiency".into();
    baseline.evaluation_policy.minimum_superiority_margin = scenario.superiority_margin;
    baseline.evaluation_policy.selection_rule =
        "Synthetic negative-control benchmark only; no physical decision permitted.".into();

    let mut candidate = baseline.clone();
    candidate.candidate.candidate_id = scenario.scenario_id.clone();
    candidate.data_provenance[0].artifact_id = format!("synthetic-case-{}", scenario.scenario_id);
    candidate.data_provenance[0].sha256 = Some(sha256_json(scenario)?);
    candidate.uncertainty.correlation_model = scenario.candidate_correlation_model.clone();
    candidate.uncertainty.method = format!(
        "synthetic negative control; candidate uncertainty scale {}",
        scenario.candidate_uncertainty_scale
    );

    for interval in &mut candidate.intervals {
        let uncertainty = &mut interval.standard_uncertainty_j;
        uncertainty.external_input *= scenario.candidate_uncertainty_scale;
        uncertainty.net_useful_output *= scenario.candidate_uncertainty_scale;
        uncertainty.auxiliary *= scenario.candidate_uncertainty_scale;
        uncertainty.rejected_or_loss *= scenario.candidate_uncertainty_scale;
        uncertainty.stored_energy_change *= scenario.candidate_uncertainty_scale;
    }

    // Inject an observed output delta while independently maintaining first-law
    // closure. The oracle truth remains only in the scenario record.
    let observed_output_delta_j =
        scenario.observed_efficiency_difference * BASELINE_INPUT_J;
    candidate.intervals[1].energy_j.net_useful_output = 35.0 + observed_output_delta_j;
    candidate.intervals[1].energy_j.rejected_or_loss = 59.0 - observed_output_delta_j;
    let output_total_j = BASELINE_OUTPUT_J + observed_output_delta_j;
    let metrics = candidate.computed_metrics.as_mut().ok_or_else(|| {
        CandidateDecisionBenchmarkError {
            reason: "embedded fixture must provide computed_metrics".into(),
        }
    })?;
    metrics.total_external_input_j = Some(BASELINE_INPUT_J);
    metrics.total_net_useful_output_j = Some(output_total_j);
    metrics.net_stored_energy_change_j = Some(0.0);
    metrics.net_cycle_efficiency = Some(output_total_j / BASELINE_INPUT_J);

    Ok((candidate, baseline))
}

fn validate_scenario(
    scenario: &CandidateDecisionScenario,
) -> Result<(), CandidateDecisionBenchmarkError> {
    if scenario.scenario_id.trim().is_empty() {
        return Err(CandidateDecisionBenchmarkError {
            reason: "scenario ID must be non-empty".into(),
        });
    }
    if !scenario.superiority_margin.is_finite()
        || !(0.0..=1.0).contains(&scenario.superiority_margin)
    {
        return Err(CandidateDecisionBenchmarkError {
            reason: "superiority margin must be finite and in [0, 1]".into(),
        });
    }
    for (label, value) in [
        ("oracle truth difference", scenario.oracle_true_efficiency_difference),
        ("observed efficiency difference", scenario.observed_efficiency_difference),
    ] {
        if !value.is_finite()
            || value < -BASELINE_EFFICIENCY
            || value > 1.0 - BASELINE_EFFICIENCY
        {
            return Err(CandidateDecisionBenchmarkError {
                reason: format!("{label} implies efficiency outside [0, 1]"),
            });
        }
    }
    if !scenario.candidate_uncertainty_scale.is_finite()
        || scenario.candidate_uncertainty_scale <= 0.0
        || scenario.candidate_uncertainty_scale > 1e6
    {
        return Err(CandidateDecisionBenchmarkError {
            reason: "candidate uncertainty scale must be finite and in (0, 1e6]".into(),
        });
    }
    if scenario.candidate_correlation_model.trim().is_empty() {
        return Err(CandidateDecisionBenchmarkError {
            reason: "candidate correlation model must be declared".into(),
        });
    }
    Ok(())
}

fn summarize(results: &[CandidateDecisionResult]) -> CandidateDecisionSummary {
    let truly_superior_cases = results
        .iter()
        .filter(|result| result.oracle_true_efficiency_difference > result.registered_margin)
        .count();
    let truly_non_superior_cases = results.len() - truly_superior_cases;
    let affirmative_superiority_claims = results
        .iter()
        .filter(|result| result.comparator_decision == "lower_bound_clears_registered_margin")
        .count();
    let false_superiority_claims = results.iter().filter(|result| result.false_superiority).count();
    let false_negative_cases = results.iter().filter(|result| result.false_negative).count();
    let indeterminate_cases = results.iter().filter(|result| result.indeterminate).count();

    CandidateDecisionSummary {
        scenario_count: results.len(),
        truly_superior_cases,
        truly_non_superior_cases,
        affirmative_superiority_claims,
        false_superiority_claims,
        false_superiority_rate: ratio_or_none(false_superiority_claims, truly_non_superior_cases),
        false_negative_cases,
        false_negative_rate: ratio_or_none(false_negative_cases, truly_superior_cases),
        indeterminate_cases,
    }
}

fn ratio_or_none(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn decision_label(comparison: &EfficiencyComparison) -> String {
    match comparison.decision {
        ComparisonDecision::LowerBoundClearsRegisteredMargin => {
            "lower_bound_clears_registered_margin".into()
        }
        ComparisonDecision::LowerBoundDoesNotClearMargin => {
            "lower_bound_does_not_clear_margin".into()
        }
        ComparisonDecision::Indeterminate => "indeterminate".into(),
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_json<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, CandidateDecisionBenchmarkError> {
    let bytes = serde_json::to_vec(value).map_err(|error| CandidateDecisionBenchmarkError {
        reason: format!("cannot serialize candidate-decision benchmark input: {error}"),
    })?;
    Ok(sha256_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controlled_bias_grid_measures_false_superiority_and_false_negative_rates() {
        let scenarios = generate_candidate_decision_scenarios();
        let report = run_candidate_decision_benchmark(&scenarios).unwrap();

        assert_eq!(report.scenario_inputs.len(), 240);
        assert_eq!(report.results.len(), scenarios.len());
        assert!(report.summary.false_superiority_rate.is_some());
        assert!(report.summary.false_negative_rate.is_some());
        assert!(report.summary.false_superiority_claims > 0);
        assert!(report.summary.indeterminate_cases > 0);
        assert!(report.results.iter().all(|result| {
            result.scenario_id.starts_with("candidate-decision-")
                && result.conservative_lower_bound.is_none()
                    == (result.comparator_decision == "indeterminate")
        }));
    }

    #[test]
    fn biased_observation_can_create_a_false_superiority_claim() {
        let mut scenario = generate_candidate_decision_scenarios().remove(0);
        scenario.scenario_id = "false-superiority-negative-control".into();
        scenario.oracle_true_efficiency_difference = 0.0;
        scenario.observed_efficiency_difference = 0.05;
        scenario.candidate_uncertainty_scale = 0.25;
        scenario.candidate_correlation_model = "independent".into();
        scenario.superiority_margin = 0.02;

        let report = run_candidate_decision_benchmark(&[scenario]).unwrap();
        let result = &report.results[0];
        assert!(result.false_superiority);
        assert_eq!(report.summary.false_superiority_claims, 1);
        assert_eq!(report.summary.false_superiority_rate, Some(1.0));
    }

    #[test]
    fn unknown_correlation_fails_closed_even_for_an_apparent_win() {
        let mut scenario = generate_candidate_decision_scenarios().remove(0);
        scenario.scenario_id = "unknown-correlation-negative-control".into();
        scenario.oracle_true_efficiency_difference = 0.05;
        scenario.observed_efficiency_difference = 0.05;
        scenario.candidate_uncertainty_scale = 0.25;
        scenario.candidate_correlation_model = "unknown".into();

        let report = run_candidate_decision_benchmark(&[scenario]).unwrap();
        assert!(report.results[0].indeterminate);
        assert!(!report.results[0].false_superiority);
    }

    #[test]
    fn candidate_decision_receipt_round_trips_and_rejects_tampering() {
        let scenarios = generate_candidate_decision_scenarios();
        let report = run_candidate_decision_benchmark(&scenarios[..12]).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let parsed = parse_and_verify_candidate_decision_report_json(&json).unwrap();
        assert_eq!(report, parsed);

        let mut tampered = report.clone();
        tampered.results[0].false_superiority = !tampered.results[0].false_superiority;
        let tampered_json = serde_json::to_string(&tampered).unwrap();
        assert!(parse_and_verify_candidate_decision_report_json(&tampered_json).is_err());
    }
}
