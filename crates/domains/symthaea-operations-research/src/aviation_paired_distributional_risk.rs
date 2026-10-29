// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bind paired energy-policy outcomes to calibrated finite-support distributional risk.
//!
//! The paired benchmark's scenario weights are design weights, not probabilities.
//! This adapter therefore requires separate calibration frequencies over exactly the
//! same scenario support, derives the TV ambiguity radius from calibration counts, and
//! refuses to score if either selected policy lacks a verified feasible result for any
//! scenario. It never drops failed cases or renormalizes the surviving subset.
//!
//! Results are conditional on the declared finite support, quantization, calibration
//! assumptions, source benchmark digest and caller-supplied receipt ID. Hashes bind
//! content identity; they do not authenticate publishers or establish representativeness.
//! This adapter is not dispatch authority.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use crate::aviation_distributional_robustness::{
    assess_tv_robust_candidate, evaluate_tv_robust_cost_delta, DistributionalRobustAssessment,
    DistributionalRobustConstraints, DistributionalRobustError, DistributionalRobustReport,
    ScenarioCostDelta,
};
use crate::aviation_energy_hub::{
    canonical_energy_hub_digest, verify_energy_hub_plan, TimeIndexedEnergyHubScenario,
};
use crate::aviation_energy_hub_paired_benchmark::{
    PairedRollingBenchmarkReport, PairedScenarioOutcome,
};
use crate::aviation_energy_hub_rolling::RollingHorizonStatus;
use crate::aviation_probability_calibration::{
    calibrate_tv_radius, CalibrationFrequency, RadiusCalibrationError, TvRadiusCalibrationReport,
};

/// Maximum supported integer cost units per one unit of scenario currency.
pub const MAX_COST_UNITS_PER_CURRENCY_UNIT: u64 = 1_000_000_000;

/// Deployable policies eligible for comparison. The perfect-information solver is
/// intentionally excluded because it is an optimistic oracle, not an online policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeployableEnergyPolicy {
    Static,
    Greedy,
    RollingHorizon,
}

impl DeployableEnergyPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Greedy => "greedy",
            Self::RollingHorizon => "rolling_horizon",
        }
    }
}

/// Complete output from the paired-policy-to-distributional-risk adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedDistributionalRiskReport {
    /// Content digest over the source report, comparison, calibration inputs, cost
    /// scale, thresholds and paired integer cost deltas.
    pub assessment_digest: String,
    pub benchmark_digest: String,
    pub baseline_scenario_digest: String,
    /// Opaque caller-supplied identifier; it is not a cryptographic signature.
    pub calibration_receipt_id: String,
    pub candidate_policy: DeployableEnergyPolicy,
    pub baseline_policy: DeployableEnergyPolicy,
    pub currency_code: String,
    pub cost_units_per_currency_unit: u64,
    /// Upper bound on quantization error for a candidate-minus-baseline cost delta,
    /// assuming the two source costs are rounded independently.
    pub maximum_pair_delta_quantization_error_currency: f64,
    pub calibration: TvRadiusCalibrationReport,
    pub paired_cost_deltas: Vec<ScenarioCostDelta>,
    pub distributional_risk: DistributionalRobustReport,
    pub assessment: DistributionalRobustAssessment,
}

/// Fail-closed errors from adapter input validation or risk evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairedDistributionalRiskError {
    EmptyBenchmarkDigest,
    EmptyCalibrationReceiptId,
    EmptyScenarioPanel,
    EmptyCurrencyCode,
    SamePolicyComparison,
    BaselineScenarioInvalid,
    BaselineScenarioDigestMismatch,
    InvalidRealizedScenarioDigest { scenario_id: String },
    RealizedScenarioSupportMismatch { scenario_id: String },
    PolicyPlanVerificationFailed {
        scenario_id: String,
        policy: DeployableEnergyPolicy,
    },
    ReportedPolicyMetricsMismatch {
        scenario_id: String,
        policy: DeployableEnergyPolicy,
    },
    InvalidCostUnitScale { scale: u64 },
    DuplicateScenarioId { scenario_id: String },
    CalibrationSupportMismatch { scenario_id: String },
    PolicyOutcomeNotUsable {
        scenario_id: String,
        policy: DeployableEnergyPolicy,
    },
    InvalidPolicyCost {
        scenario_id: String,
        policy: DeployableEnergyPolicy,
    },
    CostQuantizationOverflow {
        scenario_id: String,
        policy: DeployableEnergyPolicy,
    },
    CalibratedRadiusMismatch {
        calibrated_radius_ppm: u32,
        constraint_radius_ppm: u32,
    },
    Calibration(RadiusCalibrationError),
    Distributional(DistributionalRobustError),
}

impl std::fmt::Display for PairedDistributionalRiskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBenchmarkDigest => write!(f, "source benchmark digest must not be empty"),
            Self::EmptyCalibrationReceiptId => write!(f, "calibration receipt ID must not be empty"),
            Self::EmptyScenarioPanel => write!(f, "paired benchmark must contain scenarios"),
            Self::EmptyCurrencyCode => write!(f, "scenario currency code must not be empty"),
            Self::SamePolicyComparison => write!(f, "candidate and baseline policies must differ"),
            Self::BaselineScenarioInvalid => write!(f, "baseline scenario cannot be canonically digested"),
            Self::BaselineScenarioDigestMismatch => {
                write!(f, "paired report is not bound to the supplied baseline scenario")
            }
            Self::InvalidRealizedScenarioDigest { scenario_id } => {
                write!(f, "realized scenario digest is invalid for {scenario_id}")
            }
            Self::RealizedScenarioSupportMismatch { scenario_id } => {
                write!(f, "paired report and realized scenario set differ at {scenario_id}")
            }
            Self::PolicyPlanVerificationFailed { scenario_id, policy } => write!(
                f, "independent plan verification failed for {} in scenario {scenario_id}",
                policy.as_str()
            ),
            Self::ReportedPolicyMetricsMismatch { scenario_id, policy } => write!(
                f, "reported metrics differ from independent plan evaluation for {} in scenario {scenario_id}",
                policy.as_str()
            ),
            Self::InvalidCostUnitScale { scale } => {
                write!(f, "cost unit scale {scale} is outside the supported range")
            }
            Self::DuplicateScenarioId { scenario_id } => {
                write!(f, "paired benchmark contains duplicate scenario ID {scenario_id}")
            }
            Self::CalibrationSupportMismatch { scenario_id } => {
                write!(f, "calibration and paired-evaluation support differ at scenario {scenario_id}")
            }
            Self::PolicyOutcomeNotUsable { scenario_id, policy } => write!(
                f,
                "policy {} lacks one unambiguous verified feasible cost for scenario {scenario_id}",
                policy.as_str()
            ),
            Self::InvalidPolicyCost { scenario_id, policy } => write!(
                f,
                "policy {} has a non-finite or negative cost for scenario {scenario_id}",
                policy.as_str()
            ),
            Self::CostQuantizationOverflow { scenario_id, policy } => write!(
                f,
                "policy {} cost cannot be represented at the declared scale for scenario {scenario_id}",
                policy.as_str()
            ),
            Self::CalibratedRadiusMismatch {
                calibrated_radius_ppm,
                constraint_radius_ppm,
            } => write!(
                f,
                "risk gate radius {constraint_radius_ppm} ppm does not match calibrated radius {calibrated_radius_ppm} ppm"
            ),
            Self::Calibration(error) => write!(f, "calibration input rejected: {error:?}"),
            Self::Distributional(error) => write!(f, "distributional risk evaluation rejected: {error:?}"),
        }
    }
}

impl std::error::Error for PairedDistributionalRiskError {}

/// Evaluate a candidate deployable policy against another deployable policy.
///
/// Calibration frequencies must come from a frozen calibration partition and cover
/// precisely the same stable scenario IDs as the paired evaluation. The adapter derives
/// the TV radius from those counts; the supplied risk gate must use that exact radius.
/// The benchmark's design weights are never used as nominal probabilities.
///
/// A missing, infeasible, unresolved or conflicting result for either policy rejects the
/// distributional comparison. Use the paired report's coverage summaries to report that
/// failure. Do not insert a zero cost, drop the case, or renormalize the distribution.
/// A declared service-failure penalty would define a different objective and must be
/// preregistered and benchmarked separately.
pub fn evaluate_paired_policy_distributional_risk(
    benchmark: &PairedRollingBenchmarkReport,
    baseline_scenario: &TimeIndexedEnergyHubScenario,
    realized_scenarios: &[TimeIndexedEnergyHubScenario],
    candidate_policy: DeployableEnergyPolicy,
    baseline_policy: DeployableEnergyPolicy,
    calibration_frequencies: &[CalibrationFrequency],
    calibration_failure_probability_ppm: u32,
    calibration_receipt_id: &str,
    cost_units_per_currency_unit: u64,
    constraints: DistributionalRobustConstraints,
) -> Result<PairedDistributionalRiskReport, PairedDistributionalRiskError> {
    if benchmark.benchmark_digest.trim().is_empty() {
        return Err(PairedDistributionalRiskError::EmptyBenchmarkDigest);
    }
    if calibration_receipt_id.trim().is_empty() {
        return Err(PairedDistributionalRiskError::EmptyCalibrationReceiptId);
    }
    if benchmark.outcomes.is_empty() {
        return Err(PairedDistributionalRiskError::EmptyScenarioPanel);
    }
    if baseline_scenario.currency_code.trim().is_empty() {
        return Err(PairedDistributionalRiskError::EmptyCurrencyCode);
    }
    if candidate_policy == baseline_policy {
        return Err(PairedDistributionalRiskError::SamePolicyComparison);
    }
    if cost_units_per_currency_unit == 0
        || cost_units_per_currency_unit > MAX_COST_UNITS_PER_CURRENCY_UNIT
    {
        return Err(PairedDistributionalRiskError::InvalidCostUnitScale {
            scale: cost_units_per_currency_unit,
        });
    }

    let baseline_digest = canonical_energy_hub_digest(baseline_scenario)
        .map_err(|_| PairedDistributionalRiskError::BaselineScenarioInvalid)?;
    if baseline_scenario.input_digest != baseline_digest
        || benchmark.baseline_scenario_digest != baseline_digest
    {
        return Err(PairedDistributionalRiskError::BaselineScenarioDigestMismatch);
    }

    // Recompute every realized scenario digest and require an exact one-to-one
    // mapping to report outcomes. The adapter will independently re-evaluate each
    // selected policy plan against this scenario manifest before reading its cost.
    let mut realized_by_id: BTreeMap<String, &TimeIndexedEnergyHubScenario> = BTreeMap::new();
    for scenario in realized_scenarios {
        if scenario.scenario_id.trim().is_empty()
            || realized_by_id.contains_key(&scenario.scenario_id)
        {
            return Err(PairedDistributionalRiskError::DuplicateScenarioId {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        let scenario_digest = canonical_energy_hub_digest(scenario).map_err(|_| {
            PairedDistributionalRiskError::InvalidRealizedScenarioDigest {
                scenario_id: scenario.scenario_id.clone(),
            }
        })?;
        if scenario.input_digest != scenario_digest {
            return Err(PairedDistributionalRiskError::InvalidRealizedScenarioDigest {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        realized_by_id.insert(scenario.scenario_id.clone(), scenario);
    }

    let mut outcome_ids = BTreeSet::new();
    for outcome in &benchmark.outcomes {
        if outcome.scenario_id.trim().is_empty()
            || !outcome_ids.insert(outcome.scenario_id.clone())
        {
            return Err(PairedDistributionalRiskError::DuplicateScenarioId {
                scenario_id: outcome.scenario_id.clone(),
            });
        }
    }
    let realized_ids: BTreeSet<String> = realized_by_id.keys().cloned().collect();
    if outcome_ids != realized_ids {
        let differing_id = outcome_ids
            .symmetric_difference(&realized_ids)
            .next()
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_owned());
        return Err(PairedDistributionalRiskError::RealizedScenarioSupportMismatch {
            scenario_id: differing_id,
        });
    }

    let calibration_ids: BTreeSet<String> = calibration_frequencies
        .iter()
        .map(|item| item.scenario_id.clone())
        .collect();
    if calibration_ids.len() != calibration_frequencies.len() {
        // Ask the calibration validator to return its specific duplicate-ID error.
        calibrate_tv_radius(calibration_frequencies, calibration_failure_probability_ppm)
            .map_err(PairedDistributionalRiskError::Calibration)?;
    }
    if outcome_ids != calibration_ids {
        let differing_id = outcome_ids
            .symmetric_difference(&calibration_ids)
            .next()
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_owned());
        return Err(PairedDistributionalRiskError::CalibrationSupportMismatch {
            scenario_id: differing_id,
        });
    }

    let calibration = calibrate_tv_radius(
        calibration_frequencies,
        calibration_failure_probability_ppm,
    )
    .map_err(PairedDistributionalRiskError::Calibration)?;
    if calibration.required_tv_radius_ppm != constraints.tv_radius_ppm {
        return Err(PairedDistributionalRiskError::CalibratedRadiusMismatch {
            calibrated_radius_ppm: calibration.required_tv_radius_ppm,
            constraint_radius_ppm: constraints.tv_radius_ppm,
        });
    }

    let probabilities_by_id: BTreeMap<&str, &CalibrationFrequency> = calibration_frequencies
        .iter()
        .map(|item| (item.scenario_id.as_str(), item))
        .collect();

    let mut ordered_outcomes: Vec<&PairedScenarioOutcome> = benchmark.outcomes.iter().collect();
    ordered_outcomes.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let mut deltas = Vec::with_capacity(ordered_outcomes.len());
    for outcome in ordered_outcomes {
        let probability = probabilities_by_id
            .get(outcome.scenario_id.as_str())
            .ok_or_else(|| PairedDistributionalRiskError::CalibrationSupportMismatch {
                scenario_id: outcome.scenario_id.clone(),
            })?;

        let realized_scenario = realized_by_id.get(&outcome.scenario_id).ok_or_else(|| {
            PairedDistributionalRiskError::RealizedScenarioSupportMismatch {
                scenario_id: outcome.scenario_id.clone(),
            }
        })?;
        let candidate_cost = policy_cost(outcome, candidate_policy, realized_scenario)?;
        let baseline_cost = policy_cost(outcome, baseline_policy, realized_scenario)?;
        let candidate_units = quantize_cost(
            &outcome.scenario_id, candidate_policy, candidate_cost,
            cost_units_per_currency_unit,
        )?;
        let baseline_units = quantize_cost(
            &outcome.scenario_id, baseline_policy, baseline_cost,
            cost_units_per_currency_unit,
        )?;
        let delta = candidate_units.checked_sub(baseline_units).ok_or_else(|| {
            PairedDistributionalRiskError::CostQuantizationOverflow {
                scenario_id: outcome.scenario_id.clone(),
                policy: candidate_policy,
            }
        })?;

        deltas.push(ScenarioCostDelta {
            scenario_id: outcome.scenario_id.clone(),
            nominal_probability_ppm: probability.nominal_probability_ppm,
            candidate_minus_baseline_cost_units: delta,
        });
    }

    let distributional_risk = evaluate_tv_robust_cost_delta(
        &deltas, calibration.required_tv_radius_ppm,
    ).map_err(PairedDistributionalRiskError::Distributional)?;
    let assessment = assess_tv_robust_candidate(&distributional_risk, constraints)
        .map_err(PairedDistributionalRiskError::Distributional)?;

    let assessment_digest = paired_risk_digest(
        &benchmark.benchmark_digest,
        &baseline_digest,
        calibration_receipt_id,
        candidate_policy,
        baseline_policy,
        &baseline_scenario.currency_code,
        cost_units_per_currency_unit,
        calibration_failure_probability_ppm,
        constraints,
        calibration_frequencies,
        &deltas,
        &realized_by_id,
    );

    Ok(PairedDistributionalRiskReport {
        assessment_digest,
        benchmark_digest: benchmark.benchmark_digest.clone(),
        baseline_scenario_digest: baseline_digest,
        calibration_receipt_id: calibration_receipt_id.to_owned(),
        candidate_policy,
        baseline_policy,
        currency_code: baseline_scenario.currency_code.clone(),
        cost_units_per_currency_unit,
        maximum_pair_delta_quantization_error_currency:
            1.0 / cost_units_per_currency_unit as f64,
        calibration,
        paired_cost_deltas: deltas,
        distributional_risk,
        assessment,
    })
}

fn policy_cost(
    outcome: &PairedScenarioOutcome,
    policy: DeployableEnergyPolicy,
    realized_scenario: &TimeIndexedEnergyHubScenario,
) -> Result<f64, PairedDistributionalRiskError> {
    let reported_plan = match policy {
        DeployableEnergyPolicy::Static => match (&outcome.static_plan, &outcome.static_plan_error) {
            (Some(plan), None) => Some(plan),
            _ => None,
        },
        DeployableEnergyPolicy::Greedy => match (&outcome.greedy_plan, &outcome.greedy_error) {
            (Some(plan), None) => Some(plan),
            _ => None,
        },
        DeployableEnergyPolicy::RollingHorizon => match (
            &outcome.rolling_run.status,
            &outcome.rolling_run.evaluated_realized_plan,
        ) {
            (RollingHorizonStatus::Completed, Some(plan)) => Some(plan),
            _ => None,
        },
    }
    .ok_or_else(|| PairedDistributionalRiskError::PolicyOutcomeNotUsable {
        scenario_id: outcome.scenario_id.clone(),
        policy,
    })?;

    // Do not trust caller-supplied summary metrics. Recompute the feasible plan and
    // complete ledger from the pinned realized scenario; reject any divergence.
    let independently_evaluated =
        verify_energy_hub_plan(realized_scenario, &reported_plan.plan).map_err(|_| {
            PairedDistributionalRiskError::PolicyPlanVerificationFailed {
                scenario_id: outcome.scenario_id.clone(),
                policy,
            }
        })?;
    if independently_evaluated.metrics != reported_plan.metrics {
        return Err(PairedDistributionalRiskError::ReportedPolicyMetricsMismatch {
            scenario_id: outcome.scenario_id.clone(),
            policy,
        });
    }
    Ok(independently_evaluated.metrics.source_ledger.total_cost)
}

fn quantize_cost(
    scenario_id: &str,
    policy: DeployableEnergyPolicy,
    cost: f64,
    scale: u64,
) -> Result<i64, PairedDistributionalRiskError> {
    if !cost.is_finite() || cost < 0.0 {
        return Err(PairedDistributionalRiskError::InvalidPolicyCost {
            scenario_id: scenario_id.to_owned(),
            policy,
        });
    }
    let scaled = cost * scale as f64;
    if !scaled.is_finite() || scaled < 0.0 || scaled > i64::MAX as f64 {
        return Err(PairedDistributionalRiskError::CostQuantizationOverflow {
            scenario_id: scenario_id.to_owned(),
            policy,
        });
    }
    let rounded = scaled.round();
    if rounded > i64::MAX as f64 {
        return Err(PairedDistributionalRiskError::CostQuantizationOverflow {
            scenario_id: scenario_id.to_owned(),
            policy,
        });
    }
    Ok(rounded as i64)
}

fn paired_risk_digest(
    benchmark_digest: &str,
    baseline_scenario_digest: &str,
    calibration_receipt_id: &str,
    candidate_policy: DeployableEnergyPolicy,
    baseline_policy: DeployableEnergyPolicy,
    currency_code: &str,
    cost_units_per_currency_unit: u64,
    failure_probability_ppm: u32,
    constraints: DistributionalRobustConstraints,
    frequencies: &[CalibrationFrequency],
    deltas: &[ScenarioCostDelta],
    realized_scenarios: &BTreeMap<String, &TimeIndexedEnergyHubScenario>,
) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.paired-distributional-risk.v1\0");
    put_str(&mut bytes, benchmark_digest);
    put_str(&mut bytes, baseline_scenario_digest);
    put_str(&mut bytes, calibration_receipt_id);
    put_str(&mut bytes, candidate_policy.as_str());
    put_str(&mut bytes, baseline_policy.as_str());
    put_str(&mut bytes, currency_code);
    bytes.extend_from_slice(&cost_units_per_currency_unit.to_le_bytes());
    bytes.extend_from_slice(&failure_probability_ppm.to_le_bytes());
    bytes.extend_from_slice(&constraints.tv_radius_ppm.to_le_bytes());
    bytes.extend_from_slice(&constraints.max_worst_case_expected_delta_units.to_le_bytes());
    bytes.extend_from_slice(&constraints.max_single_scenario_regression_units.to_le_bytes());

    bytes.extend_from_slice(&(realized_scenarios.len() as u64).to_le_bytes());
    for (scenario_id, scenario) in realized_scenarios {
        put_str(&mut bytes, scenario_id);
        put_str(&mut bytes, &scenario.input_digest);
    }

    let mut sorted_frequencies: Vec<&CalibrationFrequency> = frequencies.iter().collect();
    sorted_frequencies.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    bytes.extend_from_slice(&(sorted_frequencies.len() as u64).to_le_bytes());
    for item in sorted_frequencies {
        put_str(&mut bytes, &item.scenario_id);
        bytes.extend_from_slice(&item.nominal_probability_ppm.to_le_bytes());
        bytes.extend_from_slice(&item.observed_count.to_le_bytes());
    }

    let mut sorted_deltas: Vec<&ScenarioCostDelta> = deltas.iter().collect();
    sorted_deltas.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    bytes.extend_from_slice(&(sorted_deltas.len() as u64).to_le_bytes());
    for item in sorted_deltas {
        put_str(&mut bytes, &item.scenario_id);
        bytes.extend_from_slice(&item.nominal_probability_ppm.to_le_bytes());
        bytes.extend_from_slice(&item.candidate_minus_baseline_cost_units.to_le_bytes());
    }
    format!("{:x}", Sha256::digest(bytes))
}

fn put_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aviation_energy_accounting::{EvidenceClass, TransportService};
    use crate::aviation_energy_hub::{
        seal_energy_hub_scenario, EnergyHubInterval, EnergyHubPlan, EnergyHubStepAction,
        EnergyStorageSpec, EvaluatedEnergyHubPlan,
    };
    use crate::aviation_energy_hub_paired_benchmark::{
        PairedDeltaSummary, PolicyCoverageSummary,
    };
    use crate::aviation_energy_hub_rolling::RollingHorizonRun;

    fn realized_scenario(scenario_id: &str, grid_cost: f64) -> TimeIndexedEnergyHubScenario {
        let mut scenario = TimeIndexedEnergyHubScenario {
            scenario_id: scenario_id.to_owned(),
            input_digest: "pending".to_owned(),
            currency_code: "USD".to_owned(),
            quantum_kwh: 1.0,
            service: TransportService {
                period_id: "paired-panel-period-v1".into(),
                lifecycle_boundary_id: "source-to-load-v1".into(),
                completed_passenger_km: 100.0,
                available_seat_km: 120.0,
            },
            storage: EnergyStorageSpec {
                capacity_quanta: 0,
                initial_soc_quanta: 0,
                initial_soc_ledger_ref: "synthetic:no-opening-inventory".into(),
                max_charge_input_quanta_per_interval: 0,
                max_discharge_output_quanta_per_interval: 0,
                charge_efficiency_ppm: 1_000_000,
                discharge_efficiency_ppm: 1_000_000,
                minimum_terminal_soc_quanta: 0,
            },
            intervals: vec![EnergyHubInterval {
                id: "t0".into(),
                demand_quanta: 1,
                charger_capacity_quanta: 1,
                renewable_generation_quanta: 1,
                grid_import_capacity_quanta: 1,
                renewable_cost_per_quantum: 0.0,
                renewable_lifecycle_co2e_kg_per_quantum: 0.0,
                renewable_evidence_ref: "synthetic:renewable:t0".into(),
                renewable_evidence_class: EvidenceClass::Synthetic,
                grid_cost_per_quantum: grid_cost,
                grid_lifecycle_co2e_kg_per_quantum: 10.0 * grid_cost,
                grid_evidence_ref: "synthetic:grid:t0".into(),
                grid_evidence_class: EvidenceClass::Synthetic,
            }],
        };
        seal_energy_hub_scenario(&mut scenario).unwrap();
        scenario
    }

    fn realized_panel() -> Vec<TimeIndexedEnergyHubScenario> {
        vec![
            realized_scenario("normal", 1.0),
            realized_scenario("disruption", 2.0),
        ]
    }

    fn evaluated_plan(
        scenario: &TimeIndexedEnergyHubScenario,
        use_grid: bool,
    ) -> EvaluatedEnergyHubPlan {
        let action = if use_grid {
            EnergyHubStepAction {
                interval_id: "t0".into(),
                renewable_direct_quanta: 0,
                grid_import_quanta: 1,
                battery_discharge_output_quanta: 0,
                renewable_charge_input_quanta: 0,
                renewable_curtailed_quanta: 1,
                soc_before_quanta: 0,
                soc_after_quanta: 0,
            }
        } else {
            EnergyHubStepAction {
                interval_id: "t0".into(),
                renewable_direct_quanta: 1,
                grid_import_quanta: 0,
                battery_discharge_output_quanta: 0,
                renewable_charge_input_quanta: 0,
                renewable_curtailed_quanta: 0,
                soc_before_quanta: 0,
                soc_after_quanta: 0,
            }
        };
        let plan = EnergyHubPlan {
            scenario_id: scenario.scenario_id.clone(),
            input_digest: scenario.input_digest.clone(),
            actions: vec![action],
        };
        verify_energy_hub_plan(scenario, &plan).unwrap()
    }

    fn paired_report(scenarios: &[TimeIndexedEnergyHubScenario]) -> PairedRollingBenchmarkReport {
        let mut outcomes = Vec::new();
        for scenario in scenarios {
            let static_plan = evaluated_plan(scenario, true);
            let greedy_plan = evaluated_plan(scenario, true);
            let rolling_plan = evaluated_plan(scenario, false);
            let weight_ppm = if scenario.scenario_id == "normal" {
                900_000
            } else {
                100_000
            };
            outcomes.push(PairedScenarioOutcome {
                scenario_id: scenario.scenario_id.clone(),
                weight_ppm,
                static_plan: Some(static_plan),
                static_plan_error: None,
                greedy_plan: Some(greedy_plan),
                greedy_error: None,
                rolling_run: RollingHorizonRun {
                    episode_digest: format!("episode-{}", scenario.scenario_id),
                    steps: vec![],
                    status: RollingHorizonStatus::Completed,
                    final_soc_quanta: 0,
                    evaluated_realized_plan: Some(rolling_plan.clone()),
                },
                perfect_information_plan: Some(rolling_plan),
                perfect_information_error: None,
            });
        }
        PairedRollingBenchmarkReport {
            benchmark_digest: "a".repeat(64),
            baseline_scenario_digest: String::new(),
            outcomes,
            static_plan_summary: empty_coverage(),
            greedy_summary: empty_coverage(),
            rolling_horizon_summary: empty_coverage(),
            perfect_information_summary: empty_coverage(),
            rolling_regret_vs_perfect_information: empty_delta(),
            static_minus_rolling_cost: empty_delta(),
            greedy_minus_rolling_cost: empty_delta(),
        }
    }

    fn empty_coverage() -> PolicyCoverageSummary {
        PolicyCoverageSummary {
            feasible_weight_ppm: 0,
            infeasible_weight_ppm: 0,
            unresolved_weight_ppm: 0,
            mean_cost_given_feasible: None,
            mean_accounted_co2e_kg_given_feasible: None,
            worst_cost_among_feasible_cases: None,
            worst_accounted_co2e_kg_among_feasible_cases: None,
        }
    }

    fn empty_delta() -> PairedDeltaSummary {
        PairedDeltaSummary {
            comparable_weight_ppm: 0,
            mean_cost_delta: None,
            worst_cost_delta: None,
            best_cost_delta: None,
        }
    }

    fn baseline_scenario() -> TimeIndexedEnergyHubScenario {
        realized_scenario("nominal-baseline-v1", 1.0)
    }

    fn calibration() -> Vec<CalibrationFrequency> {
        vec![
            CalibrationFrequency {
                scenario_id: "normal".into(),
                nominal_probability_ppm: 500_000,
                observed_count: 50,
            },
            CalibrationFrequency {
                scenario_id: "disruption".into(),
                nominal_probability_ppm: 500_000,
                observed_count: 50,
            },
        ]
    }

    fn constraints(frequencies: &[CalibrationFrequency]) -> DistributionalRobustConstraints {
        let report = calibrate_tv_radius(frequencies, 50_000).unwrap();
        DistributionalRobustConstraints {
            tv_radius_ppm: report.required_tv_radius_ppm,
            max_worst_case_expected_delta_units: 0,
            max_single_scenario_regression_units: 0,
        }
    }

    #[test]
    fn uses_calibrated_probabilities_not_scenario_design_weights() {
        let scenarios = realized_panel();
        let baseline = baseline_scenario();
        let mut benchmark = paired_report(&scenarios);
        benchmark.baseline_scenario_digest = canonical_energy_hub_digest(&baseline).unwrap();
        let frequencies = calibration();
        let report = evaluate_paired_policy_distributional_risk(
            &benchmark,
            &baseline,
            &scenarios,
            DeployableEnergyPolicy::RollingHorizon,
            DeployableEnergyPolicy::Static,
            &frequencies,
            50_000,
            "calibration-receipt-v1",
            1_000_000,
            constraints(&frequencies),
        )
        .unwrap();

        assert_eq!(report.distributional_risk.probabilities()[0].scenario_id, "disruption");
        assert_eq!(
            report.distributional_risk.nominal_expected_delta_numerator_ppm(),
            -1_500_000_000_000
        );
        assert_eq!(report.paired_cost_deltas.len(), 2);
        assert!(!report.assessment_digest.is_empty());
        assert_eq!(report.currency_code, "USD");
    }

    #[test]
    fn refuses_missing_policy_outcome_instead_of_dropping_probability_mass() {
        let scenarios = realized_panel();
        let baseline = baseline_scenario();
        let mut benchmark = paired_report(&scenarios);
        benchmark.baseline_scenario_digest = canonical_energy_hub_digest(&baseline).unwrap();
        benchmark.outcomes[1].static_plan = None;
        benchmark.outcomes[1].static_plan_error =
            Some(crate::aviation_energy_hub::EnergyHubError::NoFeasiblePlan);
        let frequencies = calibration();
        let result = evaluate_paired_policy_distributional_risk(
            &benchmark,
            &baseline,
            &scenarios,
            DeployableEnergyPolicy::RollingHorizon,
            DeployableEnergyPolicy::Static,
            &frequencies,
            50_000,
            "calibration-receipt-v1",
            1_000_000,
            constraints(&frequencies),
        );

        assert!(matches!(
            result,
            Err(PairedDistributionalRiskError::PolicyOutcomeNotUsable {
                scenario_id,
                policy: DeployableEnergyPolicy::Static,
            }) if scenario_id == "disruption"
        ));
    }

    #[test]
    fn refuses_different_calibration_support() {
        let scenarios = realized_panel();
        let baseline = baseline_scenario();
        let mut benchmark = paired_report(&scenarios);
        benchmark.baseline_scenario_digest = canonical_energy_hub_digest(&baseline).unwrap();
        let frequencies = vec![CalibrationFrequency {
            scenario_id: "normal".into(),
            nominal_probability_ppm: 1_000_000,
            observed_count: 100,
        }];
        let result = evaluate_paired_policy_distributional_risk(
            &benchmark,
            &baseline,
            &scenarios,
            DeployableEnergyPolicy::RollingHorizon,
            DeployableEnergyPolicy::Static,
            &frequencies,
            50_000,
            "calibration-receipt-v1",
            1_000_000,
            constraints(&frequencies),
        );
        assert!(matches!(
            result,
            Err(PairedDistributionalRiskError::CalibrationSupportMismatch { .. })
        ));
    }

    #[test]
    fn refuses_realized_scenario_set_mismatch() {
        let scenarios = realized_panel();
        let baseline = baseline_scenario();
        let mut benchmark = paired_report(&scenarios);
        benchmark.baseline_scenario_digest = canonical_energy_hub_digest(&baseline).unwrap();
        let frequencies = calibration();
        let result = evaluate_paired_policy_distributional_risk(
            &benchmark,
            &baseline,
            &scenarios[..1],
            DeployableEnergyPolicy::RollingHorizon,
            DeployableEnergyPolicy::Static,
            &frequencies,
            50_000,
            "calibration-receipt-v1",
            1_000_000,
            constraints(&frequencies),
        );
        assert!(matches!(
            result,
            Err(PairedDistributionalRiskError::RealizedScenarioSupportMismatch { .. })
        ));
    }

    #[test]
    fn rejects_a_plan_whose_reported_metrics_were_tampered() {
        let scenarios = realized_panel();
        let baseline = baseline_scenario();
        let mut benchmark = paired_report(&scenarios);
        benchmark.baseline_scenario_digest = canonical_energy_hub_digest(&baseline).unwrap();
        benchmark.outcomes[0]
            .static_plan
            .as_mut()
            .unwrap()
            .metrics
            .source_ledger
            .total_cost = 999.0;
        let frequencies = calibration();
        let result = evaluate_paired_policy_distributional_risk(
            &benchmark,
            &baseline,
            &scenarios,
            DeployableEnergyPolicy::RollingHorizon,
            DeployableEnergyPolicy::Static,
            &frequencies,
            50_000,
            "calibration-receipt-v1",
            1_000_000,
            constraints(&frequencies),
        );
        assert!(matches!(
            result,
            Err(PairedDistributionalRiskError::ReportedPolicyMetricsMismatch {
                scenario_id,
                policy: DeployableEnergyPolicy::Static,
            }) if scenario_id == "normal"
        ));
    }

    #[test]
    fn refuses_a_benchmark_not_bound_to_the_supplied_static_scenario() {
        let scenarios = realized_panel();
        let baseline = baseline_scenario();
        let mut benchmark = paired_report(&scenarios);
        benchmark.baseline_scenario_digest = "f".repeat(64);
        let frequencies = calibration();
        let result = evaluate_paired_policy_distributional_risk(
            &benchmark,
            &baseline,
            &scenarios,
            DeployableEnergyPolicy::RollingHorizon,
            DeployableEnergyPolicy::Static,
            &frequencies,
            50_000,
            "calibration-receipt-v1",
            1_000_000,
            constraints(&frequencies),
        );
        assert!(matches!(
            result,
            Err(PairedDistributionalRiskError::BaselineScenarioDigestMismatch)
        ));
    }

    #[test]
    fn refuses_a_risk_gate_with_unmatched_tv_radius() {
        let scenarios = realized_panel();
        let baseline = baseline_scenario();
        let mut benchmark = paired_report(&scenarios);
        benchmark.baseline_scenario_digest = canonical_energy_hub_digest(&baseline).unwrap();
        let frequencies = calibration();
        let mut bad_constraints = constraints(&frequencies);
        bad_constraints.tv_radius_ppm = bad_constraints.tv_radius_ppm.saturating_sub(1);
        let result = evaluate_paired_policy_distributional_risk(
            &benchmark,
            &baseline,
            &scenarios,
            DeployableEnergyPolicy::RollingHorizon,
            DeployableEnergyPolicy::Static,
            &frequencies,
            50_000,
            "calibration-receipt-v1",
            1_000_000,
            bad_constraints,
        );
        assert!(matches!(
            result,
            Err(PairedDistributionalRiskError::CalibratedRadiusMismatch { .. })
        ));
    }
}
