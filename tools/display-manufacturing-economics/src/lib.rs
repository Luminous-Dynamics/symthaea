#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Auditable output metrics for display-manufacturing route benchmarks.
//!
//! This crate computes batch-level first-pass yield, final calibrated-good yield,
//! cost per calibrated-good unit, and inspection metrics from supplied counts.
//! It does not model a process, infer stage independence, value human perception,
//! or qualify costs, inspections, equipment, or physical manufacturing.

use core::fmt;

/// Batch costs expressed in one declared currency/unit and one accounting boundary.
///
/// Every field must refer to the same batch and currency. Incremental scrap
/// disposal must exclude material costs already included in materials_consumables.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CostLedger {
    /// Material and consumable spend allocated to the batch.
    pub materials_consumables: f64,
    /// Process and manufacturing labor allocated to the batch.
    pub process_labor: f64,
    /// Equipment and facility allocation under a declared costing method.
    pub equipment_facility: f64,
    /// Measured/allocated energy and utility costs.
    pub energy_utilities: f64,
    /// Inspection, metrology and final-test costs.
    pub inspection_test: f64,
    /// Incremental repair, rework and retest costs.
    pub rework_retest: f64,
    /// Packaging and logistics costs.
    pub packaging_logistics: f64,
    /// Incremental scrap handling/disposal cost, excluding material costs above.
    pub incremental_scrap_disposal: f64,
}

/// Unique-unit denominators for a manufacturing batch.
///
/// First-pass good units passed the declared final acceptance criteria without
/// repair. Final calibrated-good units passed the complete declared acceptance
/// profile after any permitted repair/retest and reliability gates actually run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchCounts {
    /// Unique incoming candidate units in this batch.
    pub input_units: u64,
    /// Unique units meeting final acceptance on first pass.
    pub first_pass_good_units: u64,
    /// Unique units meeting final calibrated-good acceptance after permitted rework.
    pub final_calibrated_good_units: u64,
}

/// Identity of accounting scope and product/protocol assumptions for a batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostAccountingBasis {
    /// Stable batch or lot identifier.
    pub batch_id: String,
    /// Three-letter uppercase currency code; conversion is out of scope.
    pub currency_code: String,
    /// Identity of included cost categories and allocation boundary.
    pub accounting_boundary_id: String,
    /// Identity/digest of the target calibrated-good module profile.
    pub target_module_profile_digest: String,
    /// Identity/digest of the comparison protocol and denominator semantics.
    pub comparison_protocol_digest: String,
    /// Explicit class such as SYNTHETIC_SCENARIO or BENCH_MEASURED.
    pub evidence_class: String,
}

/// Explicit summary vector. No stage independence or yield-product assumption is used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchEconomics {
    /// Total batch cost under the supplied ledger.
    pub total_batch_cost: f64,
    /// First-pass good units / unique incoming units.
    pub first_pass_yield: f64,
    /// Final calibrated-good units / unique incoming units.
    pub final_calibrated_good_yield: f64,
    /// Batch cost / final calibrated-good units; None when the denominator is zero.
    pub cost_per_calibrated_good_unit: Option<f64>,
    pub input_units: u64,
    pub first_pass_good_units: u64,
    pub final_calibrated_good_units: u64,
}

/// Batch metrics bound to an explicit product, accounting and evidence identity.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidencedBatchEconomics {
    pub basis: CostAccountingBasis,
    pub metrics: BatchEconomics,
}

/// A route's conservative metric bounds under a pinned target and protocol.
///
/// The supplied values are bounds/summary estimates, not raw production truth.
/// Both candidates must share target-profile, comparison-protocol and evidence-class
/// identity. Pareto comparison will not trade away mandatory quality/reliability/
/// floors or missing evidence for a cheaper cost.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteFrontierCandidate {
    pub candidate_id: String,
    pub target_module_profile_digest: String,
    pub comparison_protocol_digest: String,
    /// Three-letter uppercase currency code, e.g. ZAR or USD.
    pub currency_code: String,
    /// Identity of the included cost categories and allocation boundary.
    pub accounting_boundary_id: String,
    pub evidence_class: String,
    /// Digest/identifier for the evidence bundle supporting these candidate bounds.
    pub evidence_bundle_digest: String,
    /// Digest/identifier for the method used to construct the uncertainty bounds.
    pub uncertainty_method_digest: String,
    /// Upper bound for cost per calibrated-good module in one declared currency.
    pub cost_per_good_unit_upper: f64,
    /// Upper bound for energy per calibrated-good module in joules.
    pub energy_per_good_unit_upper_joules: f64,
    /// Upper bound for elapsed line-cycle time per calibrated-good module, in seconds.
    pub cycle_time_per_good_unit_upper_seconds: f64,
    /// Lower bound for final calibrated-good yield, as a fraction.
    pub final_good_yield_lower: f64,
    pub optical_floors_passed: bool,
    pub reliability_floors_passed: bool,
    pub minimum_yield_floor_passed: bool,
    pub evidence_complete: bool,
}

fn validate_frontier_candidate(
    candidate: &RouteFrontierCandidate,
) -> Result<(), ManufacturingMetricError> {
    if candidate.candidate_id.trim().is_empty()
        || candidate.target_module_profile_digest.trim().is_empty()
        || candidate.comparison_protocol_digest.trim().is_empty()
        || candidate.accounting_boundary_id.trim().is_empty()
        || candidate.evidence_class.trim().is_empty()
        || candidate.evidence_bundle_digest.trim().is_empty()
        || candidate.uncertainty_method_digest.trim().is_empty()
        || candidate.currency_code.len() != 3
        || !candidate
            .currency_code
            .bytes()
            .all(|byte| byte.is_ascii_uppercase())
    {
        return Err(ManufacturingMetricError::InvalidRouteCandidate);
    }
    if !candidate.cost_per_good_unit_upper.is_finite()
        || candidate.cost_per_good_unit_upper < 0.0
        || !candidate.energy_per_good_unit_upper_joules.is_finite()
        || candidate.energy_per_good_unit_upper_joules < 0.0
        || !candidate.cycle_time_per_good_unit_upper_seconds.is_finite()
        || candidate.cycle_time_per_good_unit_upper_seconds <= 0.0
        || !candidate.final_good_yield_lower.is_finite()
        || candidate.final_good_yield_lower <= 0.0
        || candidate.final_good_yield_lower > 1.0
    {
        return Err(ManufacturingMetricError::InvalidRouteCandidate);
    }
    if !candidate.optical_floors_passed
        || !candidate.reliability_floors_passed
        || !candidate.minimum_yield_floor_passed
        || !candidate.evidence_complete
    {
        return Err(ManufacturingMetricError::UnqualifiedRouteCandidate);
    }
    Ok(())
}

/// Tests whether candidate dominates competitor without collapsing metrics into a score.
///
/// Lower cost/energy/cycle-time upper bounds and a higher final-yield lower bound
/// must be no worse in every dimension, and strictly better in at least one.
/// The target module, comparison protocol and evidence class must match; incomparable
/// candidates remain on the Pareto frontier. Inputs must already reflect the declared
/// uncertainty method. This function cannot validate those uncertainty bounds itself.
pub fn route_pareto_dominates(
    candidate: &RouteFrontierCandidate,
    competitor: &RouteFrontierCandidate,
) -> Result<bool, ManufacturingMetricError> {
    validate_frontier_candidate(candidate)?;
    validate_frontier_candidate(competitor)?;

    if candidate.target_module_profile_digest != competitor.target_module_profile_digest
        || candidate.comparison_protocol_digest != competitor.comparison_protocol_digest
        || candidate.currency_code != competitor.currency_code
        || candidate.accounting_boundary_id != competitor.accounting_boundary_id
        || candidate.evidence_class != competitor.evidence_class
    {
        return Err(ManufacturingMetricError::MismatchedRouteComparison);
    }

    let no_worse = candidate.cost_per_good_unit_upper <= competitor.cost_per_good_unit_upper
        && candidate.energy_per_good_unit_upper_joules
            <= competitor.energy_per_good_unit_upper_joules
        && candidate.cycle_time_per_good_unit_upper_seconds
            <= competitor.cycle_time_per_good_unit_upper_seconds
        && candidate.final_good_yield_lower >= competitor.final_good_yield_lower;
    let strictly_better = candidate.cost_per_good_unit_upper < competitor.cost_per_good_unit_upper
        || candidate.energy_per_good_unit_upper_joules
            < competitor.energy_per_good_unit_upper_joules
        || candidate.cycle_time_per_good_unit_upper_seconds
            < competitor.cycle_time_per_good_unit_upper_seconds
        || candidate.final_good_yield_lower > competitor.final_good_yield_lower;

    Ok(no_worse && strictly_better)
}

/// Counts from a defect-inspection study using a declared reference-review method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InspectionCounts {
    /// Total eligible units/regions under the declared inspection frame.
    pub eligible_units: u64,
    /// Unique units/regions actually inspected with a valid result.
    pub inspected_units: u64,
    /// Defects correctly called by the inspection system.
    pub true_positive: u64,
    /// Non-defective units/regions incorrectly called defective.
    pub false_positive: u64,
    /// Defective units/regions missed by the inspection system.
    pub false_negative: u64,
}

/// Inspection metric vector; precision/recall are None when their denominator is zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InspectionMetrics {
    /// Fraction of inspected eligible units/regions: inspected / eligible.
    pub coverage: f64,
    /// Fraction of positive calls that are true positives; None when there are no positive calls.
    pub precision: Option<f64>,
    /// Fraction of known defects detected; None when no defects are represented.
    pub recall: Option<f64>,
    /// False-negative rate among represented defects; None when recall is undefined.
    pub false_negative_rate: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManufacturingMetricError {
    ZeroInputUnits,
    CountExceedsInput,
    NonFiniteCost,
    NegativeCost,
    NonFiniteResult,
    InspectedExceedsEligible,
    InspectionCountsExceedPopulation,
    InvalidAccountingBasis,
    InvalidRouteCandidate,
    UnqualifiedRouteCandidate,
    MismatchedRouteComparison,
}

impl fmt::Display for ManufacturingMetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInputUnits => formatter.write_str("batch input denominator must be positive"),
            Self::CountExceedsInput => formatter.write_str("unit count exceeds unique batch input"),
            Self::NonFiniteCost => formatter.write_str("all cost fields must be finite"),
            Self::NegativeCost => formatter.write_str("cost fields must be nonnegative"),
            Self::NonFiniteResult => formatter.write_str("computed metric must be finite"),
            Self::InspectedExceedsEligible => {
                formatter.write_str("inspected count cannot exceed eligible population")
            }
            Self::InspectionCountsExceedPopulation => {
                formatter.write_str("inspection contingency counts exceed their population")
            }
            Self::InvalidAccountingBasis => {
                formatter.write_str("batch accounting identity is incomplete or invalid")
            }
            Self::InvalidRouteCandidate => {
                formatter.write_str("route candidate bounds must be finite and valid")
            }
            Self::UnqualifiedRouteCandidate => {
                formatter.write_str("route candidate fails a mandatory gate")
            }
            Self::MismatchedRouteComparison => {
                formatter.write_str("route comparison identities do not match")
            }
        }
    }
}

impl std::error::Error for ManufacturingMetricError {}

impl CostLedger {
    /// Sums all cost categories after validating common-unit numeric inputs.
    pub fn total_cost(self) -> Result<f64, ManufacturingMetricError> {
        let values = [
            self.materials_consumables,
            self.process_labor,
            self.equipment_facility,
            self.energy_utilities,
            self.inspection_test,
            self.rework_retest,
            self.packaging_logistics,
            self.incremental_scrap_disposal,
        ];

        if values.iter().any(|value| !value.is_finite()) {
            return Err(ManufacturingMetricError::NonFiniteCost);
        }
        if values.iter().any(|value| *value < 0.0) {
            return Err(ManufacturingMetricError::NegativeCost);
        }

        let total: f64 = values.iter().sum();
        if !total.is_finite() {
            return Err(ManufacturingMetricError::NonFiniteResult);
        }
        Ok(total)
    }
}

/// Calculates cost and yield from unique unit counts, not process attempts.
///
/// The caller must declare a matched product/quality profile and a consistent
/// cost boundary. Stage yields are not multiplied; correlated/clustered defects
/// remain visible in the observed final-good denominator or a separate scenario.
fn validate_accounting_basis(basis: &CostAccountingBasis) -> Result<(), ManufacturingMetricError> {
    let text_fields = [
        basis.batch_id.as_str(),
        basis.accounting_boundary_id.as_str(),
        basis.target_module_profile_digest.as_str(),
        basis.comparison_protocol_digest.as_str(),
        basis.evidence_class.as_str(),
    ];
    if text_fields.iter().any(|value| value.trim().is_empty()) {
        return Err(ManufacturingMetricError::InvalidAccountingBasis);
    }
    if basis.currency_code.len() != 3
        || !basis
            .currency_code
            .bytes()
            .all(|byte| byte.is_ascii_uppercase())
    {
        return Err(ManufacturingMetricError::InvalidAccountingBasis);
    }
    Ok(())
}

/// Batch summary with required accounting/evidence identity attached.
pub fn summarize_batch_with_basis(
    basis: CostAccountingBasis,
    costs: CostLedger,
    counts: BatchCounts,
) -> Result<EvidencedBatchEconomics, ManufacturingMetricError> {
    validate_accounting_basis(&basis)?;
    let metrics = summarize_batch(costs, counts)?;
    Ok(EvidencedBatchEconomics { basis, metrics })
}

pub fn summarize_batch(
    costs: CostLedger,
    counts: BatchCounts,
) -> Result<BatchEconomics, ManufacturingMetricError> {
    if counts.input_units == 0 {
        return Err(ManufacturingMetricError::ZeroInputUnits);
    }
    if counts.first_pass_good_units > counts.input_units
        || counts.final_calibrated_good_units > counts.input_units
    {
        return Err(ManufacturingMetricError::CountExceedsInput);
    }

    let total_batch_cost = costs.total_cost()?;
    let denominator = counts.input_units as f64;
    let first_pass_yield = counts.first_pass_good_units as f64 / denominator;
    let final_calibrated_good_yield = counts.final_calibrated_good_units as f64 / denominator;
    let cost_per_calibrated_good_unit = if counts.final_calibrated_good_units == 0 {
        None
    } else {
        Some(total_batch_cost / counts.final_calibrated_good_units as f64)
    };

    if !first_pass_yield.is_finite()
        || !final_calibrated_good_yield.is_finite()
        || cost_per_calibrated_good_unit.is_some_and(|value| !value.is_finite())
    {
        return Err(ManufacturingMetricError::NonFiniteResult);
    }

    Ok(BatchEconomics {
        total_batch_cost,
        first_pass_yield,
        final_calibrated_good_yield,
        cost_per_calibrated_good_unit,
        input_units: counts.input_units,
        first_pass_good_units: counts.first_pass_good_units,
        final_calibrated_good_units: counts.final_calibrated_good_units,
    })
}

/// Calculates inspection coverage, precision, recall and false-negative rate.
///
/// The caller supplies a declared eligible population and independent reference
/// labels. The function validates arithmetic only; it does not establish that
/// the reference labels are correct or representative.
pub fn summarize_inspection(
    counts: InspectionCounts,
) -> Result<InspectionMetrics, ManufacturingMetricError> {
    if counts.eligible_units == 0 {
        return Err(ManufacturingMetricError::ZeroInputUnits);
    }
    if counts.inspected_units > counts.eligible_units {
        return Err(ManufacturingMetricError::InspectedExceedsEligible);
    }
    let positive_calls = counts
        .true_positive
        .checked_add(counts.false_positive)
        .ok_or(ManufacturingMetricError::InspectionCountsExceedPopulation)?;
    let represented_defects = counts
        .true_positive
        .checked_add(counts.false_negative)
        .ok_or(ManufacturingMetricError::InspectionCountsExceedPopulation)?;
    if positive_calls > counts.inspected_units || represented_defects > counts.eligible_units {
        return Err(ManufacturingMetricError::InspectionCountsExceedPopulation);
    }

    let coverage = counts.inspected_units as f64 / counts.eligible_units as f64;
    let precision = if positive_calls == 0 {
        None
    } else {
        Some(counts.true_positive as f64 / positive_calls as f64)
    };
    let recall = if represented_defects == 0 {
        None
    } else {
        Some(counts.true_positive as f64 / represented_defects as f64)
    };
    let false_negative_rate = recall.map(|value| 1.0 - value);

    if !coverage.is_finite()
        || precision.is_some_and(|value| !value.is_finite())
        || recall.is_some_and(|value| !value.is_finite())
    {
        return Err(ManufacturingMetricError::NonFiniteResult);
    }

    Ok(InspectionMetrics {
        coverage,
        precision,
        recall,
        false_negative_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs() -> CostLedger {
        CostLedger {
            materials_consumables: 100.0,
            process_labor: 20.0,
            equipment_facility: 30.0,
            energy_utilities: 10.0,
            inspection_test: 5.0,
            rework_retest: 15.0,
            packaging_logistics: 20.0,
            incremental_scrap_disposal: 0.0,
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} differed from expected {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn computes_first_pass_yield_final_good_yield_and_cost_per_good_unit() {
        let result = summarize_batch(
            costs(),
            BatchCounts {
                input_units: 10,
                first_pass_good_units: 6,
                final_calibrated_good_units: 8,
            },
        )
        .expect("valid batch ledger");

        close(result.total_batch_cost, 200.0, 1e-12);
        close(result.first_pass_yield, 0.6, 1e-12);
        close(result.final_calibrated_good_yield, 0.8, 1e-12);
        close(
            result
                .cost_per_calibrated_good_unit
                .expect("positive final-good count"),
            25.0,
            1e-12,
        );
    }

    #[test]
    fn batch_economics_preserve_validated_currency_and_accounting_identity() {
        let basis = CostAccountingBasis {
            batch_id: "LOT-2026-001".to_owned(),
            currency_code: "ZAR".to_owned(),
            accounting_boundary_id: "sha256:boundary".to_owned(),
            target_module_profile_digest: "sha256:target".to_owned(),
            comparison_protocol_digest: "sha256:protocol".to_owned(),
            evidence_class: "SYNTHETIC_SCENARIO".to_owned(),
        };
        let result = summarize_batch_with_basis(
            basis.clone(),
            costs(),
            BatchCounts {
                input_units: 10,
                first_pass_good_units: 6,
                final_calibrated_good_units: 8,
            },
        )
        .expect("complete accounting identity");
        assert_eq!(result.basis, basis);
        close(
            result
                .metrics
                .cost_per_calibrated_good_unit
                .expect("good units"),
            25.0,
            1e-12,
        );

        let mut invalid = basis;
        invalid.currency_code = "zar".to_owned();
        assert_eq!(
            summarize_batch_with_basis(
                invalid,
                costs(),
                BatchCounts {
                    input_units: 10,
                    first_pass_good_units: 6,
                    final_calibrated_good_units: 8,
                },
            ),
            Err(ManufacturingMetricError::InvalidAccountingBasis)
        );
    }

    #[test]
    fn zero_final_good_units_keep_cost_per_good_undefined_but_preserve_yield() {
        let result = summarize_batch(
            costs(),
            BatchCounts {
                input_units: 10,
                first_pass_good_units: 0,
                final_calibrated_good_units: 0,
            },
        )
        .expect("zero output is a valid observed outcome");
        assert_eq!(result.final_calibrated_good_yield, 0.0);
        assert_eq!(result.cost_per_calibrated_good_unit, None);
    }

    #[test]
    fn batch_summary_rejects_empty_population_and_impossible_counts() {
        assert_eq!(
            summarize_batch(
                costs(),
                BatchCounts {
                    input_units: 0,
                    first_pass_good_units: 0,
                    final_calibrated_good_units: 0,
                },
            ),
            Err(ManufacturingMetricError::ZeroInputUnits)
        );
        assert_eq!(
            summarize_batch(
                costs(),
                BatchCounts {
                    input_units: 10,
                    first_pass_good_units: 11,
                    final_calibrated_good_units: 10,
                },
            ),
            Err(ManufacturingMetricError::CountExceedsInput)
        );
    }

    #[test]
    fn cost_ledger_rejects_negative_nonfinite_and_overflowed_costs() {
        assert_eq!(
            CostLedger {
                materials_consumables: -1.0,
                ..CostLedger::default()
            }
            .total_cost(),
            Err(ManufacturingMetricError::NegativeCost)
        );
        assert_eq!(
            CostLedger {
                process_labor: f64::NAN,
                ..CostLedger::default()
            }
            .total_cost(),
            Err(ManufacturingMetricError::NonFiniteCost)
        );
        assert_eq!(
            CostLedger {
                materials_consumables: f64::MAX,
                process_labor: f64::MAX,
                ..CostLedger::default()
            }
            .total_cost(),
            Err(ManufacturingMetricError::NonFiniteResult)
        );
    }

    #[test]
    fn inspection_metrics_report_coverage_precision_recall_and_false_negative_rate() {
        let metrics = summarize_inspection(InspectionCounts {
            eligible_units: 100,
            inspected_units: 80,
            true_positive: 18,
            false_positive: 2,
            false_negative: 6,
        })
        .expect("valid inspection counts");

        close(metrics.coverage, 0.8, 1e-12);
        close(metrics.precision.expect("positive calls"), 0.9, 1e-12);
        close(metrics.recall.expect("represented defects"), 0.75, 1e-12);
        close(
            metrics.false_negative_rate.expect("represented defects"),
            0.25,
            1e-12,
        );
    }

    #[test]
    fn undefined_inspection_denominators_are_not_smoothed() {
        let metrics = summarize_inspection(InspectionCounts {
            eligible_units: 50,
            inspected_units: 40,
            true_positive: 0,
            false_positive: 0,
            false_negative: 0,
        })
        .expect("valid no-defect/no-call sample");
        assert_eq!(metrics.precision, None);
        assert_eq!(metrics.recall, None);
        assert_eq!(metrics.false_negative_rate, None);
    }

    fn route_candidate(
        id: &str,
        cost: f64,
        energy: f64,
        cycle: f64,
        yield_lower: f64,
    ) -> RouteFrontierCandidate {
        RouteFrontierCandidate {
            candidate_id: id.to_owned(),
            target_module_profile_digest: "sha256:target-profile".to_owned(),
            comparison_protocol_digest: "sha256:comparison-protocol".to_owned(),
            currency_code: "ZAR".to_owned(),
            accounting_boundary_id: "sha256:cost-boundary".to_owned(),
            evidence_class: "SYNTHETIC_SCENARIO".to_owned(),
            evidence_bundle_digest: "sha256:synthetic-evidence-bundle".to_owned(),
            uncertainty_method_digest: "sha256:synthetic-interval-method".to_owned(),
            cost_per_good_unit_upper: cost,
            energy_per_good_unit_upper_joules: energy,
            cycle_time_per_good_unit_upper_seconds: cycle,
            final_good_yield_lower: yield_lower,
            optical_floors_passed: true,
            reliability_floors_passed: true,
            minimum_yield_floor_passed: true,
            evidence_complete: true,
        }
    }

    #[test]
    fn route_pareto_comparison_requires_compatible_target_protocol_and_evidence() {
        let a = route_candidate("A", 10.0, 20.0, 100.0, 0.90);
        let mut b = route_candidate("B", 11.0, 21.0, 110.0, 0.85);
        assert_eq!(route_pareto_dominates(&a, &b), Ok(true));

        b.comparison_protocol_digest = "sha256:different-protocol".to_owned();
        assert_eq!(
            route_pareto_dominates(&a, &b),
            Err(ManufacturingMetricError::MismatchedRouteComparison)
        );
        b.comparison_protocol_digest = a.comparison_protocol_digest.clone();
        b.currency_code = "USD".to_owned();
        assert_eq!(
            route_pareto_dominates(&a, &b),
            Err(ManufacturingMetricError::MismatchedRouteComparison)
        );
        b.currency_code = a.currency_code.clone();
        b.accounting_boundary_id = "sha256:different-boundary".to_owned();
        assert_eq!(
            route_pareto_dominates(&a, &b),
            Err(ManufacturingMetricError::MismatchedRouteComparison)
        );
    }

    #[test]
    fn route_tradeoffs_remain_incomparable_without_a_weighted_score() {
        let lower_cost = route_candidate("lower-cost", 8.0, 30.0, 105.0, 0.92);
        let lower_energy = route_candidate("lower-energy", 10.0, 20.0, 95.0, 0.90);
        assert_eq!(
            route_pareto_dominates(&lower_cost, &lower_energy),
            Ok(false)
        );
        assert_eq!(
            route_pareto_dominates(&lower_energy, &lower_cost),
            Ok(false)
        );
    }

    #[test]
    fn near_perfect_transfer_and_repair_capable_routes_remain_pareto_incomparable() {
        // Synthetic values only: route A spends more to reduce transfer defects;
        // route B accepts more first-pass defects and recovers good units via repair.
        // Neither dominates because their cost/energy/cycle/yield trade-offs differ.
        let near_perfect_transfer =
            route_candidate("near-perfect-transfer", 120.0, 40.0, 60.0, 0.9999);
        let repair_capable = route_candidate("repair-capable", 90.0, 50.0, 80.0, 0.9985);
        assert_eq!(
            route_pareto_dominates(&near_perfect_transfer, &repair_capable),
            Ok(false)
        );
        assert_eq!(
            route_pareto_dominates(&repair_capable, &near_perfect_transfer),
            Ok(false)
        );

        let integrated_inspection =
            route_candidate("integrated-inspection", 88.0, 39.0, 58.0, 0.9999);
        assert_eq!(
            route_pareto_dominates(&integrated_inspection, &near_perfect_transfer),
            Ok(true)
        );
        assert_eq!(
            route_pareto_dominates(&integrated_inspection, &repair_capable),
            Ok(true)
        );
    }

    #[test]
    fn route_pareto_refuses_quality_floor_failures_and_invalid_bounds() {
        let a = route_candidate("A", 10.0, 20.0, 100.0, 0.90);
        let mut failed_floor = route_candidate("failed", 1.0, 1.0, 1.0, 1.0);
        failed_floor.optical_floors_passed = false;
        assert_eq!(
            route_pareto_dominates(&failed_floor, &a),
            Err(ManufacturingMetricError::UnqualifiedRouteCandidate)
        );

        let mut invalid = route_candidate("invalid", 10.0, 20.0, 100.0, 0.90);
        invalid.final_good_yield_lower = f64::NAN;
        assert_eq!(
            route_pareto_dominates(&invalid, &a),
            Err(ManufacturingMetricError::InvalidRouteCandidate)
        );

        let mut missing_uncertainty =
            route_candidate("missing-uncertainty", 10.0, 20.0, 100.0, 0.90);
        missing_uncertainty.uncertainty_method_digest.clear();
        assert_eq!(
            route_pareto_dominates(&missing_uncertainty, &a),
            Err(ManufacturingMetricError::InvalidRouteCandidate)
        );
    }

    #[test]
    fn inspection_count_overflow_is_rejected_without_wrapping() {
        assert_eq!(
            summarize_inspection(InspectionCounts {
                eligible_units: u64::MAX,
                inspected_units: u64::MAX,
                true_positive: u64::MAX,
                false_positive: 1,
                false_negative: 0,
            }),
            Err(ManufacturingMetricError::InspectionCountsExceedPopulation)
        );
    }

    #[test]
    fn inspection_summary_rejects_impossible_population_counts() {
        assert_eq!(
            summarize_inspection(InspectionCounts {
                eligible_units: 10,
                inspected_units: 11,
                true_positive: 3,
                false_positive: 1,
                false_negative: 2,
            }),
            Err(ManufacturingMetricError::InspectedExceedsEligible)
        );
        assert_eq!(
            summarize_inspection(InspectionCounts {
                eligible_units: 10,
                inspected_units: 4,
                true_positive: 3,
                false_positive: 2,
                false_negative: 2,
            }),
            Err(ManufacturingMetricError::InspectionCountsExceedPopulation)
        );
    }
}
