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
        close(result.cost_per_calibrated_good_unit.expect("positive final-good count"), 25.0, 1e-12);
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
            CostLedger { materials_consumables: -1.0, ..CostLedger::default() }.total_cost(),
            Err(ManufacturingMetricError::NegativeCost)
        );
        assert_eq!(
            CostLedger { process_labor: f64::NAN, ..CostLedger::default() }.total_cost(),
            Err(ManufacturingMetricError::NonFiniteCost)
        );
        assert_eq!(
            CostLedger {
                materials_consumables: f64::MAX,
                process_labor: f64::MAX,
                ..CostLedger::default()
            }.total_cost(),
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
        }).expect("valid inspection counts");

        close(metrics.coverage, 0.8, 1e-12);
        close(metrics.precision.expect("positive calls"), 0.9, 1e-12);
        close(metrics.recall.expect("represented defects"), 0.75, 1e-12);
        close(metrics.false_negative_rate.expect("represented defects"), 0.25, 1e-12);
    }

    #[test]
    fn undefined_inspection_denominators_are_not_smoothed() {
        let metrics = summarize_inspection(InspectionCounts {
            eligible_units: 50,
            inspected_units: 40,
            true_positive: 0,
            false_positive: 0,
            false_negative: 0,
        }).expect("valid no-defect/no-call sample");
        assert_eq!(metrics.precision, None);
        assert_eq!(metrics.recall, None);
        assert_eq!(metrics.false_negative_rate, None);
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
