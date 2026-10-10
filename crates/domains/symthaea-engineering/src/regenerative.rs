// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Requirements-gated, multi-objective screening for regenerative soil-process designs.
//!
//! This module consumes deterministic process accounting from symthaea-agribot. It does
//! not predict soil efficacy, certify a product, or approve a fertilizer dose. Candidates
//! must carry provenance, satisfy explicit requirements, and pass a separately evidenced
//! product-quality gate before entering the Pareto frontier.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use symthaea_agribot::soil_process::{
    assess_biochar_climate, BiocharClimateAssessment, BiocharClimateInput, EvidenceRef,
    PyrolysisBatchResult,
};

const RELATIVE_EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesignError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl DesignError {
    fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl Error for DesignError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityGateStatus {
    Pass,
    Fail,
    Unknown,
}

/// A quality-gate assertion tied to a named test, standard, or review record.
/// Pass is evidence for this declared gate only—not blanket certification or approval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductQualityGate {
    pub status: QualityGateStatus,
    pub evidence_id: String,
    pub method_or_standard_id: String,
}

impl ProductQualityGate {
    fn validate(&self) -> Result<(), DesignError> {
        if self.evidence_id.trim().is_empty() {
            return Err(DesignError::new(
                "quality_gate.evidence_id",
                "quality gate requires an evidence ID, including when status is unknown",
            ));
        }
        if self.method_or_standard_id.trim().is_empty() {
            return Err(DesignError::new(
                "quality_gate.method_or_standard_id",
                "quality gate requires a method or standard identifier",
            ));
        }
        Ok(())
    }
}

/// A process candidate. Cost and water metrics are normalized per kg of dry feedstock.
/// Candidates compared together must use the same cost_unit, including currency and
/// price basis. Unit strings are identifiers, not an automatic currency converter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeDesignCandidate {
    pub candidate_id: String,
    pub process: PyrolysisBatchResult,
    pub cost_per_kg_dry_feedstock: f64,
    pub cost_unit: String,
    pub cost_evidence: EvidenceRef,
    pub water_l_per_kg_dry_feedstock: f64,
    pub water_evidence: EvidenceRef,
    pub quality_gate: ProductQualityGate,
    /// Optional, explicit lifecycle inventory. Required when climate is an objective or hard constraint.
    pub climate_input: Option<BiocharClimateInput>,
}

/// Requirements are scenario- or stakeholder-defined hard constraints, not universal
/// agronomic constants. Their source and version must be preserved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeDesignRequirements {
    pub specification_id: String,
    pub cost_unit: String,
    pub min_char_yield_fraction: f64,
    pub min_carbon_retained_fraction: f64,
    pub max_supplied_heat_mj_per_kg_dry_feedstock: f64,
    pub max_cost_per_kg_dry_feedstock: f64,
    pub max_water_l_per_kg_dry_feedstock: f64,
    /// Include complete net climate impact as a Pareto objective. Unknown/incomplete climate inputs become indeterminate.
    pub include_climate_objective: bool,
    /// Optional hard limit on net kg CO2e per kg dry feedstock; may be negative.
    pub max_net_climate_kg_co2e_per_kg_dry_feedstock: Option<f64>,
}

impl RegenerativeDesignRequirements {
    fn validate(&self) -> Result<(), DesignError> {
        if self.specification_id.trim().is_empty() {
            return Err(DesignError::new(
                "requirements.specification_id",
                "requirements need a versioned specification ID",
            ));
        }
        if self.cost_unit.trim().is_empty() {
            return Err(DesignError::new(
                "requirements.cost_unit",
                "requirements need an explicit cost unit and price basis",
            ));
        }
        validate_fraction(self.min_char_yield_fraction, "min_char_yield_fraction")?;
        validate_fraction(
            self.min_carbon_retained_fraction,
            "min_carbon_retained_fraction",
        )?;
        validate_nonnegative(
            self.max_supplied_heat_mj_per_kg_dry_feedstock,
            "max_supplied_heat_mj_per_kg_dry_feedstock",
        )?;
        validate_nonnegative(
            self.max_cost_per_kg_dry_feedstock,
            "max_cost_per_kg_dry_feedstock",
        )?;
        validate_nonnegative(
            self.max_water_l_per_kg_dry_feedstock,
            "max_water_l_per_kg_dry_feedstock",
        )?;
        if let Some(limit) = self.max_net_climate_kg_co2e_per_kg_dry_feedstock {
            if !limit.is_finite() {
                return Err(DesignError::new(
                    "max_net_climate_kg_co2e_per_kg_dry_feedstock",
                    "must be finite",
                ));
            }
            if !self.include_climate_objective {
                return Err(DesignError::new(
                    "include_climate_objective",
                    "a climate limit requires complete climate accounting to be enabled",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeDesignMetrics {
    pub char_yield_fraction_dry_basis: f64,
    pub carbon_retained_fraction: f64,
    pub supplied_heat_mj_per_kg_dry_feedstock: f64,
    pub cost_per_kg_dry_feedstock: f64,
    pub water_l_per_kg_dry_feedstock: f64,
    /// Net kg CO2e per kg dry feedstock; None means lifecycle data is absent/incomplete.
    pub net_climate_kg_co2e_per_kg_dry_feedstock: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateEligibility {
    /// Numeric constraints pass and the product-quality gate reports Pass. This is not
    /// field efficacy evidence or authorization to apply material to soil.
    EligibleForDesignComparison,
    /// At least one numeric requirement failed or the quality gate reports Fail.
    Ineligible,
    /// No known numeric failure was found, but a mandatory gate remains unknown.
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeDesignAssessment {
    /// Full immutable candidate snapshot, including all evidence references.
    pub candidate: RegenerativeDesignCandidate,
    pub candidate_id: String,
    pub eligibility: CandidateEligibility,
    pub metrics: RegenerativeDesignMetrics,
    pub failed_constraints: Vec<String>,
    pub quality_gate_status: QualityGateStatus,
    pub climate_assessment: Option<BiocharClimateAssessment>,
    /// Always true: design screening is not proof of field efficacy.
    pub field_validation_required: bool,
    pub scope_note: String,
}

fn validate_nonnegative(value: f64, field: &'static str) -> Result<(), DesignError> {
    if !value.is_finite() {
        return Err(DesignError::new(field, "must be finite"));
    }
    if value < 0.0 {
        return Err(DesignError::new(field, "cannot be negative"));
    }
    Ok(())
}

fn validate_fraction(value: f64, field: &'static str) -> Result<(), DesignError> {
    validate_nonnegative(value, field)?;
    if value > 1.0 {
        return Err(DesignError::new(field, "must be in the closed interval [0, 1]"));
    }
    Ok(())
}

fn approximately_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= tolerance(a, b)
}

fn validate_evidence(evidence: &EvidenceRef, field: &'static str) -> Result<(), DesignError> {
    if evidence.evidence_id.trim().is_empty() {
        return Err(DesignError::new(field, "requires a non-empty evidence ID"));
    }
    Ok(())
}

fn calculate_metrics(
    candidate: &RegenerativeDesignCandidate,
    requirements: &RegenerativeDesignRequirements,
    climate_assessment: Option<&BiocharClimateAssessment>,
) -> Result<RegenerativeDesignMetrics, DesignError> {
    if candidate.candidate_id.trim().is_empty() {
        return Err(DesignError::new("candidate_id", "cannot be empty"));
    }
    if candidate.cost_unit.trim().is_empty() || candidate.cost_unit != requirements.cost_unit {
        return Err(DesignError::new(
            "cost_unit",
            "candidate and requirements must use the identical normalized cost unit",
        ));
    }
    validate_evidence(&candidate.cost_evidence, "cost_evidence")?;
    validate_evidence(&candidate.water_evidence, "water_evidence")?;
    candidate.quality_gate.validate()?;

    let process = &candidate.process;
    validate_nonnegative(process.dry_feedstock_kg, "process.dry_feedstock_kg")?;
    if process.dry_feedstock_kg <= 0.0 {
        return Err(DesignError::new(
            "process.dry_feedstock_kg",
            "must be greater than zero to calculate per-kg metrics",
        ));
    }
    validate_nonnegative(process.char_product_kg, "process.char_product_kg")?;
    if process.char_product_kg > process.dry_feedstock_kg {
        return Err(DesignError::new(
            "process.char_product_kg",
            "cannot exceed dry feedstock mass",
        ));
    }
    validate_nonnegative(
        process.non_char_dry_products_residual_kg,
        "process.non_char_dry_products_residual_kg",
    )?;
    if !approximately_equal(
        process.char_product_kg + process.non_char_dry_products_residual_kg,
        process.dry_feedstock_kg,
    ) {
        return Err(DesignError::new(
            "process.mass_balance",
            "char plus residual dry products must equal dry feedstock",
        ));
    }

    validate_nonnegative(process.feedstock_carbon_kg, "process.feedstock_carbon_kg")?;
    validate_nonnegative(process.char_carbon_kg, "process.char_carbon_kg")?;
    validate_nonnegative(
        process.carbon_not_in_char_kg,
        "process.carbon_not_in_char_kg",
    )?;
    validate_fraction(
        process.carbon_retained_in_char_fraction,
        "process.carbon_retained_in_char_fraction",
    )?;
    if process.char_carbon_kg > process.char_product_kg
        + tolerance(process.char_carbon_kg, process.char_product_kg)
    {
        return Err(DesignError::new(
            "process.char_carbon_mass_fraction",
            "elemental carbon mass cannot exceed total char product mass",
        ));
    }
    if process.char_carbon_kg > process.feedstock_carbon_kg
        || !approximately_equal(
            process.char_carbon_kg + process.carbon_not_in_char_kg,
            process.feedstock_carbon_kg,
        )
    {
        return Err(DesignError::new(
            "process.carbon_balance",
            "carbon streams must close and char carbon cannot exceed feedstock carbon",
        ));
    }
    let expected_carbon_retention = if process.feedstock_carbon_kg == 0.0 {
        0.0
    } else {
        process.char_carbon_kg / process.feedstock_carbon_kg
    };
    if !approximately_equal(
        process.carbon_retained_in_char_fraction,
        expected_carbon_retention,
    ) {
        return Err(DesignError::new(
            "process.carbon_retained_in_char_fraction",
            "retention ratio is inconsistent with elemental carbon masses",
        ));
    }

    validate_nonnegative(
        process.dry_feedstock_sensible_heat_mj,
        "process.dry_feedstock_sensible_heat_mj",
    )?;
    validate_nonnegative(
        process.water_heating_and_vaporization_heat_mj,
        "process.water_heating_and_vaporization_heat_mj",
    )?;
    validate_nonnegative(
        process.reactor_sensible_heat_mj,
        "process.reactor_sensible_heat_mj",
    )?;
    validate_nonnegative(
        process.declared_terms_heat_duty_mj,
        "process.declared_terms_heat_duty_mj",
    )?;
    validate_nonnegative(
        process.estimated_supplied_heat_mj,
        "process.estimated_supplied_heat_mj",
    )?;
    let recomputed_duty = process.dry_feedstock_sensible_heat_mj
        + process.water_heating_and_vaporization_heat_mj
        + process.reactor_sensible_heat_mj;
    if !approximately_equal(process.declared_terms_heat_duty_mj, recomputed_duty)
        || process.estimated_supplied_heat_mj + tolerance(
            process.estimated_supplied_heat_mj,
            process.declared_terms_heat_duty_mj,
        ) < process.declared_terms_heat_duty_mj
    {
        return Err(DesignError::new(
            "process.heat_balance",
            "declared heat duty must match component duties and supplied heat cannot be lower",
        ));
    }

    for (field, id) in [
        (
            "process.evidence.feedstock",
            process.evidence.feedstock.evidence_id.as_str(),
        ),
        (
            "process.evidence.process_parameters",
            process.evidence.process_parameters.evidence_id.as_str(),
        ),
        (
            "process.evidence.thermophysical_properties",
            process.evidence.thermophysical_properties.evidence_id.as_str(),
        ),
        (
            "process.evidence.reactor_design",
            process.evidence.reactor_design.evidence_id.as_str(),
        ),
    ] {
        if id.trim().is_empty() {
            return Err(DesignError::new(field, "requires a non-empty evidence ID"));
        }
    }
    if process.evidence.input_snapshot_id.trim().is_empty() {
        return Err(DesignError::new(
            "process.evidence.input_snapshot_id",
            "requires a non-empty input snapshot ID",
        ));
    }
    validate_nonnegative(
        candidate.cost_per_kg_dry_feedstock,
        "cost_per_kg_dry_feedstock",
    )?;
    validate_nonnegative(
        candidate.water_l_per_kg_dry_feedstock,
        "water_l_per_kg_dry_feedstock",
    )?;

    let result = RegenerativeDesignMetrics {
        char_yield_fraction_dry_basis: process.char_product_kg / process.dry_feedstock_kg,
        carbon_retained_fraction: process.carbon_retained_in_char_fraction,
        supplied_heat_mj_per_kg_dry_feedstock: process.estimated_supplied_heat_mj
            / process.dry_feedstock_kg,
        cost_per_kg_dry_feedstock: candidate.cost_per_kg_dry_feedstock,
        water_l_per_kg_dry_feedstock: candidate.water_l_per_kg_dry_feedstock,
        net_climate_kg_co2e_per_kg_dry_feedstock: climate_assessment
            .and_then(|assessment| assessment.net_kg_co2e)
            .map(|net| net / process.dry_feedstock_kg),
    };
    for (field, value) in [
        (
            "metrics.char_yield_fraction_dry_basis",
            result.char_yield_fraction_dry_basis,
        ),
        ("metrics.carbon_retained_fraction", result.carbon_retained_fraction),
        (
            "metrics.supplied_heat_mj_per_kg_dry_feedstock",
            result.supplied_heat_mj_per_kg_dry_feedstock,
        ),
        ("metrics.cost_per_kg_dry_feedstock", result.cost_per_kg_dry_feedstock),
        ("metrics.water_l_per_kg_dry_feedstock", result.water_l_per_kg_dry_feedstock),
    ] {
        validate_nonnegative(value, field)?;
    }
    if result
        .net_climate_kg_co2e_per_kg_dry_feedstock
        .is_some_and(|value| !value.is_finite())
    {
        return Err(DesignError::new(
            "metrics.net_climate_kg_co2e_per_kg_dry_feedstock",
            "must be finite",
        ));
    }
    Ok(result)
}

/// Evaluate a candidate against explicit hard constraints, with no weighted composite
/// score. A scenario can pass this numerical screen but remains scenario evidence.
pub fn assess_regenerative_candidate(
    candidate: &RegenerativeDesignCandidate,
    requirements: &RegenerativeDesignRequirements,
) -> Result<RegenerativeDesignAssessment, DesignError> {
    requirements.validate()?;
    let climate_assessment = candidate
        .climate_input
        .as_ref()
        .map(|input| {
            assess_biochar_climate(&candidate.process, input)
                .map_err(|error| DesignError::new(error.field, error.reason))
        })
        .transpose()?;
    let metrics = calculate_metrics(candidate, requirements, climate_assessment.as_ref())?;
    let mut failed_constraints = Vec::new();

    if metrics.char_yield_fraction_dry_basis < requirements.min_char_yield_fraction {
        failed_constraints.push("char_yield_below_minimum".to_string());
    }
    if metrics.carbon_retained_fraction < requirements.min_carbon_retained_fraction {
        failed_constraints.push("carbon_retention_below_minimum".to_string());
    }
    if metrics.supplied_heat_mj_per_kg_dry_feedstock
        > requirements.max_supplied_heat_mj_per_kg_dry_feedstock
    {
        failed_constraints.push("supplied_heat_above_maximum".to_string());
    }
    if metrics.cost_per_kg_dry_feedstock > requirements.max_cost_per_kg_dry_feedstock {
        failed_constraints.push("cost_above_maximum".to_string());
    }
    if metrics.water_l_per_kg_dry_feedstock > requirements.max_water_l_per_kg_dry_feedstock {
        failed_constraints.push("water_use_above_maximum".to_string());
    }
    if candidate.quality_gate.status == QualityGateStatus::Fail {
        failed_constraints.push("product_quality_gate_failed".to_string());
    }
    if let (Some(limit), Some(net)) = (
        requirements.max_net_climate_kg_co2e_per_kg_dry_feedstock,
        metrics.net_climate_kg_co2e_per_kg_dry_feedstock,
    ) {
        if net > limit {
            failed_constraints.push("net_climate_above_maximum".to_string());
        }
    }

    let climate_indeterminate = requirements.include_climate_objective
        && metrics.net_climate_kg_co2e_per_kg_dry_feedstock.is_none();
    let eligibility = if !failed_constraints.is_empty() {
        CandidateEligibility::Ineligible
    } else if candidate.quality_gate.status == QualityGateStatus::Unknown || climate_indeterminate {
        CandidateEligibility::Indeterminate
    } else {
        CandidateEligibility::EligibleForDesignComparison
    };

    Ok(RegenerativeDesignAssessment {
        candidate: candidate.clone(),
        candidate_id: candidate.candidate_id.clone(),
        eligibility,
        metrics,
        failed_constraints,
        quality_gate_status: candidate.quality_gate.status,
        climate_assessment,
        field_validation_required: true,
        scope_note: "design-screen result only; not an amendment certification or field recommendation".into(),
    })
}

fn tolerance(a: f64, b: f64) -> f64 {
    RELATIVE_EPSILON * a.abs().max(b.abs()).max(1.0)
}

fn dominates(
    a: RegenerativeDesignMetrics,
    b: RegenerativeDesignMetrics,
    include_climate: bool,
) -> bool {
    // Maximize char yield and carbon retention; minimize heat, cost and water.
    let no_worse = a.char_yield_fraction_dry_basis
        + tolerance(a.char_yield_fraction_dry_basis, b.char_yield_fraction_dry_basis)
        >= b.char_yield_fraction_dry_basis
        && a.carbon_retained_fraction
            + tolerance(a.carbon_retained_fraction, b.carbon_retained_fraction)
            >= b.carbon_retained_fraction
        && a.supplied_heat_mj_per_kg_dry_feedstock
            <= b.supplied_heat_mj_per_kg_dry_feedstock
                + tolerance(
                    a.supplied_heat_mj_per_kg_dry_feedstock,
                    b.supplied_heat_mj_per_kg_dry_feedstock,
                )
        && a.cost_per_kg_dry_feedstock
            <= b.cost_per_kg_dry_feedstock
                + tolerance(
                    a.cost_per_kg_dry_feedstock,
                    b.cost_per_kg_dry_feedstock,
                )
        && a.water_l_per_kg_dry_feedstock
            <= b.water_l_per_kg_dry_feedstock
                + tolerance(
                    a.water_l_per_kg_dry_feedstock,
                    b.water_l_per_kg_dry_feedstock,
                );

    let strictly_better = a.char_yield_fraction_dry_basis
        > b.char_yield_fraction_dry_basis
            + tolerance(a.char_yield_fraction_dry_basis, b.char_yield_fraction_dry_basis)
        || a.carbon_retained_fraction
            > b.carbon_retained_fraction
                + tolerance(a.carbon_retained_fraction, b.carbon_retained_fraction)
        || a.supplied_heat_mj_per_kg_dry_feedstock
            + tolerance(
                a.supplied_heat_mj_per_kg_dry_feedstock,
                b.supplied_heat_mj_per_kg_dry_feedstock,
            )
            < b.supplied_heat_mj_per_kg_dry_feedstock
        || a.cost_per_kg_dry_feedstock
            + tolerance(a.cost_per_kg_dry_feedstock, b.cost_per_kg_dry_feedstock)
            < b.cost_per_kg_dry_feedstock
        || a.water_l_per_kg_dry_feedstock
            + tolerance(a.water_l_per_kg_dry_feedstock, b.water_l_per_kg_dry_feedstock)
            < b.water_l_per_kg_dry_feedstock;

    if include_climate {
        let (Some(a_climate), Some(b_climate)) = (
            a.net_climate_kg_co2e_per_kg_dry_feedstock,
            b.net_climate_kg_co2e_per_kg_dry_feedstock,
        ) else {
            return false;
        };
        let climate_no_worse = a_climate <= b_climate + tolerance(a_climate, b_climate);
        let climate_strictly_better =
            a_climate + tolerance(a_climate, b_climate) < b_climate;
        no_worse && climate_no_worse && (strictly_better || climate_strictly_better)
    } else {
        no_worse && strictly_better
    }
}

/// Return the sorted non-dominated set among candidates passing all numeric constraints
/// and with a known passing quality gate. This is a Pareto set, not a total ranking.
pub fn regenerative_pareto_frontier(
    candidates: &[RegenerativeDesignCandidate],
    requirements: &RegenerativeDesignRequirements,
) -> Result<Vec<String>, DesignError> {
    requirements.validate()?;
    let mut ids = std::collections::HashSet::new();
    let mut eligible = Vec::new();

    for candidate in candidates {
        if !ids.insert(candidate.candidate_id.as_str()) {
            return Err(DesignError::new(
                "candidate_id",
                "candidate IDs must be unique",
            ));
        }
        let assessment = assess_regenerative_candidate(candidate, requirements)?;
        if assessment.eligibility == CandidateEligibility::EligibleForDesignComparison {
            eligible.push((candidate.candidate_id.clone(), assessment.metrics));
        }
    }

    let mut frontier = Vec::new();
    for (i, (candidate_id, candidate_metrics)) in eligible.iter().enumerate() {
        let is_dominated = eligible.iter().enumerate().any(|(j, (_, other_metrics))| {
            i != j && dominates(*other_metrics, *candidate_metrics, requirements.include_climate_objective)
        });
        if !is_dominated {
            frontier.push(candidate_id.clone());
        }
    }
    frontier.sort();
    Ok(frontier)
}



/// Closed interval for an empirical/modelled metric, with provenance for the bounds.
/// Bounds are not assumed to be a confidence interval: callers must document how they
/// were obtained (e.g. instrument uncertainty, a calibrated prediction interval, or a
/// deliberately conservative scenario envelope) in the referenced evidence record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricInterval {
    pub lower: f64,
    pub upper: f64,
    pub evidence: EvidenceRef,
}

impl MetricInterval {
    fn validate(&self, field: &'static str, fraction: bool) -> Result<(), DesignError> {
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Err(DesignError::new(field, "interval bounds must be finite"));
        }
        if self.lower > self.upper {
            return Err(DesignError::new(field, "lower bound cannot exceed upper bound"));
        }
        if self.evidence.evidence_id.trim().is_empty() {
            return Err(DesignError::new(field, "interval bounds require an evidence ID"));
        }
        if self.lower < 0.0 {
            return Err(DesignError::new(field, "this metric interval cannot be negative"));
        }
        if fraction && self.upper > 1.0 {
            return Err(DesignError::new(field, "fraction bounds must lie in [0, 1]"));
        }
        Ok(())
    }

    fn validate_signed(&self, field: &'static str) -> Result<(), DesignError> {
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Err(DesignError::new(field, "interval bounds must be finite"));
        }
        if self.lower > self.upper {
            return Err(DesignError::new(field, "lower bound cannot exceed upper bound"));
        }
        if self.evidence.evidence_id.trim().is_empty() {
            return Err(DesignError::new(field, "interval bounds require an evidence ID"));
        }
        Ok(())
    }
}

/// Interval inputs for robust feasibility screening. Metric units are fixed by each
/// field name and match RegenerativeDesignMetrics. Climate values may be signed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeMetricIntervals {
    pub candidate_id: String,
    pub char_yield_fraction_dry_basis: MetricInterval,
    pub carbon_retained_fraction: MetricInterval,
    pub supplied_heat_mj_per_kg_dry_feedstock: MetricInterval,
    pub cost_per_kg_dry_feedstock: MetricInterval,
    pub water_l_per_kg_dry_feedstock: MetricInterval,
    pub net_climate_kg_co2e_per_kg_dry_feedstock: Option<MetricInterval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalConstraintStatus {
    /// Every value in the stated interval satisfies this threshold.
    RobustPass,
    /// Every value in the stated interval violates this threshold.
    RobustFail,
    /// The interval crosses the threshold, so the decision is unresolved.
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintDirection {
    AtLeast,
    AtMost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustFeasibilityStatus {
    RobustlyFeasible,
    RobustlyInfeasible,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalConstraintResult {
    pub constraint_id: String,
    pub status: IntervalConstraintStatus,
    pub direction: Option<ConstraintDirection>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub threshold: Option<f64>,
    pub evidence: Option<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeUncertaintyAssessment {
    pub candidate_id: String,
    pub specification_id: String,
    pub status: RobustFeasibilityStatus,
    pub constraints: Vec<IntervalConstraintResult>,
    /// Robust feasibility is still model screening, not agronomic validation.
    pub field_validation_required: bool,
    pub scope_note: String,
}

fn classify_interval(
    interval: &MetricInterval,
    threshold: f64,
    direction: ConstraintDirection,
) -> IntervalConstraintStatus {
    match direction {
        ConstraintDirection::AtLeast if interval.lower >= threshold => {
            IntervalConstraintStatus::RobustPass
        }
        ConstraintDirection::AtLeast if interval.upper < threshold => {
            IntervalConstraintStatus::RobustFail
        }
        ConstraintDirection::AtMost if interval.upper <= threshold => {
            IntervalConstraintStatus::RobustPass
        }
        ConstraintDirection::AtMost if interval.lower > threshold => {
            IntervalConstraintStatus::RobustFail
        }
        _ => IntervalConstraintStatus::Unresolved,
    }
}

fn interval_constraint(
    id: &'static str,
    interval: &MetricInterval,
    threshold: f64,
    direction: ConstraintDirection,
) -> IntervalConstraintResult {
    IntervalConstraintResult {
        constraint_id: id.to_string(),
        status: classify_interval(interval, threshold, direction),
        direction: Some(direction),
        lower: Some(interval.lower),
        upper: Some(interval.upper),
        threshold: Some(threshold),
        evidence: Some(interval.evidence.clone()),
    }
}

/// Evaluate hard design requirements against uncertainty intervals rather than only
/// point estimates. The result is conservative: an interval crossing a limit is
/// indeterminate, not a pass. Interval semantics are deliberately independent of the
/// existing point-estimate assessment and do not silently rewrite its outputs.
pub fn assess_regenerative_uncertainty(
    intervals: &RegenerativeMetricIntervals,
    requirements: &RegenerativeDesignRequirements,
) -> Result<RegenerativeUncertaintyAssessment, DesignError> {
    requirements.validate()?;
    if intervals.candidate_id.trim().is_empty() {
        return Err(DesignError::new("candidate_id", "cannot be empty"));
    }
    intervals.char_yield_fraction_dry_basis
        .validate("intervals.char_yield_fraction_dry_basis", true)?;
    intervals.carbon_retained_fraction
        .validate("intervals.carbon_retained_fraction", true)?;
    intervals.supplied_heat_mj_per_kg_dry_feedstock
        .validate("intervals.supplied_heat_mj_per_kg_dry_feedstock", false)?;
    intervals.cost_per_kg_dry_feedstock
        .validate("intervals.cost_per_kg_dry_feedstock", false)?;
    intervals.water_l_per_kg_dry_feedstock
        .validate("intervals.water_l_per_kg_dry_feedstock", false)?;
    if let Some(climate) = &intervals.net_climate_kg_co2e_per_kg_dry_feedstock {
        climate.validate_signed("intervals.net_climate_kg_co2e_per_kg_dry_feedstock")?;
    }

    let mut constraints = vec![
        interval_constraint(
            "char_yield_minimum",
            &intervals.char_yield_fraction_dry_basis,
            requirements.min_char_yield_fraction,
            ConstraintDirection::AtLeast,
        ),
        interval_constraint(
            "carbon_retention_minimum",
            &intervals.carbon_retained_fraction,
            requirements.min_carbon_retained_fraction,
            ConstraintDirection::AtLeast,
        ),
        interval_constraint(
            "supplied_heat_maximum",
            &intervals.supplied_heat_mj_per_kg_dry_feedstock,
            requirements.max_supplied_heat_mj_per_kg_dry_feedstock,
            ConstraintDirection::AtMost,
        ),
        interval_constraint(
            "cost_maximum",
            &intervals.cost_per_kg_dry_feedstock,
            requirements.max_cost_per_kg_dry_feedstock,
            ConstraintDirection::AtMost,
        ),
        interval_constraint(
            "water_use_maximum",
            &intervals.water_l_per_kg_dry_feedstock,
            requirements.max_water_l_per_kg_dry_feedstock,
            ConstraintDirection::AtMost,
        ),
    ];

    if requirements.include_climate_objective {
        match (
            intervals.net_climate_kg_co2e_per_kg_dry_feedstock.as_ref(),
            requirements.max_net_climate_kg_co2e_per_kg_dry_feedstock,
        ) {
            (Some(interval), Some(limit)) => constraints.push(interval_constraint(
                "net_climate_maximum",
                interval,
                limit,
                ConstraintDirection::AtMost,
            )),
            (Some(_), None) => {}
            (None, _) => constraints.push(IntervalConstraintResult {
                constraint_id: "climate_objective_interval_available".to_string(),
                status: IntervalConstraintStatus::Unresolved,
                direction: None,
                lower: None,
                upper: None,
                threshold: None,
                evidence: None,
            }),
        }
    }

    let status = if constraints
        .iter()
        .any(|c| c.status == IntervalConstraintStatus::RobustFail)
    {
        RobustFeasibilityStatus::RobustlyInfeasible
    } else if constraints
        .iter()
        .any(|c| c.status == IntervalConstraintStatus::Unresolved)
    {
        RobustFeasibilityStatus::Indeterminate
    } else {
        RobustFeasibilityStatus::RobustlyFeasible
    };

    Ok(RegenerativeUncertaintyAssessment {
        candidate_id: intervals.candidate_id.clone(),
        specification_id: requirements.specification_id.clone(),
        status,
        constraints,
        field_validation_required: true,
        scope_note: "interval-based design screen only; interval meaning depends on cited evidence and it is not proof of field efficacy or authorization to apply material".into(),
    })
}

#[cfg(test)]
mod uncertainty_tests {
    use super::*;
    use symthaea_agribot::soil_process::EvidenceKind;

    fn interval(lower: f64, upper: f64, id: &str) -> MetricInterval {
        MetricInterval {
            lower,
            upper,
            evidence: EvidenceRef {
                evidence_id: id.into(),
                kind: EvidenceKind::Scenario,
            },
        }
    }

    fn intervals() -> RegenerativeMetricIntervals {
        RegenerativeMetricIntervals {
            candidate_id: "candidate-interval-v1".into(),
            char_yield_fraction_dry_basis: interval(0.28, 0.32, "yield-range-v1"),
            carbon_retained_fraction: interval(0.55, 0.64, "carbon-range-v1"),
            supplied_heat_mj_per_kg_dry_feedstock: interval(3.0, 5.0, "heat-range-v1"),
            cost_per_kg_dry_feedstock: interval(2.5, 4.0, "cost-range-v1"),
            water_l_per_kg_dry_feedstock: interval(1.0, 3.0, "water-range-v1"),
            net_climate_kg_co2e_per_kg_dry_feedstock: None,
        }
    }

    #[test]
    fn interval_screen_requires_the_entire_range_to_pass() {
        let req = requirements();
        let result = assess_regenerative_uncertainty(&intervals(), &req).unwrap();
        assert_eq!(result.status, RobustFeasibilityStatus::RobustlyFeasible);
        assert!(result.constraints.iter().all(|c| {
            c.status == IntervalConstraintStatus::RobustPass
        }));
        assert!(result.field_validation_required);
    }

    #[test]
    fn threshold_crossing_is_indeterminate_not_a_point_estimate_pass() {
        let req = requirements();
        let mut input = intervals();
        input.char_yield_fraction_dry_basis = interval(0.19, 0.31, "uncertain-yield-v1");
        let result = assess_regenerative_uncertainty(&input, &req).unwrap();
        assert_eq!(result.status, RobustFeasibilityStatus::Indeterminate);
        assert_eq!(
            result.constraints.iter().find(|c| c.constraint_id == "char_yield_minimum").unwrap().status,
            IntervalConstraintStatus::Unresolved
        );
    }

    #[test]
    fn a_fully_violating_interval_is_robustly_infeasible() {
        let req = requirements();
        let mut input = intervals();
        input.supplied_heat_mj_per_kg_dry_feedstock = interval(8.1, 9.0, "heat-fail-v1");
        let result = assess_regenerative_uncertainty(&input, &req).unwrap();
        assert_eq!(result.status, RobustFeasibilityStatus::RobustlyInfeasible);
        assert_eq!(
            result.constraints.iter().find(|c| c.constraint_id == "supplied_heat_maximum").unwrap().status,
            IntervalConstraintStatus::RobustFail
        );
    }

    #[test]
    fn missing_required_climate_interval_is_indeterminate() {
        let mut req = requirements();
        req.include_climate_objective = true;
        let result = assess_regenerative_uncertainty(&intervals(), &req).unwrap();
        assert_eq!(result.status, RobustFeasibilityStatus::Indeterminate);
        assert!(result.constraints.iter().any(|c| {
            c.constraint_id == "climate_objective_interval_available"
                && c.status == IntervalConstraintStatus::Unresolved
        }));
    }

    #[test]
    fn malformed_or_unprovenanced_intervals_are_rejected() {
        let req = requirements();
        let mut input = intervals();
        input.cost_per_kg_dry_feedstock = interval(5.0, 4.0, "reversed-range-v1");
        assert!(assess_regenerative_uncertainty(&input, &req).is_err());

        input = intervals();
        input.water_l_per_kg_dry_feedstock.evidence.evidence_id.clear();
        assert!(assess_regenerative_uncertainty(&input, &req).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_agribot::soil_process::{
        CharStorageAccounting, CharStorageEligibility, ClimateFlow, ClimateFlowKind,
        ClimateInventoryStatus, EvidenceKind, PyrolysisEvidence,
    };

    fn evidence(id: &str) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind: EvidenceKind::Scenario,
        }
    }

    fn process(
        dry_kg: f64,
        char_kg: f64,
        carbon_retained: f64,
        heat_mj: f64,
    ) -> PyrolysisBatchResult {
        PyrolysisBatchResult {
            evidence: PyrolysisEvidence {
                feedstock: evidence("feedstock-scenario"),
                process_parameters: evidence("yield-scenario"),
                thermophysical_properties: evidence("properties-literature"),
                reactor_design: evidence("reactor-scenario"),
                input_snapshot_id: "input-scenario".into(),
            },
            dry_feedstock_kg: dry_kg,
            feed_water_kg: 0.0,
            char_product_kg: char_kg,
            non_char_dry_products_residual_kg: dry_kg - char_kg,
            feedstock_carbon_kg: 48.0,
            char_carbon_kg: 48.0 * carbon_retained,
            carbon_not_in_char_kg: 48.0 * (1.0 - carbon_retained),
            carbon_retained_in_char_fraction: carbon_retained,
            dry_feedstock_sensible_heat_mj: heat_mj,
            water_heating_and_vaporization_heat_mj: 0.0,
            reactor_sensible_heat_mj: 0.0,
            declared_terms_heat_duty_mj: heat_mj,
            estimated_supplied_heat_mj: heat_mj,
        }
    }

    fn candidate(
        id: &str,
        char_yield: f64,
        carbon_retained: f64,
        heat: f64,
        cost: f64,
        water: f64,
        quality: QualityGateStatus,
    ) -> RegenerativeDesignCandidate {
        let dry = 100.0;
        RegenerativeDesignCandidate {
            candidate_id: id.into(),
            process: process(dry, dry * char_yield, carbon_retained, dry * heat),
            cost_per_kg_dry_feedstock: cost,
            cost_unit: "USD_2026_per_kg_dry".into(),
            cost_evidence: evidence(&format!("{id}-cost")),
            water_l_per_kg_dry_feedstock: water,
            water_evidence: evidence(&format!("{id}-water")),
            quality_gate: ProductQualityGate {
                status: quality,
                evidence_id: format!("{id}-quality"),
                method_or_standard_id: "scenario-quality-gate-v1".into(),
            },
            climate_input: None,
        }
    }

    fn climate_input(emissions_kg_co2e: f64) -> BiocharClimateInput {
        BiocharClimateInput {
            boundary_id: "scenario-boundary-v1".into(),
            inventory_status: ClimateInventoryStatus::CompleteForDeclaredBoundary,
            inventory_evidence: evidence("scenario-inventory"),
            flows: vec![ClimateFlow {
                flow_id: "process-emissions".into(),
                kind: ClimateFlowKind::Emission,
                kg_co2e: emissions_kg_co2e,
                evidence: evidence("scenario-process-emissions"),
            }],
            char_storage: CharStorageAccounting {
                horizon_years: 100,
                eligibility: CharStorageEligibility::VerifiedIneligible,
                eligibility_evidence: Some(EvidenceRef {
                    evidence_id: "test-storage-ineligibility".into(),
                    kind: EvidenceKind::Measured,
                }),
                durable_fraction_at_horizon: None,
                persistence_evidence: None,
            },
        }
    }

    fn requirements() -> RegenerativeDesignRequirements {
        RegenerativeDesignRequirements {
            specification_id: "screening-spec-v1".into(),
            cost_unit: "USD_2026_per_kg_dry".into(),
            min_char_yield_fraction: 0.20,
            min_carbon_retained_fraction: 0.40,
            max_supplied_heat_mj_per_kg_dry_feedstock: 8.0,
            max_cost_per_kg_dry_feedstock: 10.0,
            max_water_l_per_kg_dry_feedstock: 8.0,
            include_climate_objective: false,
            max_net_climate_kg_co2e_per_kg_dry_feedstock: None,
        }
    }

    #[test]
    fn candidate_must_pass_numeric_and_quality_gates() {
        let good = candidate("good", 0.30, 0.60, 3.0, 4.0, 2.0, QualityGateStatus::Pass);
        let assessment = assess_regenerative_candidate(&good, &requirements()).unwrap();
        assert_eq!(
            assessment.eligibility,
            CandidateEligibility::EligibleForDesignComparison
        );
        assert!(assessment.field_validation_required);
        assert_eq!(assessment.candidate, good);
        assert_eq!(
            assessment.candidate.process.evidence.process_parameters.kind,
            EvidenceKind::Scenario
        );

        let unknown =
            candidate("unknown", 0.30, 0.60, 3.0, 4.0, 2.0, QualityGateStatus::Unknown);
        assert_eq!(
            assess_regenerative_candidate(&unknown, &requirements())
                .unwrap()
                .eligibility,
            CandidateEligibility::Indeterminate
        );

        let failed =
            candidate("failed", 0.30, 0.60, 3.0, 4.0, 2.0, QualityGateStatus::Fail);
        assert_eq!(
            assess_regenerative_candidate(&failed, &requirements())
                .unwrap()
                .eligibility,
            CandidateEligibility::Ineligible
        );
    }

    #[test]
    fn thresholds_are_hard_constraints_not_weighted_score_terms() {
        let too_hot = candidate("too-hot", 0.30, 0.60, 9.0, 4.0, 2.0, QualityGateStatus::Pass);
        let assessment = assess_regenerative_candidate(&too_hot, &requirements()).unwrap();
        assert_eq!(assessment.eligibility, CandidateEligibility::Ineligible);
        assert!(assessment
            .failed_constraints
            .contains(&"supplied_heat_above_maximum".to_string()));
    }

    #[test]
    fn pareto_frontier_keeps_tradeoffs_and_excludes_dominated_or_unverified() {
        let options = vec![
            candidate("a", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass),
            candidate("b", 0.25, 0.50, 4.0, 5.0, 5.0, QualityGateStatus::Pass),
            candidate("c", 0.35, 0.55, 5.0, 3.0, 3.0, QualityGateStatus::Pass),
            candidate(
                "unknown-quality",
                0.40,
                0.70,
                2.0,
                2.0,
                2.0,
                QualityGateStatus::Unknown,
            ),
            candidate(
                "failed-quality",
                0.40,
                0.70,
                2.0,
                2.0,
                2.0,
                QualityGateStatus::Fail,
            ),
        ];
        assert_eq!(
            regenerative_pareto_frontier(&options, &requirements()).unwrap(),
            vec!["a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn mismatched_cost_units_and_invalid_metrics_are_rejected() {
        let mut option = candidate("a", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        option.cost_unit = "ZAR_2026_per_kg_dry".into();
        assert!(assess_regenerative_candidate(&option, &requirements()).is_err());

        option = candidate("a", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        option.process.dry_feedstock_kg = 0.0;
        assert!(assess_regenerative_candidate(&option, &requirements()).is_err());
    }

    #[test]
    fn evaluator_rejects_internally_inconsistent_process_receipts() {
        let mut option =
            candidate("bad-carbon", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        option.process.char_carbon_kg = 150.0;
        assert!(assess_regenerative_candidate(&option, &requirements()).is_err());

        option = candidate("bad-heat", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        option.process.estimated_supplied_heat_mj = 1.0;
        assert!(assess_regenerative_candidate(&option, &requirements()).is_err());
    }

    #[test]
    fn evaluator_rejects_char_carbon_mass_above_total_char_mass() {
        let mut option =
            candidate("impossible-char", 0.25, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        // 48 kg feedstock C × 0.60 retention = 28.8 kg char C, impossible in 25 kg char.
        option.process.char_carbon_kg = 28.8;
        assert!(assess_regenerative_candidate(&option, &requirements()).is_err());
    }

    #[test]
    fn climate_objective_changes_pareto_frontier_only_with_complete_ledgers() {
        let mut requirements = requirements();
        requirements.include_climate_objective = true;

        let mut higher_emissions =
            candidate("higher-ghg", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        higher_emissions.climate_input = Some(climate_input(100.0));

        let mut lower_emissions =
            candidate("lower-ghg", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        lower_emissions.climate_input = Some(climate_input(40.0));

        let assessment =
            assess_regenerative_candidate(&higher_emissions, &requirements).unwrap();
        assert_eq!(
            assessment.metrics.net_climate_kg_co2e_per_kg_dry_feedstock,
            Some(1.0)
        );
        assert!(assessment.climate_assessment.is_some());

        assert_eq!(
            regenerative_pareto_frontier(
                &[higher_emissions.clone(), lower_emissions.clone()],
                &requirements
            )
            .unwrap(),
            vec!["lower-ghg".to_string()]
        );

        higher_emissions.climate_input = None;
        let incomplete = assess_regenerative_candidate(&higher_emissions, &requirements).unwrap();
        assert_eq!(incomplete.eligibility, CandidateEligibility::Indeterminate);
        assert!(regenerative_pareto_frontier(&[higher_emissions], &requirements)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn net_climate_hard_limit_is_enforced_with_signed_thresholds_supported() {
        let mut requirements = requirements();
        requirements.include_climate_objective = true;
        requirements.max_net_climate_kg_co2e_per_kg_dry_feedstock = Some(0.5);

        let mut option =
            candidate("over-budget", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass);
        option.climate_input = Some(climate_input(100.0));
        let assessment = assess_regenerative_candidate(&option, &requirements).unwrap();
        assert_eq!(assessment.eligibility, CandidateEligibility::Ineligible);
        assert!(assessment
            .failed_constraints
            .contains(&"net_climate_above_maximum".to_string()));

        requirements.max_net_climate_kg_co2e_per_kg_dry_feedstock = Some(-0.1);
        let negative_limit = assess_regenerative_candidate(&option, &requirements).unwrap();
        assert!(negative_limit
            .failed_constraints
            .contains(&"net_climate_above_maximum".to_string()));
    }

    #[test]
    fn candidate_ids_must_be_unique_for_deterministic_frontier() {
        let options = vec![
            candidate("same", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass),
            candidate("same", 0.35, 0.55, 5.0, 3.0, 3.0, QualityGateStatus::Pass),
        ];
        assert!(regenerative_pareto_frontier(&options, &requirements()).is_err());
    }
}
