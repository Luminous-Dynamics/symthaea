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
use symthaea_agribot::soil_process::{EvidenceRef, PyrolysisBatchResult};

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
    Ok(result)
}

/// Evaluate a candidate against explicit hard constraints, with no weighted composite
/// score. A scenario can pass this numerical screen but remains scenario evidence.
pub fn assess_regenerative_candidate(
    candidate: &RegenerativeDesignCandidate,
    requirements: &RegenerativeDesignRequirements,
) -> Result<RegenerativeDesignAssessment, DesignError> {
    requirements.validate()?;
    let metrics = calculate_metrics(candidate, requirements)?;
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

    let eligibility = if !failed_constraints.is_empty() {
        CandidateEligibility::Ineligible
    } else if candidate.quality_gate.status == QualityGateStatus::Unknown {
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
        field_validation_required: true,
        scope_note: "design-screen result only; not an amendment certification or field recommendation".into(),
    })
}

fn tolerance(a: f64, b: f64) -> f64 {
    RELATIVE_EPSILON * a.abs().max(b.abs()).max(1.0)
}

fn dominates(a: RegenerativeDesignMetrics, b: RegenerativeDesignMetrics) -> bool {
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

    no_worse && strictly_better
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
            i != j && dominates(*other_metrics, *candidate_metrics)
        });
        if !is_dominated {
            frontier.push(candidate_id.clone());
        }
    }
    frontier.sort();
    Ok(frontier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_agribot::soil_process::{EvidenceKind, PyrolysisEvidence};

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
            feedstock_carbon_kg: 100.0,
            char_carbon_kg: 100.0 * carbon_retained,
            carbon_not_in_char_kg: 100.0 * (1.0 - carbon_retained),
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
    fn candidate_ids_must_be_unique_for_deterministic_frontier() {
        let options = vec![
            candidate("same", 0.30, 0.60, 3.0, 4.0, 4.0, QualityGateStatus::Pass),
            candidate("same", 0.35, 0.55, 5.0, 3.0, 3.0, QualityGateStatus::Pass),
        ];
        assert!(regenerative_pareto_frontier(&options, &requirements()).is_err());
    }
}
