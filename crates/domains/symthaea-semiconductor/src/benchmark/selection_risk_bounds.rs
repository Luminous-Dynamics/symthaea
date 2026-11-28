// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Population-risk identification under selective metrology qualification.
//!
//! Qualified-only coverage is not automatically population coverage. This
//! module keeps qualified outcomes separate from UNKNOWN/BLOCKED/unmeasured
//! outcomes, computes worst-case partial-identification bounds, and optionally
//! estimates population risk from an independent Bernoulli audit channel.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const SELECTION_RISK_INPUT_SCHEMA_V1: &str = "luminous.semi-bench.selection-risk-input.v1";
pub const SELECTION_AUDIT_DESIGN_SCHEMA_V1: &str = "luminous.semi-bench.selection-audit-design.v1";
pub const SELECTION_RISK_REPORT_SCHEMA_V1: &str = "luminous.semi-bench.selection-risk-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopulationOutcomeStateV1 {
    Qualified,
    Unknown,
    Blocked,
    Unmeasured,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualifiedLossObservationV1 {
    pub round_id: u64,
    pub loss: f64,
    pub evidence_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionAuditDesignV1 {
    pub schema: String,
    /// Digest of the preregistered audit randomization design.
    pub design_sha256: String,
    /// Commitment/receipt for the independent randomization mechanism.
    pub randomization_provenance_sha256: String,
    /// Traceable reference-method identity; this must be distinct from the failed operational channel.
    pub reference_method_sha256: String,
    /// Declared lower bound on all per-unit Bernoulli inclusion probabilities.
    pub minimum_inclusion_probability: f64,
    pub confidence_delta: f64,
    /// Must be true: audit selection is independent of operational qualification and loss.
    pub independent_of_operational_qualification: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndependentAuditLossObservationV1 {
    pub round_id: u64,
    pub loss: f64,
    pub inclusion_probability: f64,
    pub audit_measurement_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionRiskInputV1 {
    pub schema: String,
    pub input_id: String,
    pub population_manifest_sha256: String,
    pub source_ledger_sha256: String,
    pub population_round_ids: Vec<u64>,
    pub qualified_round_ids: Vec<u64>,
    pub unknown_round_ids: Vec<u64>,
    pub blocked_round_ids: Vec<u64>,
    pub unmeasured_round_ids: Vec<u64>,
    pub qualified_losses: Vec<QualifiedLossObservationV1>,
    pub audit_design: Option<SelectionAuditDesignV1>,
    pub independent_audit_losses: Vec<IndependentAuditLossObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndependentAuditEstimateV1 {
    pub population_risk_estimate: f64,
    pub variance_estimate_of_population_risk: f64,
    pub standard_error: f64,
    pub normal_95_lower: f64,
    pub normal_95_upper: f64,
    pub hoeffding_confidence_delta: f64,
    pub hoeffding_radius: f64,
    pub hoeffding_lower: f64,
    pub hoeffding_upper: f64,
    pub effective_sample_size: f64,
    pub minimum_inclusion_probability: f64,
    pub audit_sample_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionRiskReportV1 {
    pub schema: String,
    pub input_id: String,
    pub population_count: usize,
    pub qualified_count: usize,
    pub unknown_count: usize,
    pub blocked_count: usize,
    pub unmeasured_count: usize,
    pub missing_count: usize,
    pub qualified_loss_sum: f64,
    pub qualified_subset_risk: Option<f64>,
    /// Lower/upper population risk when all missing losses may be anywhere in [0,1].
    pub partial_identification_lower: f64,
    pub partial_identification_upper: f64,
    pub identification_interval_width: f64,
    pub population_risk_exactly_identified: bool,
    pub independent_audit_estimate: Option<IndependentAuditEstimateV1>,
    pub population_point_estimate: Option<f64>,
    pub point_estimate_method: Option<String>,
    pub population_claim_authorized_by_qualified_subset_alone: bool,
    pub assumptions_and_limits: Vec<String>,
    pub input_sha256: String,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SelectionRiskErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, SelectionRiskErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        SelectionRiskErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn report_digest(report: &SelectionRiskReportV1) -> Result<String, SelectionRiskErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn bounded_loss(value: f64, field: &str) -> Result<(), SelectionRiskErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(SelectionRiskErrorV1(format!(
            "{field} must be finite and in [0,1]"
        )));
    }
    Ok(())
}

fn validate_partition(input: &SelectionRiskInputV1) -> Result<BTreeSet<u64>, SelectionRiskErrorV1> {
    let population: BTreeSet<u64> = input.population_round_ids.iter().copied().collect();
    if population.len() != input.population_round_ids.len() || population.contains(&0) {
        return Err(SelectionRiskErrorV1(
            "population round IDs must be unique and nonzero".into(),
        ));
    }
    let mut assigned = BTreeSet::new();
    for (name, ids) in [
        ("qualified", &input.qualified_round_ids),
        ("unknown", &input.unknown_round_ids),
        ("blocked", &input.blocked_round_ids),
        ("unmeasured", &input.unmeasured_round_ids),
    ] {
        let unique: BTreeSet<u64> = ids.iter().copied().collect();
        if unique.len() != ids.len() {
            return Err(SelectionRiskErrorV1(format!(
                "{name} partition repeats a round ID"
            )));
        }
        for id in unique {
            if !population.contains(&id) {
                return Err(SelectionRiskErrorV1(format!(
                    "{name} round ID {id} is outside the declared population"
                )));
            }
            if !assigned.insert(id) {
                return Err(SelectionRiskErrorV1(format!(
                    "round ID {id} appears in more than one outcome partition"
                )));
            }
        }
    }
    if assigned != population {
        return Err(SelectionRiskErrorV1(
            "qualified/unknown/blocked/unmeasured partitions do not cover the full population"
                .into(),
        ));
    }
    Ok(population)
}

fn estimate_from_independent_audit(
    input: &SelectionRiskInputV1,
    missing_ids: &BTreeSet<u64>,
    population_count: usize,
    qualified_loss_sum: f64,
) -> Result<Option<IndependentAuditEstimateV1>, SelectionRiskErrorV1> {
    let Some(design) = &input.audit_design else {
        if !input.independent_audit_losses.is_empty() {
            return Err(SelectionRiskErrorV1(
                "audit observations supplied without an audit design".into(),
            ));
        }
        return Ok(None);
    };
    if design.schema != SELECTION_AUDIT_DESIGN_SCHEMA_V1
        || !valid_sha256(&design.design_sha256)
        || !valid_sha256(&design.randomization_provenance_sha256)
        || !valid_sha256(&design.reference_method_sha256)
    {
        return Err(SelectionRiskErrorV1(
            "audit design schema or provenance digest is invalid".into(),
        ));
    }
    if !design.independent_of_operational_qualification {
        return Err(SelectionRiskErrorV1(
            "audit design is not independent of operational qualification".into(),
        ));
    }
    if !design.minimum_inclusion_probability.is_finite()
        || design.minimum_inclusion_probability <= 0.0
        || design.minimum_inclusion_probability > 1.0
        || !design.confidence_delta.is_finite()
        || !(0.0..1.0).contains(&design.confidence_delta)
    {
        return Err(SelectionRiskErrorV1(
            "audit minimum inclusion probability or confidence delta is invalid".into(),
        ));
    }

    let mut sample_ids = BTreeSet::new();
    let mut ht_missing_sum = 0.0;
    let mut variance_total = 0.0;
    let mut sum_weights = 0.0;
    let mut sum_squared_weights = 0.0;
    for observation in &input.independent_audit_losses {
        if !missing_ids.contains(&observation.round_id) || !sample_ids.insert(observation.round_id)
        {
            return Err(SelectionRiskErrorV1(
                "audit round must be unique and belong to the nonqualified population".into(),
            ));
        }
        bounded_loss(observation.loss, "audit loss")?;
        if !observation.inclusion_probability.is_finite()
            || observation.inclusion_probability < design.minimum_inclusion_probability
            || observation.inclusion_probability > 1.0
        {
            return Err(SelectionRiskErrorV1(
                "audit inclusion probability is outside the declared positivity bound".into(),
            ));
        }
        if !valid_sha256(&observation.audit_measurement_receipt_sha256) {
            return Err(SelectionRiskErrorV1(
                "audit measurement receipt digest is malformed".into(),
            ));
        }
        let pi = observation.inclusion_probability;
        let weight = 1.0 / pi;
        ht_missing_sum += observation.loss * weight;
        variance_total += (1.0 - pi) * observation.loss * observation.loss / (pi * pi);
        sum_weights += weight;
        sum_squared_weights += weight * weight;
    }
    if !ht_missing_sum.is_finite()
        || !variance_total.is_finite()
        || !sum_weights.is_finite()
        || !sum_squared_weights.is_finite()
    {
        return Err(SelectionRiskErrorV1(
            "audit estimator has a non-finite accumulated term".into(),
        ));
    }

    let n = population_count as f64;
    let estimate = (qualified_loss_sum + ht_missing_sum) / n;
    let variance = variance_total / (n * n);
    let standard_error = variance.sqrt();
    let normal_radius = 1.959_963_984_540_054 * standard_error;
    let missing_count = missing_ids.len();
    let hoeffding_radius = if missing_count == 0 {
        0.0
    } else {
        ((missing_count as f64 * (2.0 / design.confidence_delta).ln()) / 2.0).sqrt()
            / (n * design.minimum_inclusion_probability)
    };
    if !estimate.is_finite()
        || !variance.is_finite()
        || !standard_error.is_finite()
        || !hoeffding_radius.is_finite()
    {
        return Err(SelectionRiskErrorV1(
            "audit estimate or confidence radius is non-finite".into(),
        ));
    }

    Ok(Some(IndependentAuditEstimateV1 {
        population_risk_estimate: estimate,
        variance_estimate_of_population_risk: variance,
        standard_error,
        normal_95_lower: (estimate - normal_radius).clamp(0.0, 1.0),
        normal_95_upper: (estimate + normal_radius).clamp(0.0, 1.0),
        hoeffding_confidence_delta: design.confidence_delta,
        hoeffding_radius,
        hoeffding_lower: (estimate - hoeffding_radius).clamp(0.0, 1.0),
        hoeffding_upper: (estimate + hoeffding_radius).clamp(0.0, 1.0),
        effective_sample_size: if sum_squared_weights > 0.0 {
            sum_weights * sum_weights / sum_squared_weights
        } else {
            0.0
        },
        minimum_inclusion_probability: design.minimum_inclusion_probability,
        audit_sample_count: input.independent_audit_losses.len(),
    }))
}

/// Computes qualified-subset risk, worst-case population identification bounds,
/// and (when configured) an independent Bernoulli audit Horvitz-Thompson estimate.
pub fn evaluate_selection_adjusted_population_risk_v1(
    input: &SelectionRiskInputV1,
) -> Result<SelectionRiskReportV1, SelectionRiskErrorV1> {
    let mut canonical_input = input.clone();
    canonical_input.population_round_ids.sort_unstable();
    canonical_input.qualified_round_ids.sort_unstable();
    canonical_input.unknown_round_ids.sort_unstable();
    canonical_input.blocked_round_ids.sort_unstable();
    canonical_input.unmeasured_round_ids.sort_unstable();
    canonical_input
        .qualified_losses
        .sort_by_key(|item| item.round_id);
    canonical_input
        .independent_audit_losses
        .sort_by_key(|item| item.round_id);
    let input = &canonical_input;

    if input.schema != SELECTION_RISK_INPUT_SCHEMA_V1 {
        return Err(SelectionRiskErrorV1(
            "selection-risk input schema is unsupported".into(),
        ));
    }
    if input.input_id.trim().is_empty()
        || !valid_sha256(&input.population_manifest_sha256)
        || !valid_sha256(&input.source_ledger_sha256)
    {
        return Err(SelectionRiskErrorV1(
            "input identity or source manifest digest is invalid".into(),
        ));
    }
    let population = validate_partition(input)?;
    let population_count = population.len();
    if population_count == 0 {
        return Err(SelectionRiskErrorV1(
            "population must contain at least one prediction".into(),
        ));
    }

    let qualified_ids: BTreeSet<u64> = input.qualified_round_ids.iter().copied().collect();
    let mut loss_ids = BTreeSet::new();
    let mut qualified_loss_sum = 0.0;
    for observation in &input.qualified_losses {
        if !qualified_ids.contains(&observation.round_id) || !loss_ids.insert(observation.round_id)
        {
            return Err(SelectionRiskErrorV1(
                "qualified loss must uniquely reference a qualified population round".into(),
            ));
        }
        bounded_loss(observation.loss, "qualified loss")?;
        if !valid_sha256(&observation.evidence_receipt_sha256) {
            return Err(SelectionRiskErrorV1(
                "qualified loss evidence receipt digest is malformed".into(),
            ));
        }
        qualified_loss_sum += observation.loss;
    }
    if loss_ids != qualified_ids {
        return Err(SelectionRiskErrorV1(
            "each qualified population round must have exactly one qualified loss".into(),
        ));
    }

    let unknown_count = input.unknown_round_ids.len();
    let blocked_count = input.blocked_round_ids.len();
    let unmeasured_count = input.unmeasured_round_ids.len();
    let missing_count = unknown_count + blocked_count + unmeasured_count;
    let lower = qualified_loss_sum / population_count as f64;
    let upper = (qualified_loss_sum + missing_count as f64) / population_count as f64;
    let audit = estimate_from_independent_audit(
        input,
        &population.difference(&qualified_ids).copied().collect(),
        population_count,
        qualified_loss_sum,
    )?;
    let exact = missing_count == 0;
    let population_point_estimate = if exact {
        Some(lower)
    } else {
        audit
            .as_ref()
            .map(|estimate| estimate.population_risk_estimate)
    };
    let mut assumptions_and_limits = vec![
        "Qualified-subset risk is not automatically population risk under outcome-dependent qualification.".into(),
        "Worst-case population bounds assume only that each missing loss lies in [0,1]; they make no missing-at-random assumption.".into(),
    ];
    if audit.is_some() {
        assumptions_and_limits.push(
            "The independent-audit estimate assumes independent Bernoulli sampling with the declared inclusion probabilities and an audit reference measurement independent of operational qualification.".into(),
        );
        assumptions_and_limits.push(
            "The Hoeffding interval is the conservative finite-sample bound; the normal interval is descriptive and is not a finite-sample guarantee.".into(),
        );
    }
    if missing_count > 0 && audit.is_none() {
        assumptions_and_limits.push(
            "No population point estimate is identified without a complete outcome ledger or an independent probability audit.".into(),
        );
    }

    let mut report = SelectionRiskReportV1 {
        schema: SELECTION_RISK_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        population_count,
        qualified_count: input.qualified_round_ids.len(),
        unknown_count,
        blocked_count,
        unmeasured_count,
        missing_count,
        qualified_loss_sum,
        qualified_subset_risk: if input.qualified_round_ids.is_empty() {
            None
        } else {
            Some(qualified_loss_sum / input.qualified_round_ids.len() as f64)
        },
        partial_identification_lower: lower,
        partial_identification_upper: upper,
        identification_interval_width: upper - lower,
        population_risk_exactly_identified: exact,
        independent_audit_estimate: audit,
        population_point_estimate,
        point_estimate_method: if exact {
            Some("complete_qualified_population".into())
        } else if input.audit_design.is_some() {
            Some("independent_bernoulli_audit_horvitz_thompson".into())
        } else {
            None
        },
        population_claim_authorized_by_qualified_subset_alone: exact,
        assumptions_and_limits,
        input_sha256: json_sha256(input)?,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn input() -> SelectionRiskInputV1 {
        SelectionRiskInputV1 {
            schema: SELECTION_RISK_INPUT_SCHEMA_V1.into(),
            input_id: "selection-risk-fixture-001".into(),
            population_manifest_sha256: digest('a'),
            source_ledger_sha256: digest('b'),
            population_round_ids: vec![1, 2, 3, 4],
            qualified_round_ids: vec![1, 2],
            unknown_round_ids: vec![3],
            blocked_round_ids: vec![4],
            unmeasured_round_ids: vec![],
            qualified_losses: vec![
                QualifiedLossObservationV1 {
                    round_id: 1,
                    loss: 1.0,
                    evidence_receipt_sha256: digest('c'),
                },
                QualifiedLossObservationV1 {
                    round_id: 2,
                    loss: 0.0,
                    evidence_receipt_sha256: digest('d'),
                },
            ],
            audit_design: None,
            independent_audit_losses: vec![],
        }
    }

    fn audit_design() -> SelectionAuditDesignV1 {
        SelectionAuditDesignV1 {
            schema: SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_sha256: digest('e'),
            randomization_provenance_sha256: digest('f'),
            reference_method_sha256: digest('1'),
            minimum_inclusion_probability: 0.5,
            confidence_delta: 0.05,
            independent_of_operational_qualification: true,
        }
    }

    #[test]
    fn qualified_subset_risk_is_not_population_risk() {
        let report = evaluate_selection_adjusted_population_risk_v1(&input()).unwrap();
        assert_eq!(report.qualified_subset_risk, Some(0.5));
        assert_eq!(report.partial_identification_lower, 0.25);
        assert_eq!(report.partial_identification_upper, 0.75);
        assert!(!report.population_risk_exactly_identified);
        assert_eq!(report.population_point_estimate, None);
        assert!(!report.population_claim_authorized_by_qualified_subset_alone);
    }

    #[test]
    fn complete_population_has_exact_risk() {
        let mut data = input();
        data.population_round_ids = vec![1, 2];
        data.unknown_round_ids.clear();
        data.blocked_round_ids.clear();
        data.qualified_losses.truncate(2);
        let report = evaluate_selection_adjusted_population_risk_v1(&data).unwrap();
        assert_eq!(report.partial_identification_lower, 0.5);
        assert_eq!(report.partial_identification_upper, 0.5);
        assert!(report.population_risk_exactly_identified);
        assert_eq!(report.population_point_estimate, Some(0.5));
        assert!(report.population_claim_authorized_by_qualified_subset_alone);
    }

    #[test]
    fn independent_bernoulli_audit_estimates_population_risk() {
        let mut data = input();
        data.audit_design = Some(audit_design());
        data.independent_audit_losses = vec![
            IndependentAuditLossObservationV1 {
                round_id: 3,
                loss: 1.0,
                inclusion_probability: 0.5,
                audit_measurement_receipt_sha256: digest('2'),
            },
            IndependentAuditLossObservationV1 {
                round_id: 4,
                loss: 0.0,
                inclusion_probability: 1.0,
                audit_measurement_receipt_sha256: digest('3'),
            },
        ];
        let report = evaluate_selection_adjusted_population_risk_v1(&data).unwrap();
        let audit = report.independent_audit_estimate.unwrap();
        assert_eq!(audit.population_risk_estimate, 0.75);
        assert_eq!(audit.audit_sample_count, 2);
        assert!(audit.hoeffding_lower <= 0.5);
        assert!(audit.hoeffding_upper >= 0.5);
        assert!(audit.effective_sample_size > 0.0);
    }

    #[test]
    fn audit_must_be_independent_of_operational_qualification() {
        let mut data = input();
        let mut design = audit_design();
        design.independent_of_operational_qualification = false;
        data.audit_design = Some(design);
        assert!(evaluate_selection_adjusted_population_risk_v1(&data).is_err());
    }

    #[test]
    fn audit_positivity_and_receipt_integrity_are_enforced() {
        let mut data = input();
        data.audit_design = Some(audit_design());
        data.independent_audit_losses = vec![IndependentAuditLossObservationV1 {
            round_id: 3,
            loss: 1.0,
            inclusion_probability: 0.25,
            audit_measurement_receipt_sha256: digest('2'),
        }];
        assert!(evaluate_selection_adjusted_population_risk_v1(&data).is_err());
        data.independent_audit_losses[0].inclusion_probability = 0.5;
        data.independent_audit_losses[0].audit_measurement_receipt_sha256 = "bad".into();
        assert!(evaluate_selection_adjusted_population_risk_v1(&data).is_err());
    }

    #[test]
    fn partition_overlap_or_unassigned_round_is_rejected() {
        let mut data = input();
        data.blocked_round_ids.push(3);
        assert!(evaluate_selection_adjusted_population_risk_v1(&data).is_err());
        let mut data = input();
        data.unmeasured_round_ids.push(5);
        assert!(evaluate_selection_adjusted_population_risk_v1(&data).is_err());
    }

    #[test]
    fn audit_sample_must_be_from_missing_population_only() {
        let mut data = input();
        data.audit_design = Some(audit_design());
        data.independent_audit_losses = vec![IndependentAuditLossObservationV1 {
            round_id: 1,
            loss: 0.0,
            inclusion_probability: 0.5,
            audit_measurement_receipt_sha256: digest('2'),
        }];
        assert!(evaluate_selection_adjusted_population_risk_v1(&data).is_err());
    }

    #[test]
    fn input_order_permutations_preserve_risk_report_digest() {
        let mut first = input();
        first.audit_design = Some(audit_design());
        first.independent_audit_losses = vec![
            IndependentAuditLossObservationV1 {
                round_id: 3,
                loss: 1.0,
                inclusion_probability: 0.5,
                audit_measurement_receipt_sha256: digest('2'),
            },
            IndependentAuditLossObservationV1 {
                round_id: 4,
                loss: 0.0,
                inclusion_probability: 1.0,
                audit_measurement_receipt_sha256: digest('3'),
            },
        ];
        let report_a = evaluate_selection_adjusted_population_risk_v1(&first).unwrap();
        first.population_round_ids.reverse();
        first.qualified_round_ids.reverse();
        first.unknown_round_ids.reverse();
        first.blocked_round_ids.reverse();
        first.qualified_losses.reverse();
        first.independent_audit_losses.reverse();
        let report_b = evaluate_selection_adjusted_population_risk_v1(&first).unwrap();
        assert_eq!(report_a.report_sha256, report_b.report_sha256);
    }

    #[test]
    fn audit_empty_sample_has_no_false_population_point_certainty() {
        let mut data = input();
        data.audit_design = Some(audit_design());
        let report = evaluate_selection_adjusted_population_risk_v1(&data).unwrap();
        let audit = report.independent_audit_estimate.unwrap();
        assert_eq!(audit.audit_sample_count, 0);
        assert!(!report.population_risk_exactly_identified);
        assert_eq!(report.partial_identification_upper, 0.75);
        assert!(audit.hoeffding_upper >= report.partial_identification_lower);
    }
}
