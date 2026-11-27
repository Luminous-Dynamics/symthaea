// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Population-risk accounting under selective, delayed and qualified metrology.
//!
//! This module separates physical observation from online tracking/use. It
//! always emits worst-case bounds for a bounded [0,1] loss. Horvitz-Thompson
//! estimates are emitted only when all conditional stage propensities and the
//! declared ignorability/positivity/factorization assumptions are present.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const QUALIFIED_OBSERVATION_RISK_SCHEMA_V1: &str =
    "luminous.semi-bench.qualified-observation-risk.v1";
pub const QUALIFIED_OBSERVATION_RISK_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.qualified-observation-risk-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologyQualificationStatusV1 {
    NotSelected,
    NotReceived,
    Qualified,
    Unknown,
    Blocked,
    Revoked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopulationRiskIdentificationV1 {
    ExactAllUnitRisk,
    IpwIdentifiedUnderDeclaredAssumptions,
    WorstCaseBoundsOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationStageProbabilitiesV1 {
    /// P(selected for physical metrology).
    pub selection_probability: Option<f64>,
    /// P(physical result received | selected).
    pub physical_delivery_probability_given_selection: Option<f64>,
    /// P(K1-qualified | physical result received).
    pub qualification_probability_given_delivery: Option<f64>,
    /// P(admitted to online feedback tracking | selected).
    pub online_tracking_probability_given_selection: Option<f64>,
    /// P(available by the update frontier | K1-qualified).
    pub frontier_availability_probability_given_qualified: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualifiedObservationUnitV1 {
    pub round: u64,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub selected_for_metrology: bool,
    pub admitted_to_online_tracking: bool,
    pub physical_result_received: bool,
    pub qualification_status: MetrologyQualificationStatusV1,
    pub available_by_update_frontier: bool,
    /// True only when the valid qualified outcome was actually admitted to an
    /// online update. Offline evaluation may still retain a label not tracked
    /// online.
    pub online_update_used: bool,
    /// Bounded loss in [0,1] from a physically received and K1-qualified
    /// outcome. It may exist even when the online tracking slot was not used.
    pub qualified_loss: Option<f64>,
    pub probabilities: ObservationStageProbabilitiesV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationProcessAssumptionsV1 {
    /// Declared conditional-probability factorization order is S, R|S, Q|R,
    /// A|S, V|Q. This flag records whether each probability is justified.
    pub conditional_probability_factorization_verified: bool,
    /// Qualification/delivery are ignorable conditional on the declared
    /// covariates and stages. False for unresolved MNAR mechanisms.
    pub conditional_ignorability_verified: bool,
    /// All required stage propensities are known and strictly positive.
    pub positivity_verified: bool,
    pub evidence_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationRiskInputV1 {
    pub schema: String,
    pub input_id: String,
    pub population_round_count: u64,
    pub units: Vec<QualifiedObservationUnitV1>,
    pub assumptions: ObservationProcessAssumptionsV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseProbabilityRiskEstimateV1 {
    pub estimate: f64,
    pub observed_units: usize,
    pub population_units: u64,
    pub effective_sample_size: f64,
    pub minimum_joint_inclusion_probability: f64,
    pub maximum_inverse_weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PopulationRiskBoundsV1 {
    /// Deterministic finite-population identification bound, not a confidence interval.
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub observed_qualified_units: usize,
    pub missing_or_unqualified_units: usize,
    pub population_units: u64,
    pub observed_only_mean: Option<f64>,
    pub interval_width: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualifiedObservationRiskReportV1 {
    pub schema: String,
    pub input_id: String,
    pub population_units: u64,
    pub identification: PopulationRiskIdentificationV1,
    pub worst_case_bounds: PopulationRiskBoundsV1,
    /// Horvitz-Thompson estimates of population risk, only when all declared
    /// assumptions and all required propensities are available.
    pub physical_observation_ipw: Option<InverseProbabilityRiskEstimateV1>,
    /// Optional IPW estimate based only on feedback actually used online.
    /// This helps diagnose the additional tracking/availability selection.
    pub online_update_ipw: Option<InverseProbabilityRiskEstimateV1>,
    pub qualified_fraction: f64,
    pub online_update_fraction: f64,
    pub unknown_fraction: f64,
    pub blocked_or_revoked_fraction: f64,
    pub mnar_or_propensity_warning: bool,
    pub theorem_or_guarantee_claimed: bool,
    pub reason_codes: Vec<String>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct QualifiedObservationRiskErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), QualifiedObservationRiskErrorV1> {
    if value.trim().is_empty() {
        return Err(QualifiedObservationRiskErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, QualifiedObservationRiskErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        QualifiedObservationRiskErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_probability(
    probability: Option<f64>,
    field: &str,
) -> Result<f64, QualifiedObservationRiskErrorV1> {
    let Some(value) = probability else {
        return Err(QualifiedObservationRiskErrorV1(format!(
            "{field}: propensity is unknown"
        )));
    };
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(QualifiedObservationRiskErrorV1(format!(
            "{field}: propensity must be in (0,1]"
        )));
    }
    Ok(value)
}

fn physical_joint_probability(
    probabilities: &ObservationStageProbabilitiesV1,
) -> Result<f64, QualifiedObservationRiskErrorV1> {
    Ok(validate_probability(
        probabilities.selection_probability,
        "selection_probability",
    )? * validate_probability(
        probabilities.physical_delivery_probability_given_selection,
        "physical_delivery_probability_given_selection",
    )? * validate_probability(
        probabilities.qualification_probability_given_delivery,
        "qualification_probability_given_delivery",
    )?)
}

fn online_joint_probability(
    probabilities: &ObservationStageProbabilitiesV1,
) -> Result<f64, QualifiedObservationRiskErrorV1> {
    Ok(physical_joint_probability(probabilities)?
        * validate_probability(
            probabilities.online_tracking_probability_given_selection,
            "online_tracking_probability_given_selection",
        )?
        * validate_probability(
            probabilities.frontier_availability_probability_given_qualified,
            "frontier_availability_probability_given_qualified",
        )?)
}

fn validate_unit(unit: &QualifiedObservationUnitV1) -> Result<(), QualifiedObservationRiskErrorV1> {
    require_text(&unit.sample_id, "unit.sample_id")?;
    require_text(&unit.measurement_event_id, "unit.measurement_event_id")?;

    if unit.admitted_to_online_tracking && !unit.selected_for_metrology {
        return Err(QualifiedObservationRiskErrorV1(format!(
            "round {} was tracked without being selected",
            unit.round
        )));
    }
    if unit.physical_result_received && !unit.selected_for_metrology {
        return Err(QualifiedObservationRiskErrorV1(format!(
            "round {} has a physical result without selection",
            unit.round
        )));
    }
    if unit.available_by_update_frontier && !unit.physical_result_received {
        return Err(QualifiedObservationRiskErrorV1(format!(
            "round {} is available without a received physical result",
            unit.round
        )));
    }
    if unit.online_update_used
        && (!unit.selected_for_metrology
            || !unit.admitted_to_online_tracking
            || !unit.physical_result_received
            || unit.qualification_status != MetrologyQualificationStatusV1::Qualified
            || !unit.available_by_update_frontier)
    {
        return Err(QualifiedObservationRiskErrorV1(format!(
            "round {} was used online without every required observation stage",
            unit.round
        )));
    }
    match (unit.qualification_status, unit.qualified_loss) {
        (MetrologyQualificationStatusV1::Qualified, Some(loss))
            if unit.physical_result_received && loss.is_finite() && (0.0..=1.0).contains(&loss) =>
        {
            Ok(())
        }
        (MetrologyQualificationStatusV1::Qualified, _) => {
            Err(QualifiedObservationRiskErrorV1(format!(
                "round {} is marked qualified without a finite bounded loss",
                unit.round
            )))
        }
        (_, Some(_)) => Err(QualifiedObservationRiskErrorV1(format!(
            "round {} exposes a loss for a non-qualified result",
            unit.round
        ))),
        _ => Ok(()),
    }
}

fn horvitz_thompson(
    units: &[QualifiedObservationUnitV1],
    population: u64,
    online_only: bool,
) -> Result<InverseProbabilityRiskEstimateV1, QualifiedObservationRiskErrorV1> {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut squared_weight_sum = 0.0;
    let mut minimum_probability = 1.0_f64;
    let mut maximum_weight = 0.0_f64;
    let mut observed = 0_usize;

    for unit in units {
        let used = if online_only {
            unit.online_update_used
        } else {
            unit.qualified_loss.is_some()
        };
        if !used {
            continue;
        }
        let loss = unit.qualified_loss.expect("observed unit has a loss");
        let probability = if online_only {
            online_joint_probability(&unit.probabilities)?
        } else {
            physical_joint_probability(&unit.probabilities)?
        };
        let weight = 1.0 / probability;
        weighted_sum += weight * loss;
        weight_sum += weight;
        squared_weight_sum += weight * weight;
        minimum_probability = minimum_probability.min(probability);
        maximum_weight = maximum_weight.max(weight);
        observed += 1;
    }

    if observed == 0 {
        return Err(QualifiedObservationRiskErrorV1(
            "no observed outcomes are available for IPW".into(),
        ));
    }
    let estimate = weighted_sum / population as f64;
    let effective_sample_size = if squared_weight_sum > 0.0 {
        weight_sum * weight_sum / squared_weight_sum
    } else {
        0.0
    };
    if !estimate.is_finite() || !effective_sample_size.is_finite() {
        return Err(QualifiedObservationRiskErrorV1(
            "IPW estimate or effective sample size is non-finite".into(),
        ));
    }
    Ok(InverseProbabilityRiskEstimateV1 {
        estimate,
        observed_units: observed,
        population_units: population,
        effective_sample_size,
        minimum_joint_inclusion_probability: minimum_probability,
        maximum_inverse_weight: maximum_weight,
    })
}

/// Computes finite-population worst-case bounds unconditionally and emits IPW
/// estimates only when the caller explicitly declares the conditional
/// factorization, ignorability and positivity assumptions.
pub fn compute_qualified_observation_risk_v1(
    input: &ObservationRiskInputV1,
) -> Result<QualifiedObservationRiskReportV1, QualifiedObservationRiskErrorV1> {
    if input.schema != QUALIFIED_OBSERVATION_RISK_SCHEMA_V1 {
        return Err(QualifiedObservationRiskErrorV1(
            "observation risk input schema is unsupported".into(),
        ));
    }
    require_text(&input.input_id, "input.input_id")?;
    if input.population_round_count == 0
        || input.units.len() as u64 != input.population_round_count
    {
        return Err(QualifiedObservationRiskErrorV1(
            "population count must equal the number of unique observation units".into(),
        ));
    }
    if !valid_sha256(&input.assumptions.evidence_receipt_sha256) {
        return Err(QualifiedObservationRiskErrorV1(
            "assumption evidence receipt digest is malformed".into(),
        ));
    }

    let mut rounds = BTreeSet::new();
    let mut samples = BTreeSet::new();
    let mut observed_loss_sum = 0.0;
    let mut observed_count = 0_usize;
    let mut online_count = 0_usize;
    let mut unknown_count = 0_usize;
    let mut blocked_count = 0_usize;
    let mut probabilities_complete = true;
    let mut positive_propensities = true;

    let mut sorted_units = input.units.clone();
    sorted_units.sort_by_key(|unit| unit.round);
    for unit in &sorted_units {
        validate_unit(unit)?;
        if !rounds.insert(unit.round) || !samples.insert(unit.sample_id.as_str()) {
            return Err(QualifiedObservationRiskErrorV1(
                "duplicate round or sample identity in observation ledger".into(),
            ));
        }

        match unit.qualification_status {
            MetrologyQualificationStatusV1::Unknown => unknown_count += 1,
            MetrologyQualificationStatusV1::Blocked | MetrologyQualificationStatusV1::Revoked => {
                blocked_count += 1;
            }
            _ => {}
        }
        if let Some(loss) = unit.qualified_loss {
            observed_loss_sum += loss;
            observed_count += 1;
        }
        if unit.online_update_used {
            online_count += 1;
        }

        let required = [
            unit.probabilities.selection_probability,
            unit.probabilities.physical_delivery_probability_given_selection,
            unit.probabilities.qualification_probability_given_delivery,
            unit.probabilities.online_tracking_probability_given_selection,
            unit.probabilities.frontier_availability_probability_given_qualified,
        ];
        if required.iter().any(Option::is_none) {
            probabilities_complete = false;
        }
        for value in required.into_iter().flatten() {
            if !value.is_finite() || value <= 0.0 || value > 1.0 {
                positive_propensities = false;
            }
        }
    }

    let population = input.population_round_count as f64;
    let missing_count = input.population_round_count as usize - observed_count;
    let lower_bound = observed_loss_sum / population;
    let upper_bound = (observed_loss_sum + missing_count as f64) / population;
    let observed_only_mean = if observed_count > 0 {
        Some(observed_loss_sum / observed_count as f64)
    } else {
        None
    };
    let worst_case_bounds = PopulationRiskBoundsV1 {
        lower_bound,
        upper_bound,
        observed_qualified_units: observed_count,
        missing_or_unqualified_units: missing_count,
        population_units: input.population_round_count,
        observed_only_mean,
        interval_width: upper_bound - lower_bound,
    };

    let assumptions_verified = input.assumptions.conditional_probability_factorization_verified
        && input.assumptions.conditional_ignorability_verified
        && input.assumptions.positivity_verified
        && probabilities_complete
        && positive_propensities;

    let (physical_observation_ipw, online_update_ipw, identification, mnar_warning, reason_codes) =
        if observed_count == input.population_round_count {
            (
                Some(InverseProbabilityRiskEstimateV1 {
                    estimate: observed_loss_sum / population,
                    observed_units: observed_count,
                    population_units: input.population_round_count,
                    effective_sample_size: population,
                    minimum_joint_inclusion_probability: 1.0,
                    maximum_inverse_weight: 1.0,
                }),
                None,
                PopulationRiskIdentificationV1::ExactAllUnitRisk,
                false,
                Vec::new(),
            )
        } else if assumptions_verified {
            let physical = horvitz_thompson(
                &sorted_units,
                input.population_round_count,
                false,
            )?;
            let online = if online_count > 0 {
                Some(horvitz_thompson(
                    &sorted_units,
                    input.population_round_count,
                    true,
                )?)
            } else {
                None
            };
            (
                Some(physical),
                online,
                PopulationRiskIdentificationV1::IpwIdentifiedUnderDeclaredAssumptions,
                false,
                Vec::new(),
            )
        } else {
            let mut reasons = Vec::new();
            if !input.assumptions.conditional_probability_factorization_verified {
                reasons.push("conditional_probability_factorization_unverified".to_string());
            }
            if !input.assumptions.conditional_ignorability_verified {
                reasons.push("conditional_ignorability_unverified_or_mnar_possible".to_string());
            }
            if !input.assumptions.positivity_verified || !positive_propensities {
                reasons.push("positivity_not_verified".to_string());
            }
            if !probabilities_complete {
                reasons.push("stage_propensities_incomplete".to_string());
            }
            (
                None,
                None,
                PopulationRiskIdentificationV1::WorstCaseBoundsOnly,
                true,
                reasons,
            )
        };

    let mut report = QualifiedObservationRiskReportV1 {
        schema: QUALIFIED_OBSERVATION_RISK_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        population_units: input.population_round_count,
        identification,
        worst_case_bounds,
        physical_observation_ipw,
        online_update_ipw,
        qualified_fraction: observed_count as f64 / population,
        online_update_fraction: online_count as f64 / population,
        unknown_fraction: unknown_count as f64 / population,
        blocked_or_revoked_fraction: blocked_count as f64 / population,
        mnar_or_propensity_warning: mnar_warning,
        theorem_or_guarantee_claimed: false,
        reason_codes,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 = json_sha256(&canonical)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn probabilities(
        selection: f64,
        delivery: f64,
        qualification: f64,
        tracking: f64,
        available: f64,
    ) -> ObservationStageProbabilitiesV1 {
        ObservationStageProbabilitiesV1 {
            selection_probability: Some(selection),
            physical_delivery_probability_given_selection: Some(delivery),
            qualification_probability_given_delivery: Some(qualification),
            online_tracking_probability_given_selection: Some(tracking),
            frontier_availability_probability_given_qualified: Some(available),
        }
    }

    fn qualified_unit(round: u64, loss: f64, online: bool) -> QualifiedObservationUnitV1 {
        QualifiedObservationUnitV1 {
            round,
            sample_id: format!("sample-{round}"),
            measurement_event_id: format!("event-{round}"),
            selected_for_metrology: true,
            admitted_to_online_tracking: online,
            physical_result_received: true,
            qualification_status: MetrologyQualificationStatusV1::Qualified,
            available_by_update_frontier: online,
            online_update_used: online,
            qualified_loss: Some(loss),
            probabilities: probabilities(1.0, 1.0, 1.0, 1.0, 1.0),
        }
    }

    fn missing_unit(
        round: u64,
        status: MetrologyQualificationStatusV1,
        selected: bool,
        received: bool,
    ) -> QualifiedObservationUnitV1 {
        QualifiedObservationUnitV1 {
            round,
            sample_id: format!("sample-{round}"),
            measurement_event_id: format!("event-{round}"),
            selected_for_metrology: selected,
            admitted_to_online_tracking: false,
            physical_result_received: received,
            qualification_status: status,
            available_by_update_frontier: false,
            online_update_used: false,
            qualified_loss: None,
            probabilities: probabilities(0.5, 0.5, 0.5, 0.5, 0.5),
        }
    }

    fn input(units: Vec<QualifiedObservationUnitV1>, assumptions: ObservationProcessAssumptionsV1) -> ObservationRiskInputV1 {
        ObservationRiskInputV1 {
            schema: QUALIFIED_OBSERVATION_RISK_SCHEMA_V1.into(),
            input_id: "observation-risk-input-1".into(),
            population_round_count: units.len() as u64,
            units,
            assumptions,
        }
    }

    fn assumptions(ignorable: bool) -> ObservationProcessAssumptionsV1 {
        ObservationProcessAssumptionsV1 {
            conditional_probability_factorization_verified: true,
            conditional_ignorability_verified: ignorable,
            positivity_verified: true,
            evidence_receipt_sha256: digest('a'),
        }
    }

    #[test]
    fn complete_population_returns_exact_risk_not_an_ipw_claim() {
        let units = vec![
            qualified_unit(1, 1.0, true),
            qualified_unit(2, 0.0, true),
            qualified_unit(3, 1.0, false),
        ];
        let report = compute_qualified_observation_risk_v1(&input(units, assumptions(false))).unwrap();
        assert_eq!(report.identification, PopulationRiskIdentificationV1::ExactAllUnitRisk);
        assert!((report.worst_case_bounds.lower_bound - 2.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(report.worst_case_bounds.interval_width, 0.0);
        assert!(!report.theorem_or_guarantee_claimed);
    }

    #[test]
    fn selected_only_qualified_subset_gets_worst_case_population_bounds() {
        let units = vec![
            qualified_unit(1, 1.0, true),
            missing_unit(2, MetrologyQualificationStatusV1::NotSelected, false, false),
            missing_unit(3, MetrologyQualificationStatusV1::NotSelected, false, false),
        ];
        let report = compute_qualified_observation_risk_v1(&input(units, assumptions(false))).unwrap();
        assert_eq!(report.identification, PopulationRiskIdentificationV1::WorstCaseBoundsOnly);
        assert!((report.worst_case_bounds.lower_bound - 1.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(report.worst_case_bounds.upper_bound, 1.0);
        assert_eq!(report.worst_case_bounds.missing_or_unqualified_units, 2);
        assert!(report.mnar_or_propensity_warning);
        assert!(report.physical_observation_ipw.is_none());
    }

    #[test]
    fn known_mar_propensities_enable_physical_and_online_ipw() {
        let mut observed = qualified_unit(1, 1.0, true);
        observed.probabilities = probabilities(0.5, 1.0, 1.0, 0.5, 1.0);
        let mut missing = missing_unit(2, MetrologyQualificationStatusV1::NotSelected, false, false);
        missing.probabilities = probabilities(0.5, 1.0, 1.0, 0.5, 1.0);
        let report = compute_qualified_observation_risk_v1(&input(
            vec![observed, missing],
            assumptions(true),
        ))
        .unwrap();
        assert_eq!(report.identification, PopulationRiskIdentificationV1::IpwIdentifiedUnderDeclaredAssumptions);
        assert!((report.physical_observation_ipw.unwrap().estimate - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn mnar_qualification_failure_suppresses_ipw_even_when_numbers_are_supplied() {
        let mut observed = qualified_unit(1, 0.0, true);
        observed.probabilities = probabilities(0.5, 1.0, 0.5, 1.0, 1.0);
        let mut unknown = missing_unit(2, MetrologyQualificationStatusV1::Unknown, true, true);
        unknown.probabilities = probabilities(0.5, 1.0, 0.5, 1.0, 1.0);
        let report = compute_qualified_observation_risk_v1(&input(
            vec![observed, unknown],
            assumptions(false),
        ))
        .unwrap();
        assert!(report.physical_observation_ipw.is_none());
        assert_eq!(report.identification, PopulationRiskIdentificationV1::WorstCaseBoundsOnly);
        assert!(report.reason_codes.contains(
            &"conditional_ignorability_unverified_or_mnar_possible".to_string()
        ));
    }

    #[test]
    fn measured_but_not_tracked_unit_still_contributes_to_offline_population_risk() {
        let units = vec![
            qualified_unit(1, 0.0, true),
            qualified_unit(2, 1.0, false),
        ];
        let report = compute_qualified_observation_risk_v1(&input(units, assumptions(false))).unwrap();
        assert_eq!(report.worst_case_bounds.observed_qualified_units, 2);
        assert!((report.worst_case_bounds.lower_bound - 0.5).abs() < 1.0e-12);
        assert_eq!(report.online_update_fraction, 0.5);
        assert!(report.online_update_ipw.is_none());
    }

    #[test]
    fn unknown_and_blocked_results_widen_bounds_and_are_counted_separately() {
        let units = vec![
            qualified_unit(1, 0.0, true),
            missing_unit(2, MetrologyQualificationStatusV1::Unknown, true, true),
            missing_unit(3, MetrologyQualificationStatusV1::Blocked, true, true),
            missing_unit(4, MetrologyQualificationStatusV1::Revoked, true, true),
        ];
        let report = compute_qualified_observation_risk_v1(&input(units, assumptions(false))).unwrap();
        assert_eq!(report.unknown_fraction, 0.25);
        assert_eq!(report.blocked_or_revoked_fraction, 0.50);
        assert_eq!(report.worst_case_bounds.upper_bound, 0.75);
    }

    #[test]
    fn zero_or_unknown_propensity_fails_closed_to_worst_case_bounds() {
        let mut observed = qualified_unit(1, 0.4, true);
        observed.probabilities.qualification_probability_given_delivery = Some(0.0);
        let mut missing = missing_unit(2, MetrologyQualificationStatusV1::NotSelected, false, false);
        missing.probabilities.selection_probability = None;
        let report = compute_qualified_observation_risk_v1(&input(
            vec![observed, missing],
            assumptions(true),
        ))
        .unwrap();
        assert!(report.physical_observation_ipw.is_none());
        assert!(report.mnar_or_propensity_warning);
    }

    #[test]
    fn online_update_flag_requires_all_observation_stages() {
        let mut invalid = qualified_unit(1, 0.4, false);
        invalid.online_update_used = true;
        assert!(compute_qualified_observation_risk_v1(&input(
            vec![invalid],
            assumptions(true)
        ))
        .is_err());
    }

    #[test]
    fn duplicate_units_and_out_of_range_losses_are_rejected() {
        let unit = qualified_unit(1, 0.4, true);
        assert!(compute_qualified_observation_risk_v1(&input(
            vec![unit.clone(), unit],
            assumptions(true)
        ))
        .is_err());
        let mut invalid = qualified_unit(1, 1.2, true);
        assert!(compute_qualified_observation_risk_v1(&input(
            vec![invalid],
            assumptions(true)
        ))
        .is_err());
    }

    #[test]
    fn all_missing_population_has_uninformative_but_valid_bounds() {
        let units = vec![
            missing_unit(1, MetrologyQualificationStatusV1::NotSelected, false, false),
            missing_unit(2, MetrologyQualificationStatusV1::Unknown, true, true),
        ];
        let report = compute_qualified_observation_risk_v1(&input(units, assumptions(false))).unwrap();
        assert_eq!(report.worst_case_bounds.lower_bound, 0.0);
        assert_eq!(report.worst_case_bounds.upper_bound, 1.0);
        assert_eq!(report.worst_case_bounds.interval_width, 1.0);
    }
}
