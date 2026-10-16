// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::model::{
    DecisionUse, EvidenceClass, EvidenceValue, MarineFreightCase, ProvenanceRecord,
};
use crate::CLAIM_CEILING;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FindingStatus {
    WithinDeclaredBound,
    ViolatesDeclaredBound,
    Indeterminate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Finding {
    pub metric: String,
    pub status: FindingStatus,
    pub observed: Option<f64>,
    pub bound: Option<f64>,
    pub unit: String,
    pub calculation: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssessmentReport {
    pub schema_version: String,
    pub case_id: String,
    pub evidence_class: EvidenceClass,
    pub decision_use: DecisionUse,
    pub findings: Vec<Finding>,
    pub claim_ceiling: String,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssessmentError {
    UnsupportedSchema(String),
    EmptyCaseId,
    EmptyProvenanceId,
    EmptyProvenanceDescription(String),
    DuplicateProvenanceId(String),
    ClaimCeilingMismatch,
    UnsupportedEvidenceClass(String),
    UnsupportedDecisionUse,
}

impl fmt::Display for AssessmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(version) => write!(f, "unsupported case schema version: {version}"),
            Self::EmptyCaseId => write!(f, "case_id must not be empty"),
            Self::EmptyProvenanceId => write!(f, "provenance IDs must not be empty"),
            Self::EmptyProvenanceDescription(id) => {
                write!(f, "provenance description must not be empty for {id}")
            }
            Self::DuplicateProvenanceId(id) => write!(f, "duplicate provenance ID: {id}"),
            Self::ClaimCeilingMismatch => write!(f, "case claim ceiling does not match this evaluator's frozen ceiling"),
            Self::UnsupportedEvidenceClass(class) => write!(f, "v1 accepts synthetic illustrations only; received evidence class {class}"),
            Self::UnsupportedDecisionUse => write!(f, "v1 accepts not_for_decision cases only"),
        }
    }
}

impl std::error::Error for AssessmentError {}

/// Assess one declared case. Each finding is independent; no aggregate score
/// or overall PASS is generated. Unknown, malformed, or unproven inputs result
/// in an indeterminate finding rather than being replaced by zero.
pub fn assess(case: &MarineFreightCase) -> Result<AssessmentReport, AssessmentError> {
    if case.schema_version != "1.0.0" {
        return Err(AssessmentError::UnsupportedSchema(case.schema_version.clone()));
    }
    if case.case_id.trim().is_empty() {
        return Err(AssessmentError::EmptyCaseId);
    }
    if case.claim_ceiling != CLAIM_CEILING {
        return Err(AssessmentError::ClaimCeilingMismatch);
    }
    if case.evidence_class != EvidenceClass::SyntheticIllustration {
        return Err(AssessmentError::UnsupportedEvidenceClass(format!("{:?}", case.evidence_class)));
    }
    if case.decision_use != DecisionUse::NotForDecision {
        return Err(AssessmentError::UnsupportedDecisionUse);
    }

    let sources = validate_provenance(&case.provenance)?;
    let demand = duty_cycle_energy_kwh(case, &sources);
    let usable_capacity = usable_battery_energy_kwh(case, &sources);
    let charge_energy = charge_energy_available_kwh(case, &sources);
    let charge_duration = charge_duration_h(case, &sources, &demand);

    let mut findings = Vec::with_capacity(10);
    findings.push(compare_le(
        "route_length_envelope",
        resolve(&case.vessel.length_overall_m, "m", &sources),
        resolve(&case.route.maximum_vessel_length_m, "m", &sources),
        "m",
        "vessel length overall <= route maximum vessel length",
    ));
    findings.push(compare_le(
        "route_beam_envelope",
        resolve(&case.vessel.beam_m, "m", &sources),
        resolve(&case.route.maximum_vessel_beam_m, "m", &sources),
        "m",
        "vessel beam <= route maximum vessel beam",
    ));
    findings.push(compare_le(
        "route_draft_envelope",
        resolve(&case.vessel.design_draft_m, "m", &sources),
        resolve(&case.route.maximum_vessel_draft_m, "m", &sources),
        "m",
        "design draft <= declared route draft limit",
    ));
    findings.push(compare_le(
        "under_keel_clearance",
        sum_values(
            &[
                (&case.vessel.design_draft_m, "m"),
                (&case.route.minimum_under_keel_clearance_m, "m"),
            ],
            &sources,
        ),
        resolve(&case.route.available_water_depth_m, "m", &sources),
        "m",
        "design draft + required under-keel clearance <= declared water depth",
    ));
    findings.push(compare_le(
        "berth_length_envelope",
        resolve(&case.vessel.length_overall_m, "m", &sources),
        resolve(&case.terminal.berth_length_m, "m", &sources),
        "m",
        "vessel length overall <= declared berth length",
    ));
    findings.push(compare_le(
        "berth_beam_envelope",
        resolve(&case.vessel.beam_m, "m", &sources),
        resolve(&case.terminal.berth_beam_limit_m, "m", &sources),
        "m",
        "vessel beam <= declared berth beam limit",
    ));
    findings.push(compare_le(
        "payload_capacity",
        resolve(&case.duty_cycle.required_payload_tonnes, "t", &sources),
        resolve(&case.vessel.cargo_capacity_tonnes, "t", &sources),
        "t",
        "required payload <= declared cargo capacity",
    ));
    findings.push(compare_le(
        "battery_energy_per_duty_cycle",
        demand.clone(),
        usable_capacity,
        "kWh",
        "sum of four declared leg loads <= nominal battery energy × temperature factor × end-of-life factor × (1 − reserve fraction)",
    ));
    findings.push(compare_le(
        "shore_recharge_before_departure",
        demand.clone(),
        charge_energy,
        "kWh",
        "duty-cycle energy <= min(charger power, shore connection power) × charge window × charging efficiency",
    ));
    findings.push(compare_le(
        "service_turnaround_window",
        match (resolve(&case.duty_cycle.round_trip_duration_h, "h", &sources), charge_duration) {
            (Ok(mission_h), Ok(charge_h)) => finite_result(mission_h + charge_h, "mission plus recharge time"),
            (Err(reason), _) => Err(reason),
            (_, Err(reason)) => Err(reason),
        },
        resolve(&case.terminal.available_service_window_h, "h", &sources),
        "h",
        "declared round-trip duration + idealized recharge duration <= declared service window",
    ));

    Ok(AssessmentReport {
        schema_version: "1.0.0".to_owned(),
        case_id: case.case_id.clone(),
        evidence_class: case.evidence_class.clone(),
        decision_use: case.decision_use.clone(),
        findings,
        claim_ceiling: CLAIM_CEILING.to_owned(),
        notes: vec![
            "Each finding is a screening calculation over declared inputs, not a physical qualification.".to_owned(),
            "Energy demand is supplied by the case; this evaluator does not derive hull resistance or propulsion loads.".to_owned(),
            "Charge duration assumes constant delivered power and a constant declared efficiency; taper, thermal limits, grid dynamics, and degradation during the cycle are not modeled.".to_owned(),
            "No overall PASS, weighted score, design ranking, or operating authority is produced.".to_owned(),
        ],
    })
}

fn validate_provenance<'a>(
    records: &'a [ProvenanceRecord],
) -> Result<BTreeMap<&'a str, &'a ProvenanceRecord>, AssessmentError> {
    let mut sources = BTreeMap::new();
    for record in records {
        if record.id.trim().is_empty() {
            return Err(AssessmentError::EmptyProvenanceId);
        }
        if record.description.trim().is_empty() {
            return Err(AssessmentError::EmptyProvenanceDescription(record.id.clone()));
        }
        if sources.insert(record.id.as_str(), record).is_some() {
            return Err(AssessmentError::DuplicateProvenanceId(record.id.clone()));
        }
    }
    Ok(sources)
}

fn resolve(
    input: &EvidenceValue,
    expected_unit: &str,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<f64, String> {
    match input {
        EvidenceValue::Unknown { reason } => Err(format!("unknown input: {}", reason.trim())),
        EvidenceValue::Known {
            value,
            unit,
            provenance_id,
        } => {
            if !value.is_finite() {
                return Err("input value is not finite".to_owned());
            }
            if *value < 0.0 {
                return Err("input value must not be negative".to_owned());
            }
            if unit != expected_unit {
                return Err(format!(
                    "unit mismatch: expected {expected_unit}, received {unit}"
                ));
            }
            if provenance_id.trim().is_empty() {
                return Err("missing provenance reference".to_owned());
            }
            let Some(source) = sources.get(provenance_id.as_str()) else {
                return Err(format!("unresolved provenance reference: {provenance_id}"));
            };
            if source.class == EvidenceClass::Unknown {
                return Err(format!("provenance class is unknown: {provenance_id}"));
            }
            Ok(*value)
        }
    }
}

fn resolve_fraction(
    input: &EvidenceValue,
    label: &str,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<f64, String> {
    let value = resolve(input, "fraction", sources)?;
    if value > 1.0 {
        return Err(format!("{label} must be in interval [0, 1]"));
    }
    Ok(value)
}

fn sum_values(
    inputs: &[(&EvidenceValue, &str)],
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<f64, String> {
    let values = inputs
        .iter()
        .map(|(input, unit)| resolve(input, unit, sources))
        .collect::<Result<Vec<_>, _>>()?;
    finite_result(values.iter().sum(), "sum of declared inputs")
}

fn duty_cycle_energy_kwh(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<f64, String> {
    sum_values(
        &[
            (&case.duty_cycle.loaded_leg_propulsion_kwh, "kWh"),
            (&case.duty_cycle.loaded_leg_auxiliary_kwh, "kWh"),
            (&case.duty_cycle.return_leg_propulsion_kwh, "kWh"),
            (&case.duty_cycle.return_leg_auxiliary_kwh, "kWh"),
        ],
        sources,
    )
}

fn usable_battery_energy_kwh(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<f64, String> {
    let nominal = resolve(&case.energy_system.nominal_battery_kwh, "kWh", sources)?;
    let temperature = resolve_fraction(
        &case.energy_system.temperature_capacity_factor,
        "temperature capacity factor",
        sources,
    )?;
    let end_of_life = resolve_fraction(
        &case.energy_system.end_of_life_capacity_factor,
        "end-of-life capacity factor",
        sources,
    )?;
    let reserve = resolve_fraction(
        &case.energy_system.reserve_fraction,
        "reserve fraction",
        sources,
    )?;
    finite_result(
        nominal * temperature * end_of_life * (1.0 - reserve),
        "derated usable battery energy",
    )
}

fn effective_charge_rate_kw(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<f64, String> {
    let charger = resolve(&case.energy_system.charger_power_kw, "kW", sources)?;
    let shore = resolve(&case.terminal.shore_connection_power_kw, "kW", sources)?;
    finite_result(charger.min(shore), "effective charge power")
}

fn charge_energy_available_kwh(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<f64, String> {
    let effective_power = effective_charge_rate_kw(case, sources)?;
    let window = resolve(&case.terminal.available_charge_window_h, "h", sources)?;
    let efficiency = resolve_fraction(
        &case.terminal.charging_efficiency,
        "charging efficiency",
        sources,
    )?;
    finite_result(effective_power * window * efficiency, "available battery-side recharge energy")
}

fn charge_duration_h(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
    demand: &Result<f64, String>,
) -> Result<f64, String> {
    let demand = *demand.as_ref().map_err(Clone::clone)?;
    let effective_power = effective_charge_rate_kw(case, sources)?;
    let efficiency = resolve_fraction(
        &case.terminal.charging_efficiency,
        "charging efficiency",
        sources,
    )?;
    let rate = effective_power * efficiency;
    if rate == 0.0 {
        return if demand == 0.0 {
            Ok(0.0)
        } else {
            Err("no positive charge rate is available".to_owned())
        };
    }
    finite_result(demand / rate, "idealized recharge duration")
}

fn finite_result(value: f64, label: &str) -> Result<f64, String> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(format!("{label} produced a non-finite or negative result"))
    }
}

fn compare_le(
    metric: &str,
    observed: Result<f64, String>,
    bound: Result<f64, String>,
    unit: &str,
    calculation: &str,
) -> Finding {
    match (observed, bound) {
        (Ok(observed), Ok(bound)) => {
            let within = observed <= bound;
            Finding {
                metric: metric.to_owned(),
                status: if within {
                    FindingStatus::WithinDeclaredBound
                } else {
                    FindingStatus::ViolatesDeclaredBound
                },
                observed: Some(observed),
                bound: Some(bound),
                unit: unit.to_owned(),
                calculation: calculation.to_owned(),
                reason: if within {
                    "arithmetic comparison is within the declared bound; this is not physical qualification".to_owned()
                } else {
                    "observed value exceeds the declared bound".to_owned()
                },
            }
        }
        (left, right) => {
            let reasons = [
                left.err().map(|e| format!("observed value: {e}")),
                right.err().map(|e| format!("bound: {e}")),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("; ");
            Finding {
                metric: metric.to_owned(),
                status: FindingStatus::Indeterminate,
                observed: None,
                bound: None,
                unit: unit.to_owned(),
                calculation: calculation.to_owned(),
                reason: reasons,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EvidenceValue;

    const FIXTURE: &str =
        include_str!("../fixtures/marine_freight_reference_case_v1.synthetic.json");

    fn case() -> MarineFreightCase {
        serde_json::from_str(FIXTURE).expect("the checked-in reference fixture must parse")
    }

    fn finding<'a>(report: &'a AssessmentReport, metric: &str) -> &'a Finding {
        report.findings.iter().find(|f| f.metric == metric).expect("finding exists")
    }

    fn set_number(input: &mut EvidenceValue, value: f64) {
        match input {
            EvidenceValue::Known { value: slot, .. } => *slot = value,
            EvidenceValue::Unknown { .. } => panic!("test expected a known input"),
        }
    }

    #[test]
    fn synthetic_reference_case_matches_hand_calculated_baseline() {
        let case = case();
        let report = assess(&case).expect("valid synthetic case");
        assert_eq!(report.case_id, "marine-freight-synthetic-screen-v1");
        assert_eq!(report.findings.len(), 10);
        assert!(report.findings.iter().all(|f| f.status == FindingStatus::WithinDeclaredBound));

        let energy = finding(&report, "battery_energy_per_duty_cycle");
        assert_eq!(energy.observed, Some(3235.0));
        assert!((energy.bound.expect("known battery bound") - 3511.35).abs() < 1e-9);

        let charge = finding(&report, "shore_recharge_before_departure");
        assert_eq!(charge.observed, Some(3235.0));
        assert_eq!(charge.bound, Some(4416.0));

        let turnaround = finding(&report, "service_turnaround_window");
        let expected_h = 15.0 + 3235.0 / (600.0 * 0.92);
        assert!((turnaround.observed.expect("known turnaround") - expected_h).abs() < 1e-12);
        assert_eq!(turnaround.bound, Some(24.0));
        assert!(!report.findings.iter().any(|f| f.metric == "overall_pass"));
        assert_eq!(report.decision_use, DecisionUse::NotForDecision);
    }

    #[test]
    fn insufficient_derated_battery_energy_is_a_violation() {
        let mut case = case();
        set_number(&mut case.energy_system.nominal_battery_kwh, 4000.0);
        let report = assess(&case).expect("valid input case");
        assert_eq!(
            finding(&report, "battery_energy_per_duty_cycle").status,
            FindingStatus::ViolatesDeclaredBound
        );
    }

    #[test]
    fn grid_bottleneck_can_make_recharge_infeasible() {
        let mut case = case();
        set_number(&mut case.terminal.shore_connection_power_kw, 100.0);
        let report = assess(&case).expect("valid input case");
        assert_eq!(
            finding(&report, "shore_recharge_before_departure").status,
            FindingStatus::ViolatesDeclaredBound
        );
        assert_eq!(
            finding(&report, "service_turnaround_window").status,
            FindingStatus::ViolatesDeclaredBound
        );
    }

    #[test]
    fn route_limit_violation_is_not_hidden_by_other_findings() {
        let mut case = case();
        set_number(&mut case.route.maximum_vessel_length_m, 70.0);
        let report = assess(&case).expect("valid input case");
        assert_eq!(
            finding(&report, "route_length_envelope").status,
            FindingStatus::ViolatesDeclaredBound
        );
    }

    #[test]
    fn missing_water_depth_is_indeterminate_not_zero() {
        let mut case = case();
        case.route.available_water_depth_m = EvidenceValue::Unknown {
            reason: "no sourced route depth".to_owned(),
        };
        let report = assess(&case).expect("unknowns are admissible inputs");
        let result = finding(&report, "under_keel_clearance");
        assert_eq!(result.status, FindingStatus::Indeterminate);
        assert!(result.observed.is_none());
        assert!(result.bound.is_none());
    }

    #[test]
    fn missing_provenance_makes_the_affected_finding_indeterminate() {
        let mut case = case();
        if let EvidenceValue::Known { provenance_id, .. } = &mut case.vessel.length_overall_m {
            *provenance_id = "missing-record".to_owned();
        }
        let report = assess(&case).expect("referential gaps stay visible in findings");
        assert_eq!(
            finding(&report, "route_length_envelope").status,
            FindingStatus::Indeterminate
        );
    }

    #[test]
    fn unit_mismatch_is_indeterminate_not_silently_converted() {
        let mut case = case();
        if let EvidenceValue::Known { unit, .. } = &mut case.vessel.design_draft_m {
            *unit = "km".to_owned();
        }
        let report = assess(&case).expect("unit mismatches become findings");
        assert_eq!(
            finding(&report, "route_draft_envelope").status,
            FindingStatus::Indeterminate
        );
        assert_eq!(
            finding(&report, "under_keel_clearance").status,
            FindingStatus::Indeterminate
        );
    }

    #[test]
    fn overloaded_payload_is_a_separate_violation() {
        let mut case = case();
        set_number(&mut case.duty_cycle.required_payload_tonnes, 1000.0);
        let report = assess(&case).expect("valid input case");
        assert_eq!(finding(&report, "payload_capacity").status, FindingStatus::ViolatesDeclaredBound);
    }

    #[test]
    fn unknown_energy_is_never_treated_as_zero() {
        let mut case = case();
        case.duty_cycle.loaded_leg_propulsion_kwh = EvidenceValue::Unknown {
            reason: "no load estimate for the loaded leg".to_owned(),
        };
        let report = assess(&case).expect("unknown inputs are reportable");
        assert_eq!(
            finding(&report, "battery_energy_per_duty_cycle").status,
            FindingStatus::Indeterminate
        );
        assert_eq!(
            finding(&report, "shore_recharge_before_departure").status,
            FindingStatus::Indeterminate
        );
        assert_eq!(
            finding(&report, "service_turnaround_window").status,
            FindingStatus::Indeterminate
        );
    }

    #[test]
    fn unknown_provenance_class_cannot_support_a_numeric_finding() {
        let mut case = case();
        let source = case.provenance.iter_mut().find(|p| p.id == "synthetic-duty-cycle").expect("fixture record");
        source.class = EvidenceClass::Unknown;
        let report = assess(&case).expect("unknown provenance is indeterminate");
        assert_eq!(
            finding(&report, "battery_energy_per_duty_cycle").status,
            FindingStatus::Indeterminate
        );
    }

    #[test]
    fn duplicate_provenance_ids_reject_the_case() {
        let mut case = case();
        case.provenance.push(case.provenance[0].clone());
        assert!(matches!(
            assess(&case),
            Err(AssessmentError::DuplicateProvenanceId(_))
        ));
    }

    #[test]
    fn measured_case_is_rejected_until_uncertainty_and_bias_bounds_exist() {
        let mut case = case();
        case.evidence_class = EvidenceClass::Measured;
        assert!(matches!(
            assess(&case),
            Err(AssessmentError::UnsupportedEvidenceClass(_))
        ));
    }

    #[test]
    fn decision_use_cannot_be_promoted_in_v1() {
        let mut case = case();
        case.decision_use = DecisionUse::ResearchScreeningOnly;
        assert_eq!(assess(&case), Err(AssessmentError::UnsupportedDecisionUse));
    }

    #[test]
    fn claim_ceiling_cannot_be_escalated_by_input_data() {
        let mut case = case();
        case.claim_ceiling = "vessel is safe for operation".to_owned();
        assert_eq!(assess(&case), Err(AssessmentError::ClaimCeilingMismatch));
    }

    #[test]
    fn reserve_factor_above_one_is_indeterminate() {
        let mut case = case();
        set_number(&mut case.energy_system.reserve_fraction, 1.1);
        let report = assess(&case).expect("invalid factor is surfaced as an indeterminate finding");
        assert_eq!(
            finding(&report, "battery_energy_per_duty_cycle").status,
            FindingStatus::Indeterminate
        );
    }
}
