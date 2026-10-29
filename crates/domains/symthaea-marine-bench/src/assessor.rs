// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::CLAIM_CEILING;
use crate::model::{
    DecisionUse, EvidenceClass, EvidenceValue, MarineFreightCase, ProvenanceRecord,
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
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
    /// Nominal observed quantity (for report context only).
    pub observed: Option<f64>,
    /// Nominal constraint quantity (for report context only).
    pub bound: Option<f64>,
    pub observed_lower: Option<f64>,
    pub observed_upper: Option<f64>,
    pub bound_lower: Option<f64>,
    pub bound_upper: Option<f64>,
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
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported case schema version: {version}")
            }
            Self::EmptyCaseId => write!(f, "case_id must not be empty"),
            Self::EmptyProvenanceId => write!(f, "provenance IDs must not be empty"),
            Self::EmptyProvenanceDescription(id) => {
                write!(f, "provenance description must not be empty for {id}")
            }
            Self::DuplicateProvenanceId(id) => write!(f, "duplicate provenance ID: {id}"),
            Self::ClaimCeilingMismatch => {
                write!(
                    f,
                    "case claim ceiling does not match this evaluator's frozen ceiling"
                )
            }
            Self::UnsupportedEvidenceClass(class) => {
                write!(
                    f,
                    "v2 accepts synthetic illustrations only; received evidence class {class}"
                )
            }
            Self::UnsupportedDecisionUse => write!(f, "v2 accepts not_for_decision cases only"),
        }
    }
}

impl std::error::Error for AssessmentError {}

/// Assess one declared case. Each finding is independent; no aggregate score
/// or overall PASS is generated. Unknown, malformed, or unproven inputs result
/// in an indeterminate finding rather than being replaced by zero.
pub fn assess(case: &MarineFreightCase) -> Result<AssessmentReport, AssessmentError> {
    if case.schema_version != "2.0.0" {
        return Err(AssessmentError::UnsupportedSchema(
            case.schema_version.clone(),
        ));
    }
    if case.case_id.trim().is_empty() {
        return Err(AssessmentError::EmptyCaseId);
    }
    if case.claim_ceiling != CLAIM_CEILING {
        return Err(AssessmentError::ClaimCeilingMismatch);
    }
    if case.evidence_class != EvidenceClass::SyntheticIllustration {
        return Err(AssessmentError::UnsupportedEvidenceClass(format!(
            "{:?}",
            case.evidence_class
        )));
    }
    if case.decision_use != DecisionUse::NotForDecision {
        return Err(AssessmentError::UnsupportedDecisionUse);
    }

    let sources = validate_provenance(&case.provenance)?;
    let demand = duty_cycle_energy_kwh(case, &sources);
    let usable_capacity = usable_battery_energy_kwh(case, &sources);
    let charge_energy = charge_energy_available_kwh(case, &sources);
    let charge_duration = charge_duration_h(case, &sources, &demand);

    let findings = vec![
        compare_le(
            "route_length_envelope",
            resolve(&case.vessel.length_overall_m, "m", &sources),
            resolve(&case.route.maximum_vessel_length_m, "m", &sources),
            "m",
            "vessel length overall <= route maximum vessel length",
        ),
        compare_le(
            "route_beam_envelope",
            resolve(&case.vessel.beam_m, "m", &sources),
            resolve(&case.route.maximum_vessel_beam_m, "m", &sources),
            "m",
            "vessel beam <= route maximum vessel beam",
        ),
        compare_le(
            "route_draft_envelope",
            resolve(&case.vessel.design_draft_m, "m", &sources),
            resolve(&case.route.maximum_vessel_draft_m, "m", &sources),
            "m",
            "design draft <= declared route draft limit",
        ),
        compare_le(
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
        ),
        compare_le(
            "berth_length_envelope",
            resolve(&case.vessel.length_overall_m, "m", &sources),
            resolve(&case.terminal.berth_length_m, "m", &sources),
            "m",
            "vessel length overall <= declared berth length",
        ),
        compare_le(
            "berth_beam_envelope",
            resolve(&case.vessel.beam_m, "m", &sources),
            resolve(&case.terminal.berth_beam_limit_m, "m", &sources),
            "m",
            "vessel beam <= declared berth beam limit",
        ),
        compare_le(
            "payload_capacity",
            resolve(&case.duty_cycle.required_payload_tonnes, "t", &sources),
            resolve(&case.vessel.cargo_capacity_tonnes, "t", &sources),
            "t",
            "required payload <= declared cargo capacity",
        ),
        compare_le(
            "battery_energy_per_duty_cycle",
            demand.clone(),
            usable_capacity,
            "kWh",
            concat!(
                "sum of four declared leg loads <= nominal battery energy × temperature factor × ",
                "end-of-life factor × (1 − reserve fraction)",
            ),
        ),
        compare_le(
            "shore_recharge_before_departure",
            demand.clone(),
            charge_energy,
            "kWh",
            concat!(
                "duty-cycle energy <= min(charger power, shore connection power) × charge window × ",
                "charging efficiency",
            ),
        ),
        compare_le(
            "service_turnaround_window",
            match (
                resolve(&case.duty_cycle.round_trip_duration_h, "h", &sources),
                charge_duration,
            ) {
                (Ok(mission_h), Ok(charge_h)) => {
                    mission_h.add(charge_h, "mission plus recharge time")
                }
                (Err(reason), _) => Err(reason),
                (_, Err(reason)) => Err(reason),
            },
            resolve(&case.terminal.available_service_window_h, "h", &sources),
            "h",
            "declared round-trip duration + idealized recharge duration <= declared service window",
        ),
    ];

    Ok(AssessmentReport {
        schema_version: "2.0.0".to_owned(),
        case_id: case.case_id.clone(),
        evidence_class: case.evidence_class.clone(),
        decision_use: case.decision_use.clone(),
        findings,
        claim_ceiling: CLAIM_CEILING.to_owned(),
        notes: vec![
            concat!(
                "Each finding uses conservative closed-interval arithmetic over declared bounds; ",
                "bounds do not imply a probability distribution or physical calibration.",
            )
            .to_owned(),
            concat!(
                "A WithinDeclaredBound result requires the entire observed interval to sit below ",
                "the entire constraint interval; overlapping intervals are indeterminate.",
            )
            .to_owned(),
            concat!(
                "Energy demand is supplied by the case; this evaluator does not derive ",
                "hull resistance or propulsion loads.",
            )
            .to_owned(),
            concat!(
                "Charge duration assumes constant delivered power and declared efficiency; taper, ",
                "thermal limits, grid dynamics, and degradation during the cycle are not modeled.",
            )
            .to_owned(),
            concat!(
                "No overall PASS, weighted score, design ranking, or operating authority is ",
                "produced.",
            )
            .to_owned(),
        ],
    })
}

fn validate_provenance(
    records: &[ProvenanceRecord],
) -> Result<BTreeMap<&str, &ProvenanceRecord>, AssessmentError> {
    let mut sources = BTreeMap::new();
    for record in records {
        if record.id.trim().is_empty() {
            return Err(AssessmentError::EmptyProvenanceId);
        }
        if record.description.trim().is_empty() {
            return Err(AssessmentError::EmptyProvenanceDescription(
                record.id.clone(),
            ));
        }
        if sources.insert(record.id.as_str(), record).is_some() {
            return Err(AssessmentError::DuplicateProvenanceId(record.id.clone()));
        }
    }
    Ok(sources)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Interval {
    nominal: f64,
    lower: f64,
    upper: f64,
}

impl Interval {
    fn add(self, other: Self, label: &str) -> Result<Self, String> {
        checked_interval(
            self.nominal + other.nominal,
            self.lower + other.lower,
            self.upper + other.upper,
            label,
        )
    }

    fn multiply_nonnegative(self, other: Self, label: &str) -> Result<Self, String> {
        checked_interval(
            self.nominal * other.nominal,
            self.lower * other.lower,
            self.upper * other.upper,
            label,
        )
    }

    fn complement_unit_interval(self, label: &str) -> Result<Self, String> {
        checked_interval(
            1.0 - self.nominal,
            1.0 - self.upper,
            1.0 - self.lower,
            label,
        )
    }
}

fn checked_interval(nominal: f64, lower: f64, upper: f64, label: &str) -> Result<Interval, String> {
    if !nominal.is_finite() || !lower.is_finite() || !upper.is_finite() {
        return Err(format!("{label} produced a non-finite interval"));
    }
    if lower < 0.0 || lower > nominal || nominal > upper {
        return Err(format!(
            "{label} produced an invalid interval (require 0 <= lower <= nominal <= upper)"
        ));
    }
    Ok(Interval {
        nominal,
        lower,
        upper,
    })
}

fn resolve(
    input: &EvidenceValue,
    expected_unit: &str,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<Interval, String> {
    match input {
        EvidenceValue::Unknown { reason } => Err(format!("unknown input: {}", reason.trim())),
        EvidenceValue::Known {
            value,
            lower_bound,
            upper_bound,
            unit,
            provenance_id,
        } => {
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
            checked_interval(*value, *lower_bound, *upper_bound, "input")
        }
    }
}

fn resolve_fraction(
    input: &EvidenceValue,
    label: &str,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<Interval, String> {
    let interval = resolve(input, "fraction", sources)?;
    if interval.upper > 1.0 {
        return Err(format!("{label} must have bounds in interval [0, 1]"));
    }
    Ok(interval)
}

fn sum_values(
    inputs: &[(&EvidenceValue, &str)],
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<Interval, String> {
    let mut total = Interval {
        nominal: 0.0,
        lower: 0.0,
        upper: 0.0,
    };
    for (input, unit) in inputs {
        total = total.add(resolve(input, unit, sources)?, "sum of declared inputs")?;
    }
    Ok(total)
}

fn duty_cycle_energy_kwh(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<Interval, String> {
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
) -> Result<Interval, String> {
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
    nominal
        .multiply_nonnegative(temperature, "battery × temperature derating")?
        .multiply_nonnegative(end_of_life, "battery × end-of-life derating")?
        .multiply_nonnegative(
            reserve.complement_unit_interval("reserve complement")?,
            "battery reserve-adjusted usable energy",
        )
}

fn effective_charge_rate_kw(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<Interval, String> {
    let charger = resolve(&case.energy_system.charger_power_kw, "kW", sources)?;
    let shore = resolve(&case.terminal.shore_connection_power_kw, "kW", sources)?;
    checked_interval(
        charger.nominal.min(shore.nominal),
        charger.lower.min(shore.lower),
        charger.upper.min(shore.upper),
        "effective charge power",
    )
}

fn charge_energy_available_kwh(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
) -> Result<Interval, String> {
    let effective_power = effective_charge_rate_kw(case, sources)?;
    let window = resolve(&case.terminal.available_charge_window_h, "h", sources)?;
    let efficiency = resolve_fraction(
        &case.terminal.charging_efficiency,
        "charging efficiency",
        sources,
    )?;
    effective_power
        .multiply_nonnegative(window, "power × charge window")?
        .multiply_nonnegative(efficiency, "available battery-side recharge energy")
}

fn charge_duration_h(
    case: &MarineFreightCase,
    sources: &BTreeMap<&str, &ProvenanceRecord>,
    demand: &Result<Interval, String>,
) -> Result<Interval, String> {
    let demand = *demand.as_ref().map_err(Clone::clone)?;
    let effective_power = effective_charge_rate_kw(case, sources)?;
    let efficiency = resolve_fraction(
        &case.terminal.charging_efficiency,
        "charging efficiency",
        sources,
    )?;
    let rate = effective_power.multiply_nonnegative(efficiency, "delivered charge rate")?;
    if rate.upper == 0.0 {
        return if demand.upper == 0.0 {
            Ok(Interval {
                nominal: 0.0,
                lower: 0.0,
                upper: 0.0,
            })
        } else {
            Err("no positive charge rate is available".to_owned())
        };
    }
    if rate.lower <= 0.0 {
        return Err(
            "recharge-time upper bound is unbounded because the delivered-rate lower bound is zero"
                .to_owned(),
        );
    }
    checked_interval(
        demand.nominal / rate.nominal,
        demand.lower / rate.upper,
        demand.upper / rate.lower,
        "idealized recharge duration",
    )
}

fn compare_le(
    metric: &str,
    observed: Result<Interval, String>,
    bound: Result<Interval, String>,
    unit: &str,
    calculation: &str,
) -> Finding {
    match (observed, bound) {
        (Ok(observed), Ok(bound)) => {
            let status = if observed.upper <= bound.lower {
                FindingStatus::WithinDeclaredBound
            } else if observed.lower > bound.upper {
                FindingStatus::ViolatesDeclaredBound
            } else {
                FindingStatus::Indeterminate
            };
            let reason = match status {
                FindingStatus::WithinDeclaredBound => concat!(
                    "entire observed interval is within the entire declared bound interval; ",
                    "this is not physical qualification",
                )
                .to_owned(),
                FindingStatus::ViolatesDeclaredBound => {
                    "even the lowest observed value exceeds the highest declared bound".to_owned()
                }
                FindingStatus::Indeterminate => concat!(
                    "input intervals overlap or cross; available bounds do not establish ",
                    "feasibility or violation",
                )
                .to_owned(),
            };
            Finding {
                metric: metric.to_owned(),
                status,
                observed: Some(observed.nominal),
                bound: Some(bound.nominal),
                observed_lower: Some(observed.lower),
                observed_upper: Some(observed.upper),
                bound_lower: Some(bound.lower),
                bound_upper: Some(bound.upper),
                unit: unit.to_owned(),
                calculation: calculation.to_owned(),
                reason,
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
                observed_lower: None,
                observed_upper: None,
                bound_lower: None,
                bound_upper: None,
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
        include_str!("../fixtures/marine_freight_reference_case_v2.synthetic.json");

    fn case() -> MarineFreightCase {
        serde_json::from_str(FIXTURE).expect("the checked-in reference fixture must parse")
    }

    fn finding<'a>(report: &'a AssessmentReport, metric: &str) -> &'a Finding {
        report
            .findings
            .iter()
            .find(|f| f.metric == metric)
            .expect("finding exists")
    }

    fn set_number(input: &mut EvidenceValue, value: f64) {
        match input {
            EvidenceValue::Known {
                value: slot,
                lower_bound,
                upper_bound,
                ..
            } => {
                *slot = value;
                *lower_bound = value;
                *upper_bound = value;
            }
            EvidenceValue::Unknown { .. } => panic!("test expected a known input"),
        }
    }

    #[test]
    fn synthetic_reference_case_matches_hand_calculated_baseline() {
        let case = case();
        let report = assess(&case).expect("valid synthetic case");
        assert_eq!(report.case_id, "marine-freight-synthetic-screen-v2");
        assert_eq!(report.findings.len(), 10);
        assert!(
            report
                .findings
                .iter()
                .all(|f| f.status == FindingStatus::WithinDeclaredBound)
        );

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
        assert_eq!(
            finding(&report, "payload_capacity").status,
            FindingStatus::ViolatesDeclaredBound
        );
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
        let source = case
            .provenance
            .iter_mut()
            .find(|p| p.id == "synthetic-duty-cycle")
            .expect("fixture record");
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

    fn set_range(input: &mut EvidenceValue, nominal: f64, lower: f64, upper: f64) {
        match input {
            EvidenceValue::Known {
                value,
                lower_bound,
                upper_bound,
                ..
            } => {
                *value = nominal;
                *lower_bound = lower;
                *upper_bound = upper;
            }
            EvidenceValue::Unknown { .. } => panic!("test expected a known input"),
        }
    }

    #[test]
    fn overlapping_energy_intervals_are_indeterminate_not_nominal_pass() {
        let mut case = case();
        set_range(
            &mut case.duty_cycle.loaded_leg_propulsion_kwh,
            1850.0,
            1700.0,
            2500.0,
        );
        let report = assess(&case).expect("synthetic bounded case");
        let energy = finding(&report, "battery_energy_per_duty_cycle");
        assert_eq!(energy.status, FindingStatus::Indeterminate);
        assert!(
            energy.observed_lower.expect("lower bound")
                < energy.bound_upper.expect("upper capacity")
        );
        assert!(
            energy.observed_upper.expect("upper bound")
                > energy.bound_lower.expect("lower capacity")
        );
    }

    #[test]
    fn fully_separated_input_ranges_prove_robust_violation() {
        let mut case = case();
        set_range(
            &mut case.duty_cycle.loaded_leg_propulsion_kwh,
            2500.0,
            2400.0,
            2600.0,
        );
        let report = assess(&case).expect("synthetic bounded case");
        let energy = finding(&report, "battery_energy_per_duty_cycle");
        assert_eq!(energy.status, FindingStatus::ViolatesDeclaredBound);
        assert!(
            energy.observed_lower.expect("lower demand")
                > energy.bound_upper.expect("upper capacity")
        );
    }

    #[test]
    fn malformed_interval_does_not_get_optimistic_default() {
        let mut case = case();
        set_range(&mut case.vessel.length_overall_m, 80.0, 82.0, 81.0);
        let report = assess(&case).expect("range error is a finding, not a panic");
        assert_eq!(
            finding(&report, "route_length_envelope").status,
            FindingStatus::Indeterminate
        );
    }

    #[test]
    fn uncertainty_ranges_are_propagated_through_under_keel_clearance() {
        let mut case = case();
        set_range(&mut case.vessel.design_draft_m, 2.2, 2.1, 2.4);
        set_range(
            &mut case.route.minimum_under_keel_clearance_m,
            0.5,
            0.4,
            0.7,
        );
        set_range(&mut case.route.available_water_depth_m, 3.2, 3.0, 3.25);
        let report = assess(&case).expect("synthetic bounded case");
        assert_eq!(
            finding(&report, "under_keel_clearance").status,
            FindingStatus::Indeterminate
        );
    }
    #[test]
    fn interval_comparator_matches_frozen_oracle_for_ten_thousand_cases() {
        for i in 0_u32..10_000 {
            let observed_lower = (i.wrapping_mul(37) % 101) as f64;
            let observed_upper = observed_lower + (i.wrapping_mul(11) % 37) as f64;
            let bound_lower = (i.wrapping_mul(17) % 101) as f64;
            let bound_upper = bound_lower + (i.wrapping_mul(23) % 37) as f64;
            let observed = Interval {
                nominal: (observed_lower + observed_upper) / 2.0,
                lower: observed_lower,
                upper: observed_upper,
            };
            let bound = Interval {
                nominal: (bound_lower + bound_upper) / 2.0,
                lower: bound_lower,
                upper: bound_upper,
            };
            let expected = if observed_upper <= bound_lower {
                FindingStatus::WithinDeclaredBound
            } else if observed_lower > bound_upper {
                FindingStatus::ViolatesDeclaredBound
            } else {
                FindingStatus::Indeterminate
            };
            let actual = compare_le(
                "interval_oracle",
                Ok(observed),
                Ok(bound),
                "unit",
                "fixed rule",
            );
            assert_eq!(actual.status, expected, "case index {i}");
            if actual.status == FindingStatus::WithinDeclaredBound {
                assert!(actual.observed_upper.unwrap() <= actual.bound_lower.unwrap());
            }
            if actual.status == FindingStatus::ViolatesDeclaredBound {
                assert!(actual.observed_lower.unwrap() > actual.bound_upper.unwrap());
            }
        }
    }
}
