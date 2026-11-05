//! Dependency-free, auditable storage-feasibility reference kernel.
//!
//! This is a synthetic engineering oracle for physical and arithmetic invariants.
//! It is not a SAWEM market-clearing implementation, a legal interpretation of
//! the Market Code, a dispatch controller, or evidence of market eligibility.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

const EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperatingMode {
    Standstill,
    Charging,
    Discharging,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReserveProduct {
    Regulating,
    Instantaneous,
    TenMinute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReserveDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReserveCommitment {
    pub product: ReserveProduct,
    pub direction: ReserveDirection,
    pub mw: f64,
    /// Asset response evidence measured from the scheduled operating point.
    pub start_response_seconds: Option<f64>,
    pub full_activation_seconds: Option<f64>,
    /// Sustained activation duration required by the contract / system profile.
    /// SAWEM Rev 2.3 does not define this for every reserve product.
    pub sustain_duration_seconds: Option<f64>,
    /// Independently demonstrated asset sustain capability, not a nameplate claim.
    pub proven_sustain_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketReserveRequirement {
    pub maximum_start_response_seconds: Option<f64>,
    pub maximum_full_activation_seconds: Option<f64>,
    pub minimum_sustain_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveQualification {
    Qualified,
    Infeasible,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReserveAssessment {
    pub qualification: ReserveQualification,
    pub committed_up_mw: f64,
    pub committed_down_mw: f64,
    pub available_up_mw: f64,
    pub available_down_mw: f64,
    pub indeterminate_reasons: Vec<&'static str>,
    pub violations: Vec<Violation>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionTimes {
    pub standstill_to_charging_seconds: f64,
    pub standstill_to_discharging_seconds: f64,
    pub charging_to_discharging_seconds: f64,
    pub discharging_to_charging_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssetLimits {
    pub max_charge_mw: f64,
    pub min_stable_charge_mw: f64,
    pub max_discharge_mw: f64,
    pub min_energy_mwh: f64,
    pub max_energy_mwh: f64,
    /// Symmetric efficiency assumption: charge and discharge efficiencies are
    /// each sqrt(round_trip_efficiency). Replace with separate measured values
    /// when the source contract provides them.
    pub round_trip_efficiency: f64,
    pub transitions: TransitionTimes,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DispatchStep {
    pub duration_hours: f64,
    pub charge_mw: f64,
    pub discharge_mw: f64,
    /// Declared downtime at the start of this interval for a mode transition.
    pub transition_seconds_before_action: f64,
    pub reserves: &'static [ReserveCommitment],
    pub network_export_limit_mw: f64,
    pub network_import_limit_mw: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Violation {
    SimultaneousChargeAndDischarge,
    ChargePowerExceeded,
    DischargePowerExceeded,
    BelowMinimumStableCharging,
    TransitionTimeInsufficient,
    TransitionTimeExceedsInterval,
    ExportLimitExceeded,
    ImportLimitExceeded,
    ReservePowerHeadroomExceeded,
    ReserveEnergyHeadroomExceeded,
    ReserveTargetBelowMinimumStableCharge,
    ReserveResponseDeadlineExceeded,
    ReserveSustainInsufficient,
    UnsupportedReserveProductDirection,
    InvalidReserveValue,
    DuplicateReserveProductDirection,
    EnergyBelowMinimum,
    EnergyAboveMaximum,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    pub index: usize,
    pub requested_mode: OperatingMode,
    pub achieved_mode: OperatingMode,
    pub energy_before_mwh: f64,
    /// Feasible state carried into the next interval. Hard-invalid actions do not commit.
    pub energy_after_mwh: f64,
    /// Candidate state before hard-constraint commit/rejection.
    pub candidate_energy_after_mwh: f64,
    pub net_export_mw: f64,
    pub transition_seconds_used: f64,
    pub reserve_up_mw: f64,
    pub reserve_down_mw: f64,
    pub reserve_assessment: ReserveAssessment,
    pub violations: Vec<Violation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchResult {
    pub feasible: bool,
    pub initial_energy_mwh: f64,
    pub final_energy_mwh: f64,
    pub steps: Vec<StepResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputError {
    NonFiniteValue(&'static str),
    NegativeValue(&'static str),
    InvalidEnergyBounds,
    InvalidChargingBounds,
    InvalidEfficiency,
    InvalidInitialEnergy,
    InvalidSevenDayDeclarationCount,
    IntegerOverflow,
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteValue(field) => write!(f, "{field} must be finite"),
            Self::NegativeValue(field) => write!(f, "{field} must be nonnegative"),
            Self::InvalidEnergyBounds => write!(f, "minimum energy must be below maximum energy"),
            Self::InvalidChargingBounds => write!(
                f,
                "minimum stable charge must not exceed maximum charge power"
            ),
            Self::InvalidEfficiency => write!(f, "round-trip efficiency must be in (0, 1]"),
            Self::InvalidInitialEnergy => write!(f, "initial energy is outside asset limits"),
            Self::InvalidSevenDayDeclarationCount => {
                write!(f, "exactly seven daily energy declarations are required")
            }
            Self::IntegerOverflow => write!(f, "integer arithmetic overflow"),
        }
    }
}

impl Error for InputError {}

impl AssetLimits {
    pub fn validate(self) -> Result<(), InputError> {
        let positive = [
            ("max_charge_mw", self.max_charge_mw),
            ("max_discharge_mw", self.max_discharge_mw),
            ("max_energy_mwh", self.max_energy_mwh),
        ];
        for (field, value) in positive {
            if !value.is_finite() {
                return Err(InputError::NonFiniteValue(field));
            }
            if value <= 0.0 {
                return Err(InputError::NegativeValue(field));
            }
        }

        let nonnegative = [
            ("min_stable_charge_mw", self.min_stable_charge_mw),
            ("min_energy_mwh", self.min_energy_mwh),
            (
                "standstill_to_charging_seconds",
                self.transitions.standstill_to_charging_seconds,
            ),
            (
                "standstill_to_discharging_seconds",
                self.transitions.standstill_to_discharging_seconds,
            ),
            (
                "charging_to_discharging_seconds",
                self.transitions.charging_to_discharging_seconds,
            ),
            (
                "discharging_to_charging_seconds",
                self.transitions.discharging_to_charging_seconds,
            ),
        ];
        for (field, value) in nonnegative {
            if !value.is_finite() {
                return Err(InputError::NonFiniteValue(field));
            }
            if value < 0.0 {
                return Err(InputError::NegativeValue(field));
            }
        }

        if self.min_stable_charge_mw > self.max_charge_mw + EPSILON {
            return Err(InputError::InvalidChargingBounds);
        }
        if self.min_energy_mwh < 0.0 || self.min_energy_mwh >= self.max_energy_mwh {
            return Err(InputError::InvalidEnergyBounds);
        }
        if !self.round_trip_efficiency.is_finite()
            || self.round_trip_efficiency <= 0.0
            || self.round_trip_efficiency > 1.0
        {
            return Err(InputError::InvalidEfficiency);
        }
        Ok(())
    }
}

fn mode_for(charge_mw: f64, discharge_mw: f64) -> OperatingMode {
    if charge_mw > EPSILON {
        OperatingMode::Charging
    } else if discharge_mw > EPSILON {
        OperatingMode::Discharging
    } else {
        OperatingMode::Standstill
    }
}

fn required_transition_seconds(
    previous: OperatingMode,
    next: OperatingMode,
    times: TransitionTimes,
) -> f64 {
    match (previous, next) {
        (OperatingMode::Standstill, OperatingMode::Charging) => {
            times.standstill_to_charging_seconds
        }
        (OperatingMode::Standstill, OperatingMode::Discharging) => {
            times.standstill_to_discharging_seconds
        }
        (OperatingMode::Charging, OperatingMode::Discharging) => {
            times.charging_to_discharging_seconds
        }
        (OperatingMode::Discharging, OperatingMode::Charging) => {
            times.discharging_to_charging_seconds
        }
        _ => 0.0,
    }
}

/// SAWEM Market Code Rev 2.3 product response requirements.
///
/// The Code specifies no minimum sustain duration for regulating or ten-minute
/// reserve in these definitions. That remains an external operator/contract
/// input; it is deliberately not inferred here.
pub fn sawem_rev_2_3_reserve_requirement(
    product: ReserveProduct,
    direction: ReserveDirection,
) -> Result<MarketReserveRequirement, Violation> {
    match (product, direction) {
        (ReserveProduct::Regulating, ReserveDirection::Up)
        | (ReserveProduct::Regulating, ReserveDirection::Down) => {
            Ok(MarketReserveRequirement {
                maximum_start_response_seconds: Some(10.0),
                maximum_full_activation_seconds: Some(600.0),
                minimum_sustain_seconds: None,
            })
        }
        (ReserveProduct::Instantaneous, ReserveDirection::Up) => {
            Ok(MarketReserveRequirement {
                maximum_start_response_seconds: None,
                maximum_full_activation_seconds: Some(10.0),
                minimum_sustain_seconds: Some(600.0),
            })
        }
        (ReserveProduct::TenMinute, ReserveDirection::Up) => {
            Ok(MarketReserveRequirement {
                maximum_start_response_seconds: None,
                maximum_full_activation_seconds: Some(600.0),
                minimum_sustain_seconds: None,
            })
        }
        _ => Err(Violation::UnsupportedReserveProductDirection),
    }
}

fn operating_point_for_net_power(
    net_export_mw: f64,
    min_stable_charge_mw: f64,
) -> Result<(f64, f64), Violation> {
    if net_export_mw < -EPSILON {
        let charge_mw = -net_export_mw;
        if charge_mw + EPSILON < min_stable_charge_mw {
            return Err(Violation::ReserveTargetBelowMinimumStableCharge);
        }
        Ok((charge_mw, 0.0))
    } else {
        Ok((0.0, net_export_mw.max(0.0)))
    }
}

fn validate_reserve_response(
    reserve: ReserveCommitment,
    requirement: MarketReserveRequirement,
    indeterminate_reasons: &mut Vec<&'static str>,
    violations: &mut Vec<Violation>,
) {
    for value in [
        reserve.mw,
        reserve.start_response_seconds.unwrap_or(0.0),
        reserve.full_activation_seconds.unwrap_or(0.0),
        reserve.sustain_duration_seconds.unwrap_or(0.0),
        reserve.proven_sustain_seconds.unwrap_or(0.0),
    ] {
        if !value.is_finite() || value < 0.0 {
            violations.push(Violation::InvalidReserveValue);
            return;
        }
    }
    if reserve.mw <= EPSILON {
        violations.push(Violation::InvalidReserveValue);
        return;
    }

    if let Some(deadline) = requirement.maximum_start_response_seconds {
        match reserve.start_response_seconds {
            Some(actual) if actual <= deadline + EPSILON => {}
            Some(_) => violations.push(Violation::ReserveResponseDeadlineExceeded),
            None => indeterminate_reasons.push("missing_start_response_evidence"),
        }
    }

    if let Some(deadline) = requirement.maximum_full_activation_seconds {
        match reserve.full_activation_seconds {
            Some(actual) if actual <= deadline + EPSILON => {}
            Some(_) => violations.push(Violation::ReserveResponseDeadlineExceeded),
            None => indeterminate_reasons.push("missing_full_activation_evidence"),
        }
    }

    let required_minimum = requirement.minimum_sustain_seconds.unwrap_or(0.0);
    let energy_assessment_duration = match reserve.sustain_duration_seconds {
        Some(duration) => duration.max(required_minimum),
        None if required_minimum > 0.0 => required_minimum,
        None => {
            indeterminate_reasons.push("sustain_duration_not_defined_by_market_profile");
            return;
        }
    };
    if energy_assessment_duration <= EPSILON {
        violations.push(Violation::InvalidReserveValue);
        return;
    }

    match reserve.proven_sustain_seconds {
        Some(proven) if proven + EPSILON >= energy_assessment_duration => {}
        Some(_) => violations.push(Violation::ReserveSustainInsufficient),
        None => indeterminate_reasons.push("missing_asset_sustain_evidence"),
    }
}

fn evaluate_reserve_energy_scenario(
    reserves: &[ReserveCommitment],
    direction: ReserveDirection,
    base_charge_mw: f64,
    base_discharge_mw: f64,
    starting_energy_mwh: f64,
    limits: AssetLimits,
    charge_efficiency: f64,
    discharge_efficiency: f64,
) -> Result<(), Violation> {
    let mut timed_reserves = Vec::new();
    for reserve in reserves.iter().filter(|item| item.direction == direction) {
        let requirement = sawem_rev_2_3_reserve_requirement(reserve.product, reserve.direction)?;
        let minimum = requirement.minimum_sustain_seconds.unwrap_or(0.0);
        let duration = match reserve.sustain_duration_seconds {
            Some(value) => value.max(minimum),
            None if minimum > 0.0 => minimum,
            None => return Err(Violation::InvalidReserveValue),
        };
        if reserve
            .proven_sustain_seconds
            .is_none_or(|proven| proven + EPSILON < duration)
        {
            return Err(Violation::ReserveSustainInsufficient);
        }
        timed_reserves.push((reserve.mw, duration));
    }
    if timed_reserves.is_empty() {
        return Ok(());
    }

    let mut boundaries: Vec<f64> = timed_reserves.iter().map(|(_, duration)| *duration).collect();
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup_by(|left, right| (*left - *right).abs() <= EPSILON);

    let baseline_net = base_discharge_mw - base_charge_mw;
    let mut energy = starting_energy_mwh;
    let mut previous_boundary = 0.0;

    for boundary in boundaries {
        let active_reserve_mw: f64 = timed_reserves
            .iter()
            .filter(|(_, duration)| *duration > previous_boundary + EPSILON)
            .map(|(mw, _)| *mw)
            .sum();
        let target_net = match direction {
            ReserveDirection::Up => baseline_net + active_reserve_mw,
            ReserveDirection::Down => baseline_net - active_reserve_mw,
        };
        let (target_charge_mw, target_discharge_mw) =
            operating_point_for_net_power(target_net, limits.min_stable_charge_mw)?;

        if target_charge_mw > limits.max_charge_mw + EPSILON
            || target_discharge_mw > limits.max_discharge_mw + EPSILON
        {
            return Err(Violation::ReservePowerHeadroomExceeded);
        }

        // Integrate only the incremental energy effect of reserve activation
        // relative to the scheduled operating point. This correctly treats
        // stopping an existing charge/discharge as reserve response.
        let incremental_energy_rate =
            (target_charge_mw - base_charge_mw) * charge_efficiency
                - (target_discharge_mw - base_discharge_mw) / discharge_efficiency;
        let elapsed_hours = (boundary - previous_boundary) / 3_600.0;
        energy += incremental_energy_rate * elapsed_hours;

        if energy < limits.min_energy_mwh - EPSILON {
            return Err(Violation::ReserveEnergyHeadroomExceeded);
        }
        if energy > limits.max_energy_mwh + EPSILON {
            return Err(Violation::ReserveEnergyHeadroomExceeded);
        }
        previous_boundary = boundary;
    }
    Ok(())
}

fn assess_reserves(
    limits: AssetLimits,
    charge_mw: f64,
    discharge_mw: f64,
    energy_after_schedule_mwh: f64,
    charge_efficiency: f64,
    discharge_efficiency: f64,
    reserves: &[ReserveCommitment],
) -> ReserveAssessment {
    let baseline_net = discharge_mw - charge_mw;
    let available_up_mw = (limits.max_discharge_mw - baseline_net).max(0.0);
    let available_down_mw = (baseline_net + limits.max_charge_mw).max(0.0);
    let mut committed_up_mw = 0.0;
    let mut committed_down_mw = 0.0;
    let mut seen = BTreeSet::new();
    let mut indeterminate_reasons = Vec::new();
    let mut violations = Vec::new();

    for reserve in reserves {
        if !seen.insert((reserve.product, reserve.direction)) {
            violations.push(Violation::DuplicateReserveProductDirection);
            continue;
        }
        let requirement = match sawem_rev_2_3_reserve_requirement(reserve.product, reserve.direction) {
            Ok(value) => value,
            Err(violation) => {
                violations.push(violation);
                continue;
            }
        };
        validate_reserve_response(
            *reserve,
            requirement,
            &mut indeterminate_reasons,
            &mut violations,
        );
        match reserve.direction {
            ReserveDirection::Up => committed_up_mw += reserve.mw,
            ReserveDirection::Down => committed_down_mw += reserve.mw,
        }
    }

    if committed_up_mw > available_up_mw + EPSILON {
        violations.push(Violation::ReservePowerHeadroomExceeded);
    }
    if committed_down_mw > available_down_mw + EPSILON {
        violations.push(Violation::ReservePowerHeadroomExceeded);
    }

    if committed_up_mw > EPSILON {
        let target = baseline_net + committed_up_mw;
        if let Err(violation) = operating_point_for_net_power(target, limits.min_stable_charge_mw) {
            violations.push(violation);
        }
    }
    if committed_down_mw > EPSILON {
        let target = baseline_net - committed_down_mw;
        if let Err(violation) = operating_point_for_net_power(target, limits.min_stable_charge_mw) {
            violations.push(violation);
        }
    }

    if !reserves.is_empty() {
        if reserves.iter().any(|reserve| {
            let requirement = sawem_rev_2_3_reserve_requirement(reserve.product, reserve.direction);
            requirement.is_err()
                || (reserve.sustain_duration_seconds.is_none()
                    && requirement
                        .ok()
                        .is_some_and(|value| value.minimum_sustain_seconds.is_none()))
        }) {
            indeterminate_reasons.push("reserve_energy_duration_incomplete");
        } else {
            for direction in [ReserveDirection::Up, ReserveDirection::Down] {
                match evaluate_reserve_energy_scenario(
                    reserves,
                    direction,
                    charge_mw,
                    discharge_mw,
                    energy_after_schedule_mwh,
                    limits,
                    charge_efficiency,
                    discharge_efficiency,
                ) {
                    Ok(()) => {}
                    Err(violation) => violations.push(violation),
                }
            }
        }
    }

    violations.sort_by_key(|value| format!("{value:?}"));
    violations.dedup();
    indeterminate_reasons.sort_unstable();
    indeterminate_reasons.dedup();

    let qualification = if !violations.is_empty() {
        ReserveQualification::Infeasible
    } else if !indeterminate_reasons.is_empty() {
        ReserveQualification::Indeterminate
    } else {
        ReserveQualification::Qualified
    };
    ReserveAssessment {
        qualification,
        committed_up_mw,
        committed_down_mw,
        available_up_mw,
        available_down_mw,
        indeterminate_reasons,
        violations,
    }
}

pub fn evaluate_dispatch(
    limits: AssetLimits,
    initial_energy_mwh: f64,
    steps: &[DispatchStep],
) -> Result<DispatchResult, InputError> {
    limits.validate()?;
    if !initial_energy_mwh.is_finite() {
        return Err(InputError::NonFiniteValue("initial_energy_mwh"));
    }
    if initial_energy_mwh < limits.min_energy_mwh - EPSILON
        || initial_energy_mwh > limits.max_energy_mwh + EPSILON
    {
        return Err(InputError::InvalidInitialEnergy);
    }

    let charge_efficiency = limits.round_trip_efficiency.sqrt();
    let discharge_efficiency = charge_efficiency;
    let mut energy = initial_energy_mwh;
    let mut previous_mode = OperatingMode::Standstill;
    let mut results = Vec::with_capacity(steps.len());

    for (index, step) in steps.iter().enumerate() {
        let mut violations = Vec::new();
        let finite_values = [
            ("duration_hours", step.duration_hours),
            ("charge_mw", step.charge_mw),
            ("discharge_mw", step.discharge_mw),
            (
                "transition_seconds_before_action",
                step.transition_seconds_before_action,
            ),
            ("reserve_duration_hours", step.reserve_duration_hours),
            ("network_export_limit_mw", step.network_export_limit_mw),
            ("network_import_limit_mw", step.network_import_limit_mw),
        ];
        for (field, value) in finite_values {
            if !value.is_finite() {
                return Err(InputError::NonFiniteValue(field));
            }
        }
        if step.duration_hours <= 0.0
            || step.charge_mw < 0.0
            || step.discharge_mw < 0.0
            || step.transition_seconds_before_action < 0.0
            || step.reserve_duration_hours < 0.0
            || step.network_export_limit_mw < 0.0
            || step.network_import_limit_mw < 0.0
        {
            return Err(InputError::NegativeValue("dispatch_step"));
        }

        let requested_mode = mode_for(step.charge_mw, step.discharge_mw);
        let simultaneous_charge_and_discharge =
            step.charge_mw > EPSILON && step.discharge_mw > EPSILON;
        if simultaneous_charge_and_discharge {
            violations.push(Violation::SimultaneousChargeAndDischarge);
        }
        let charge_power_valid = step.charge_mw <= limits.max_charge_mw + EPSILON;
        if !charge_power_valid {
            violations.push(Violation::ChargePowerExceeded);
        }
        let stable_charge_valid =
            step.charge_mw <= EPSILON || step.charge_mw + EPSILON >= limits.min_stable_charge_mw;
        if !stable_charge_valid {
            violations.push(Violation::BelowMinimumStableCharging);
        }
        let discharge_power_valid = step.discharge_mw <= limits.max_discharge_mw + EPSILON;
        if !discharge_power_valid {
            violations.push(Violation::DischargePowerExceeded);
        }
        let mode_request_valid = !simultaneous_charge_and_discharge
            && charge_power_valid
            && stable_charge_valid
            && discharge_power_valid;

        let required_transition =
            required_transition_seconds(previous_mode, requested_mode, limits.transitions);
        if step.transition_seconds_before_action + EPSILON < required_transition {
            violations.push(Violation::TransitionTimeInsufficient);
        }
        let interval_seconds = step.duration_hours * 3_600.0;
        if step.transition_seconds_before_action > interval_seconds + EPSILON {
            violations.push(Violation::TransitionTimeExceedsInterval);
        }

        let transition_valid = step.transition_seconds_before_action + EPSILON
            >= required_transition
            && step.transition_seconds_before_action <= interval_seconds + EPSILON;
        let net_export = step.discharge_mw - step.charge_mw;
        let export_valid = net_export <= step.network_export_limit_mw + EPSILON;
        if !export_valid {
            violations.push(Violation::ExportLimitExceeded);
        }
        let import_valid = -net_export <= step.network_import_limit_mw + EPSILON;
        if !import_valid {
            violations.push(Violation::ImportLimitExceeded);
        }
        let network_valid = export_valid && import_valid;

        let action_valid = mode_request_valid && transition_valid && network_valid;
        let active_hours = if action_valid {
            step.duration_hours - step.transition_seconds_before_action / 3_600.0
        } else {
            0.0
        };

        let (reserve_up, reserve_down) = match sum_reserves(step.reserves) {
            Ok(values) => values,
            Err(mut errors) => {
                violations.append(&mut errors);
                (0.0, 0.0)
            }
        };

        if step.discharge_mw + reserve_up > limits.max_discharge_mw + EPSILON
            || step.charge_mw + reserve_down > limits.max_charge_mw + EPSILON
        {
            violations.push(Violation::ReservePowerHeadroomExceeded);
        }

        let energy_before = energy;
        let mut candidate_energy = energy_before;
        if action_valid {
            candidate_energy += step.charge_mw * charge_efficiency * active_hours;
            candidate_energy -= step.discharge_mw / discharge_efficiency * active_hours;
        }

        let energy_bounds_valid = candidate_energy >= limits.min_energy_mwh - EPSILON
            && candidate_energy <= limits.max_energy_mwh + EPSILON;
        if candidate_energy < limits.min_energy_mwh - EPSILON {
            violations.push(Violation::EnergyBelowMinimum);
        }
        if candidate_energy > limits.max_energy_mwh + EPSILON {
            violations.push(Violation::EnergyAboveMaximum);
        }

        let upward_energy_required =
            reserve_up * step.reserve_duration_hours / discharge_efficiency;
        let downward_energy_required =
            reserve_down * step.reserve_duration_hours * charge_efficiency;
        if candidate_energy - limits.min_energy_mwh + EPSILON < upward_energy_required
            || limits.max_energy_mwh - candidate_energy + EPSILON < downward_energy_required
        {
            violations.push(Violation::ReserveEnergyHeadroomExceeded);
        }

        // Commit only physically and network-feasible dispatch. Retain the
        // attempted energy separately so the audit explains hard failures.
        let action_committed = action_valid && energy_bounds_valid;
        if action_committed {
            energy = candidate_energy;
        }
        let achieved_mode = if action_committed {
            requested_mode
        } else {
            previous_mode
        };

        results.push(StepResult {
            index,
            requested_mode,
            achieved_mode,
            energy_before_mwh: energy_before,
            energy_after_mwh: energy,
            candidate_energy_after_mwh: candidate_energy,
            net_export_mw: net_export,
            transition_seconds_used: step.transition_seconds_before_action,
            reserve_up_mw: reserve_up,
            reserve_down_mw: reserve_down,
            violations,
        });
        previous_mode = achieved_mode;
    }

    let feasible = results.iter().all(|step| step.violations.is_empty());
    Ok(DispatchResult {
        feasible,
        initial_energy_mwh,
        final_energy_mwh: energy,
        steps: results,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SevenDayEnergyLimits {
    pub daily_limit_mwh: f64,
    pub next_six_days_limit_mwh: f64,
    pub seven_day_total_limit_mwh: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyLimitResult {
    pub feasible: bool,
    pub daily_total_mwh: f64,
    pub next_six_days_total_mwh: f64,
    pub seven_day_total_mwh: f64,
    pub violations: Vec<&'static str>,
}

pub fn validate_seven_day_energy_declarations(
    declarations_mwh: &[f64],
    limits: SevenDayEnergyLimits,
) -> Result<EnergyLimitResult, InputError> {
    if declarations_mwh.len() != 7 {
        return Err(InputError::InvalidSevenDayDeclarationCount);
    }
    for value in declarations_mwh {
        if !value.is_finite() {
            return Err(InputError::NonFiniteValue("daily_declaration_mwh"));
        }
        if *value < 0.0 {
            return Err(InputError::NegativeValue("daily_declaration_mwh"));
        }
    }
    let caps = [
        ("daily_limit_mwh", limits.daily_limit_mwh),
        ("next_six_days_limit_mwh", limits.next_six_days_limit_mwh),
        (
            "seven_day_total_limit_mwh",
            limits.seven_day_total_limit_mwh,
        ),
    ];
    for (name, value) in caps {
        if !value.is_finite() {
            return Err(InputError::NonFiniteValue(name));
        }
        if value < 0.0 {
            return Err(InputError::NegativeValue(name));
        }
    }

    let daily_total = declarations_mwh[0];
    let next_six_total: f64 = declarations_mwh[1..].iter().sum();
    let seven_day_total: f64 = declarations_mwh.iter().sum();
    let mut violations = Vec::new();
    if declarations_mwh
        .iter()
        .any(|value| *value > limits.daily_limit_mwh + EPSILON)
    {
        violations.push("daily_limit_exceeded");
    }
    if next_six_total > limits.next_six_days_limit_mwh + EPSILON {
        violations.push("next_six_days_limit_exceeded");
    }
    if seven_day_total > limits.seven_day_total_limit_mwh + EPSILON {
        violations.push("seven_day_total_limit_exceeded");
    }

    Ok(EnergyLimitResult {
        feasible: violations.is_empty(),
        daily_total_mwh: daily_total,
        next_six_days_total_mwh: next_six_total,
        seven_day_total_mwh: seven_day_total,
        violations,
    })
}

pub fn required_credit_cover_minor_units(
    actual_exposure_minor_units: u64,
    undefined_potential_exposure_minor_units: u64,
) -> Result<u64, InputError> {
    actual_exposure_minor_units
        .checked_add(undefined_potential_exposure_minor_units)
        .ok_or(InputError::IntegerOverflow)
}

pub fn sum_balancing_payment_components_minor_units(components: &[i64]) -> Result<i64, InputError> {
    components.iter().try_fold(0_i64, |total, component| {
        total
            .checked_add(*component)
            .ok_or(InputError::IntegerOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> AssetLimits {
        AssetLimits {
            max_charge_mw: 50.0,
            min_stable_charge_mw: 5.0,
            max_discharge_mw: 60.0,
            min_energy_mwh: 20.0,
            max_energy_mwh: 200.0,
            round_trip_efficiency: 0.9,
            transitions: TransitionTimes {
                standstill_to_charging_seconds: 300.0,
                standstill_to_discharging_seconds: 900.0,
                charging_to_discharging_seconds: 600.0,
                discharging_to_charging_seconds: 600.0,
            },
        }
    }

    fn empty_reserves() -> &'static [ReserveCommitment] {
        &[]
    }

    fn step(
        charge_mw: f64,
        discharge_mw: f64,
        transition_seconds: f64,
        reserves: &'static [ReserveCommitment],
    ) -> DispatchStep {
        DispatchStep {
            duration_hours: 1.0,
            charge_mw,
            discharge_mw,
            transition_seconds_before_action: transition_seconds,
            reserve_duration_hours: 0.25,
            reserves,
            network_export_limit_mw: 60.0,
            network_import_limit_mw: 50.0,
        }
    }

    #[test]
    fn feasible_discharge_reserve_and_energy_balance() {
        static RESERVES: [ReserveCommitment; 2] = [
            ReserveCommitment {
                product: ReserveProduct::Regulating,
                direction: ReserveDirection::Up,
                mw: 5.0,
            },
            ReserveCommitment {
                product: ReserveProduct::TenMinute,
                direction: ReserveDirection::Up,
                mw: 5.0,
            },
        ];
        let result = evaluate_dispatch(limits(), 150.0, &[step(0.0, 30.0, 900.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(result.feasible);
        assert!((result.final_energy_mwh - (150.0 - 30.0 / 0.9_f64.sqrt() * 0.75)).abs() < 1.0e-8);
        assert_eq!(result.steps[0].reserve_up_mw, 10.0);
    }

    #[test]
    fn rejects_energy_plus_reserve_double_counting() {
        static RESERVES: [ReserveCommitment; 1] = [ReserveCommitment {
            product: ReserveProduct::Instantaneous,
            direction: ReserveDirection::Up,
            mw: 10.0,
        }];
        let result = evaluate_dispatch(limits(), 150.0, &[step(0.0, 55.0, 900.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ReservePowerHeadroomExceeded)
        );
    }

    #[test]
    fn rejects_reserve_without_soc_energy_headroom() {
        static RESERVES: [ReserveCommitment; 1] = [ReserveCommitment {
            product: ReserveProduct::Regulating,
            direction: ReserveDirection::Up,
            mw: 20.0,
        }];
        let result = evaluate_dispatch(limits(), 25.0, &[step(0.0, 0.0, 0.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ReserveEnergyHeadroomExceeded)
        );
    }

    #[test]
    fn rejects_minimum_stable_charge_violation() {
        let result = evaluate_dispatch(limits(), 100.0, &[step(2.0, 0.0, 300.0, empty_reserves())])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::BelowMinimumStableCharging)
        );
    }

    #[test]
    fn rejects_insufficient_mode_transition_time() {
        let result = evaluate_dispatch(limits(), 100.0, &[step(0.0, 10.0, 60.0, empty_reserves())])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::TransitionTimeInsufficient)
        );
    }

    #[test]
    fn failed_transition_does_not_promote_internal_mode() {
        let result = evaluate_dispatch(
            limits(),
            100.0,
            &[
                step(0.0, 10.0, 60.0, empty_reserves()),
                step(0.0, 10.0, 0.0, empty_reserves()),
            ],
        )
        .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert_eq!(result.steps[0].achieved_mode, OperatingMode::Standstill);
        assert_eq!(result.steps[1].achieved_mode, OperatingMode::Standstill);
        assert!(
            result.steps[1]
                .violations
                .contains(&Violation::TransitionTimeInsufficient)
        );
        assert_eq!(result.final_energy_mwh, 100.0);
    }

    #[test]
    fn rejects_network_export_violation() {
        let mut s = step(0.0, 40.0, 900.0, empty_reserves());
        s.network_export_limit_mw = 30.0;
        let result = evaluate_dispatch(limits(), 100.0, &[s]).expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ExportLimitExceeded)
        );
        assert_eq!(result.steps[0].energy_after_mwh, 100.0);
        assert_eq!(result.final_energy_mwh, 100.0);
        assert_eq!(result.steps[0].achieved_mode, OperatingMode::Standstill);
    }

    #[test]
    fn rejects_soc_floor_and_ceiling_violations() {
        let discharge =
            evaluate_dispatch(limits(), 25.0, &[step(0.0, 10.0, 900.0, empty_reserves())])
                .expect("fixture inputs are valid");
        assert!(!discharge.feasible);
        assert!(
            discharge.steps[0]
                .violations
                .contains(&Violation::EnergyBelowMinimum)
        );
        assert_eq!(discharge.steps[0].energy_after_mwh, 25.0);
        assert!(discharge.steps[0].candidate_energy_after_mwh < 20.0);

        let charge =
            evaluate_dispatch(limits(), 195.0, &[step(10.0, 0.0, 300.0, empty_reserves())])
                .expect("fixture inputs are valid");
        assert!(!charge.feasible);
        assert!(
            charge.steps[0]
                .violations
                .contains(&Violation::EnergyAboveMaximum)
        );
        assert_eq!(charge.steps[0].energy_after_mwh, 195.0);
        assert!(charge.steps[0].candidate_energy_after_mwh > 200.0);
    }

    #[test]
    fn rejects_duplicate_reserve_product_direction() {
        static RESERVES: [ReserveCommitment; 2] = [
            ReserveCommitment {
                product: ReserveProduct::Regulating,
                direction: ReserveDirection::Up,
                mw: 2.0,
            },
            ReserveCommitment {
                product: ReserveProduct::Regulating,
                direction: ReserveDirection::Up,
                mw: 3.0,
            },
        ];
        let result = evaluate_dispatch(limits(), 100.0, &[step(0.0, 0.0, 0.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::DuplicateReserveProductDirection)
        );
    }

    #[test]
    fn validates_daily_and_rolling_week_energy_caps() {
        let limits = SevenDayEnergyLimits {
            daily_limit_mwh: 250.0,
            next_six_days_limit_mwh: 1_200.0,
            seven_day_total_limit_mwh: 1_450.0,
        };
        let declarations = [180.0, 190.0, 210.0, 175.0, 200.0, 190.0, 180.0];
        let result = validate_seven_day_energy_declarations(&declarations, limits)
            .expect("fixture inputs are valid");
        assert!(result.feasible);
        assert_eq!(result.next_six_days_total_mwh, 1_145.0);
        assert_eq!(result.seven_day_total_mwh, 1_325.0);

        let invalid = [180.0, 240.0, 240.0, 240.0, 240.0, 240.0, 240.0];
        let rejected = validate_seven_day_energy_declarations(&invalid, limits)
            .expect("fixture inputs are valid");
        assert!(!rejected.feasible);
        assert!(
            rejected
                .violations
                .contains(&"next_six_days_limit_exceeded")
        );
        assert!(
            rejected
                .violations
                .contains(&"seven_day_total_limit_exceeded")
        );
    }

    #[test]
    fn credit_cover_and_multiple_balancing_components_use_checked_arithmetic() {
        assert_eq!(
            required_credit_cover_minor_units(25_000_000, 10_000_000),
            Ok(35_000_000)
        );
        assert_eq!(
            sum_balancing_payment_components_minor_units(&[-4_500_000, 1_000_000]),
            Ok(-3_500_000)
        );
        assert_eq!(
            required_credit_cover_minor_units(u64::MAX, 1),
            Err(InputError::IntegerOverflow)
        );
    }
}
