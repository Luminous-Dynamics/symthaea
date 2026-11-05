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
    pub reserve_duration_hours: f64,
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
    DuplicateReserveProductDirection,
    EnergyBelowMinimum,
    EnergyAboveMaximum,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    pub index: usize,
    pub mode: OperatingMode,
    pub energy_before_mwh: f64,
    pub energy_after_mwh: f64,
    pub net_export_mw: f64,
    pub transition_seconds_used: f64,
    pub reserve_up_mw: f64,
    pub reserve_down_mw: f64,
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

fn sum_reserves(reserves: &[ReserveCommitment]) -> Result<(f64, f64), Vec<Violation>> {
    let mut seen = BTreeSet::new();
    let mut upward = 0.0;
    let mut downward = 0.0;
    let mut violations = Vec::new();

    for reserve in reserves {
        if !reserve.mw.is_finite() || reserve.mw < 0.0 {
            violations.push(Violation::ReservePowerHeadroomExceeded);
            continue;
        }
        if !seen.insert((reserve.product, reserve.direction)) {
            violations.push(Violation::DuplicateReserveProductDirection);
        }
        match reserve.direction {
            ReserveDirection::Up => upward += reserve.mw,
            ReserveDirection::Down => downward += reserve.mw,
        }
    }
    if violations.is_empty() {
        Ok((upward, downward))
    } else {
        Err(violations)
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

        let mode = mode_for(step.charge_mw, step.discharge_mw);
        if step.charge_mw > EPSILON && step.discharge_mw > EPSILON {
            violations.push(Violation::SimultaneousChargeAndDischarge);
        }
        if step.charge_mw > limits.max_charge_mw + EPSILON {
            violations.push(Violation::ChargePowerExceeded);
        }
        if step.charge_mw > EPSILON && step.charge_mw + EPSILON < limits.min_stable_charge_mw {
            violations.push(Violation::BelowMinimumStableCharging);
        }
        if step.discharge_mw > limits.max_discharge_mw + EPSILON {
            violations.push(Violation::DischargePowerExceeded);
        }

        let required_transition =
            required_transition_seconds(previous_mode, mode, limits.transitions);
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
        let active_hours = if transition_valid {
            step.duration_hours - step.transition_seconds_before_action / 3_600.0
        } else {
            0.0
        };

        let net_export = step.discharge_mw - step.charge_mw;
        if net_export > step.network_export_limit_mw + EPSILON {
            violations.push(Violation::ExportLimitExceeded);
        }
        if -net_export > step.network_import_limit_mw + EPSILON {
            violations.push(Violation::ImportLimitExceeded);
        }

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
        energy += step.charge_mw * charge_efficiency * active_hours;
        energy -= step.discharge_mw / discharge_efficiency * active_hours;

        if energy < limits.min_energy_mwh - EPSILON {
            violations.push(Violation::EnergyBelowMinimum);
        }
        if energy > limits.max_energy_mwh + EPSILON {
            violations.push(Violation::EnergyAboveMaximum);
        }

        let upward_energy_required =
            reserve_up * step.reserve_duration_hours / discharge_efficiency;
        let downward_energy_required =
            reserve_down * step.reserve_duration_hours * charge_efficiency;
        if energy - limits.min_energy_mwh + EPSILON < upward_energy_required
            || limits.max_energy_mwh - energy + EPSILON < downward_energy_required
        {
            violations.push(Violation::ReserveEnergyHeadroomExceeded);
        }

        results.push(StepResult {
            index,
            mode,
            energy_before_mwh: energy_before,
            energy_after_mwh: energy,
            net_export_mw: net_export,
            transition_seconds_used: step.transition_seconds_before_action,
            reserve_up_mw: reserve_up,
            reserve_down_mw: reserve_down,
            violations,
        });
        previous_mode = mode;
    }

    let feasible = results.iter().all(|step| step.violations.is_empty());
    Ok(DispatchResult {
        feasible,
        initial_energy_mwh,
        final_energy_mwh: energy,
        steps: results,
    })
}

#[derive(Debug, Clone, PartialEq)]
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

        let charge =
            evaluate_dispatch(limits(), 195.0, &[step(10.0, 0.0, 300.0, empty_reserves())])
                .expect("fixture inputs are valid");
        assert!(!charge.feasible);
        assert!(
            charge.steps[0]
                .violations
                .contains(&Violation::EnergyAboveMaximum)
        );
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
