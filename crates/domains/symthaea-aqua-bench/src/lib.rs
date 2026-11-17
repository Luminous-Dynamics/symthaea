//! Deterministic resource accounting for Luminous Aqua service benchmarks.
//!
//! This crate performs arithmetic and input validation only. It is not a water-quality
//! authority, hygiene assessor, plumbing controller, safety interlock, hydraulic solver,
//! certification tool, or proof of physical performance. Missing measurements are errors
//! for whole-service calculations; callers must not turn unknown values into zero.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub const LEDGER_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq)]
pub enum LedgerError {
    MissingMeasurement { field: &'static str },
    NonFiniteMeasurement { field: &'static str },
    NegativeMeasurement { field: &'static str },
    NonPositiveControl { field: &'static str },
}

impl Display for LedgerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMeasurement { field } => write!(f, "missing required measurement: {field}"),
            Self::NonFiniteMeasurement { field } => write!(f, "non-finite measurement: {field}"),
            Self::NegativeMeasurement { field } => write!(f, "negative measurement: {field}"),
            Self::NonPositiveControl { field } => write!(f, "control must be greater than zero: {field}"),
        }
    }
}

impl Error for LedgerError {}

fn nonnegative(field: &'static str, value: Option<f64>) -> Result<f64, LedgerError> {
    let value = value.ok_or(LedgerError::MissingMeasurement { field })?;
    if !value.is_finite() {
        return Err(LedgerError::NonFiniteMeasurement { field });
    }
    if value < 0.0 {
        return Err(LedgerError::NegativeMeasurement { field });
    }
    Ok(value)
}

fn signed_finite(field: &'static str, value: Option<f64>) -> Result<f64, LedgerError> {
    let value = value.ok_or(LedgerError::MissingMeasurement { field })?;
    if !value.is_finite() {
        return Err(LedgerError::NonFiniteMeasurement { field });
    }
    Ok(value)
}

/// Whole-service water ledger, from request through post-use ready state.
///
/// All values are litres. `recirculated_delivery_l` is reported separately and is never
/// added to fresh-water consumption: it is an internal transfer within the session.
/// `periodic_cleaning_allocated_l` is the measured periodic-cleaning burden allocated
/// to this service using a declared allocation rule.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WaterServiceLedger {
    pub startup_l: Option<f64>,
    pub priming_l: Option<f64>,
    pub active_fresh_l: Option<f64>,
    pub post_use_cleaning_l: Option<f64>,
    pub purge_l: Option<f64>,
    pub periodic_cleaning_allocated_l: Option<f64>,
    pub retry_recovery_l: Option<f64>,
    pub recirculated_delivery_l: Option<f64>,
}

impl WaterServiceLedger {
    /// Returns complete fresh-water burden. Every component must be measured, including
    /// explicit zero values for operations that do not apply.
    pub fn fresh_water_l(&self) -> Result<f64, LedgerError> {
        let values = [
            ("startup_l", self.startup_l),
            ("priming_l", self.priming_l),
            ("active_fresh_l", self.active_fresh_l),
            ("post_use_cleaning_l", self.post_use_cleaning_l),
            ("purge_l", self.purge_l),
            ("periodic_cleaning_allocated_l", self.periodic_cleaning_allocated_l),
            ("retry_recovery_l", self.retry_recovery_l),
        ];
        values.into_iter().try_fold(0.0, |sum, (field, value)| {
            let total = sum + nonnegative(field, value)?;
            if total.is_finite() {
                Ok(total)
            } else {
                Err(LedgerError::NonFiniteMeasurement { field: "fresh_water_total_l" })
            }
        })
    }

    /// Returns the internally recirculated delivery volume as a separate quantity.
    pub fn recirculated_delivery_l(&self) -> Result<f64, LedgerError> {
        nonnegative("recirculated_delivery_l", self.recirculated_delivery_l)
    }
}

/// Returns fresh-water savings against a matched control service.
///
/// The caller is responsible for ensuring equal task, duration, operating conditions,
/// output quality, and complete accounting boundaries. This function cannot establish
/// comparability or service quality.
pub fn water_savings_percent(
    control: &WaterServiceLedger,
    candidate: &WaterServiceLedger,
) -> Result<f64, LedgerError> {
    let control_l = control.fresh_water_l()?;
    if control_l <= 0.0 {
        return Err(LedgerError::NonPositiveControl { field: "control_fresh_water_l" });
    }
    let candidate_l = candidate.fresh_water_l()?;
    let savings = 100.0 * (1.0 - candidate_l / control_l);
    if !savings.is_finite() {
        return Err(LedgerError::NonFiniteMeasurement { field: "water_savings_percent" });
    }
    Ok(savings)
}

/// Purchased energy ledger. Values are kWh; fuel and electricity remain separately
/// visible. Useful recovered heat is reported, not subtracted from purchased energy,
/// which avoids mixing a thermal output with purchased input or double-crediting it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyServiceLedger {
    pub heater_electricity_kwh: Option<f64>,
    pub pump_electricity_kwh: Option<f64>,
    pub treatment_electricity_kwh: Option<f64>,
    pub controls_electricity_kwh: Option<f64>,
    pub cleaning_electricity_kwh: Option<f64>,
    pub allocated_room_electricity_kwh: Option<f64>,
    pub heater_fuel_input_kwh: Option<f64>,
    pub recovered_useful_heat_kwh: Option<f64>,
}

impl EnergyServiceLedger {
    pub fn purchased_electricity_kwh(&self) -> Result<f64, LedgerError> {
        let values = [
            ("heater_electricity_kwh", self.heater_electricity_kwh),
            ("pump_electricity_kwh", self.pump_electricity_kwh),
            ("treatment_electricity_kwh", self.treatment_electricity_kwh),
            ("controls_electricity_kwh", self.controls_electricity_kwh),
            ("cleaning_electricity_kwh", self.cleaning_electricity_kwh),
            ("allocated_room_electricity_kwh", self.allocated_room_electricity_kwh),
        ];
        values.into_iter().try_fold(0.0, |sum, (field, value)| {
            let total = sum + nonnegative(field, value)?;
            if total.is_finite() {
                Ok(total)
            } else {
                Err(LedgerError::NonFiniteMeasurement { field: "purchased_electricity_kwh" })
            }
        })
    }

    pub fn purchased_fuel_kwh(&self) -> Result<f64, LedgerError> {
        nonnegative("heater_fuel_input_kwh", self.heater_fuel_input_kwh)
    }

    pub fn purchased_energy_kwh(&self) -> Result<f64, LedgerError> {
        let total = self.purchased_electricity_kwh()? + self.purchased_fuel_kwh()?;
        if !total.is_finite() {
            return Err(LedgerError::NonFiniteMeasurement { field: "purchased_energy_kwh" });
        }
        Ok(total)
    }

    pub fn recovered_useful_heat_kwh(&self) -> Result<f64, LedgerError> {
        nonnegative("recovered_useful_heat_kwh", self.recovered_useful_heat_kwh)
    }
}

/// Whole-boundary water balance for a completed shower service.
///
/// Internal recirculation is intentionally excluded because it is not an external
/// source or sink. `storage_change_l` is end-minus-start storage (positive means
/// accumulation, negative means depletion). All other output streams are nonnegative.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WaterBalance {
    pub source_input_l: Option<f64>,
    pub end_use_discharge_l: Option<f64>,
    pub cleaning_purge_discharge_l: Option<f64>,
    pub reject_discharge_l: Option<f64>,
    pub evaporative_or_aerosol_loss_l: Option<f64>,
    pub storage_change_l: Option<f64>,
    pub expanded_uncertainty_l: Option<f64>,
}

impl WaterBalance {
    /// Positive residual means measured source input exceeds measured outputs and
    /// retained storage. Negative residual means measured outputs/storage exceed input.
    pub fn closure_residual_l(&self) -> Result<f64, LedgerError> {
        let input = nonnegative("source_input_l", self.source_input_l)?;
        let end_use = nonnegative("end_use_discharge_l", self.end_use_discharge_l)?;
        let cleaning = nonnegative("cleaning_purge_discharge_l", self.cleaning_purge_discharge_l)?;
        let reject = nonnegative("reject_discharge_l", self.reject_discharge_l)?;
        let loss = nonnegative("evaporative_or_aerosol_loss_l", self.evaporative_or_aerosol_loss_l)?;
        let storage_change = signed_finite("storage_change_l", self.storage_change_l)?;
        let residual = input - end_use - cleaning - reject - loss - storage_change;
        if !residual.is_finite() {
            return Err(LedgerError::NonFiniteMeasurement { field: "water_balance_residual_l" });
        }
        Ok(residual)
    }

    pub fn closes_within_uncertainty(&self) -> Result<bool, LedgerError> {
        let uncertainty = nonnegative("expanded_uncertainty_l", self.expanded_uncertainty_l)?;
        Ok(self.closure_residual_l()?.abs() <= uncertainty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fully_measured_water(active: f64, clean: f64) -> WaterServiceLedger {
        WaterServiceLedger {
            startup_l: Some(0.0),
            priming_l: Some(0.0),
            active_fresh_l: Some(active),
            post_use_cleaning_l: Some(clean),
            purge_l: Some(0.0),
            periodic_cleaning_allocated_l: Some(0.0),
            retry_recovery_l: Some(0.0),
            recirculated_delivery_l: Some(0.0),
        }
    }

    #[test]
    fn counts_post_use_cleaning_in_fresh_water_total() {
        // KOHLER's stated 5 US gal session + 0.75 US gal Rapid Clean example.
        let session_l = 5.0 * 3.785_411_784;
        let cleaning_l = 0.75 * 3.785_411_784;
        let ledger = fully_measured_water(session_l, cleaning_l);
        let total = ledger.fresh_water_l().expect("complete ledger");
        assert!((total - 21.766_117_758).abs() < 1e-9);
    }

    #[test]
    fn calculates_boundary_normalized_savings() {
        let control = fully_measured_water(25.0 * 3.785_411_784, 0.0);
        let candidate = fully_measured_water(5.0 * 3.785_411_784, 0.75 * 3.785_411_784);
        let savings = water_savings_percent(&control, &candidate).expect("valid comparison arithmetic");
        assert!((savings - 77.0).abs() < 1e-10);
    }

    #[test]
    fn internal_recirculation_is_not_counted_as_fresh_water() {
        let mut ledger = fully_measured_water(20.0, 0.0);
        ledger.recirculated_delivery_l = Some(80.0);
        assert_eq!(ledger.fresh_water_l().expect("fresh water"), 20.0);
        assert_eq!(ledger.recirculated_delivery_l().expect("recirculated delivery"), 80.0);
    }

    #[test]
    fn rejects_overflow_in_derived_fresh_water_total() {
        let ledger = fully_measured_water(f64::MAX, f64::MAX);
        assert_eq!(
            ledger.fresh_water_l(),
            Err(LedgerError::NonFiniteMeasurement { field: "fresh_water_total_l" })
        );
    }

    #[test]
    fn missing_cleaning_measurement_rejects_whole_service_total() {
        let mut ledger = fully_measured_water(20.0, 0.0);
        ledger.post_use_cleaning_l = None;
        assert_eq!(
            ledger.fresh_water_l(),
            Err(LedgerError::MissingMeasurement { field: "post_use_cleaning_l" })
        );
    }

    #[test]
    fn rejects_negative_and_non_finite_measurements() {
        let mut ledger = fully_measured_water(20.0, 0.0);
        ledger.active_fresh_l = Some(-0.1);
        assert_eq!(ledger.fresh_water_l(), Err(LedgerError::NegativeMeasurement { field: "active_fresh_l" }));
        ledger.active_fresh_l = Some(f64::NAN);
        assert_eq!(ledger.fresh_water_l(), Err(LedgerError::NonFiniteMeasurement { field: "active_fresh_l" }));
    }

    #[test]
    fn zero_control_cannot_produce_savings_percentage() {
        let control = fully_measured_water(0.0, 0.0);
        let candidate = fully_measured_water(1.0, 0.0);
        assert_eq!(
            water_savings_percent(&control, &candidate),
            Err(LedgerError::NonPositiveControl { field: "control_fresh_water_l" })
        );
    }

    #[test]
    fn water_balance_excludes_internal_recirculation_and_checks_uncertainty() {
        let balance = WaterBalance {
            source_input_l: Some(100.0),
            end_use_discharge_l: Some(90.0),
            cleaning_purge_discharge_l: Some(5.0),
            reject_discharge_l: Some(1.0),
            evaporative_or_aerosol_loss_l: Some(2.0),
            storage_change_l: Some(2.0),
            expanded_uncertainty_l: Some(0.1),
        };
        assert_eq!(balance.closure_residual_l().expect("complete balance"), 0.0);
        assert!(balance.closes_within_uncertainty().expect("uncertainty supplied"));
    }

    #[test]
    fn unresolved_water_balance_is_not_marked_closed() {
        let balance = WaterBalance {
            source_input_l: Some(100.0),
            end_use_discharge_l: Some(90.0),
            cleaning_purge_discharge_l: Some(5.0),
            reject_discharge_l: Some(1.0),
            evaporative_or_aerosol_loss_l: Some(2.0),
            storage_change_l: Some(2.5),
            expanded_uncertainty_l: Some(0.1),
        };
        assert!(!balance.closes_within_uncertainty().expect("complete balance"));
    }

    #[test]
    fn recovered_heat_is_reported_not_subtracted_from_purchased_energy() {
        let ledger = EnergyServiceLedger {
            heater_electricity_kwh: Some(1.0),
            pump_electricity_kwh: Some(0.1),
            treatment_electricity_kwh: Some(0.2),
            controls_electricity_kwh: Some(0.05),
            cleaning_electricity_kwh: Some(0.05),
            allocated_room_electricity_kwh: Some(0.1),
            heater_fuel_input_kwh: Some(0.5),
            recovered_useful_heat_kwh: Some(2.0),
        };
        assert!((ledger.purchased_electricity_kwh().expect("electricity") - 1.5).abs() < 1e-12);
        assert!((ledger.purchased_energy_kwh().expect("purchased") - 2.0).abs() < 1e-12);
        assert_eq!(ledger.recovered_useful_heat_kwh().expect("recovered heat"), 2.0);
    }
}
