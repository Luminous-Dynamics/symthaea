// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Storage-scheduling scenario harness: battery + solar + load + a
//! time-of-use tariff, scored on cost / unserved energy / battery cycles.
//!
//! Per PLANETARY_ENERGY_COORDINATION_PLAN_2026-07-06.md Phase 2: "Symthaea
//! proposes charge/discharge setpoints inside the guard envelope; score vs
//! baseline on cost, unserved energy, battery cycles. This is the
//! presence-of-improvement gate — a regression gate is not enough."
//!
//! [`naive_greedy_policy`] is the baseline (no look-ahead, no reserve
//! management). [`ReserveAwarePolicy`] is the first concrete advisor: a
//! rule-based (not learned) policy that holds back a battery reserve during
//! daylight hours so it isn't stranded without storage for the predictable
//! evening peak. It is intentionally NOT the full HDC/consciousness-driven
//! advisor described elsewhere in the plan -- it exists to prove the
//! `DispatchPolicy` interface and scoring harness can show a real,
//! measured improvement over a naive baseline, which a future
//! learned/HDC-driven policy can plug into the same slot.

use crate::battery::Battery;

/// Time-of-use import/export tariff.
#[derive(Debug, Clone, Copy)]
pub struct TariffSchedule {
    pub off_peak_price_per_kwh: f64,
    pub peak_price_per_kwh: f64,
    pub peak_start_hour: f64,
    pub peak_end_hour: f64,
    /// Price paid for exported surplus (typically well below the import
    /// price -- net-metering-adjacent assumption).
    pub export_price_per_kwh: f64,
}

impl TariffSchedule {
    pub fn import_price(&self, time_of_day_hours: f64) -> f64 {
        let hour = time_of_day_hours.rem_euclid(24.0);
        // Support both same-day windows (for example 17:00-21:00) and
        // windows that cross midnight (for example 22:00-06:00).
        let in_peak = if self.peak_start_hour <= self.peak_end_hour {
            hour >= self.peak_start_hour && hour < self.peak_end_hour
        } else {
            hour >= self.peak_start_hour || hour < self.peak_end_hour
        };
        if in_peak {
            self.peak_price_per_kwh
        } else {
            self.off_peak_price_per_kwh
        }
    }
}

/// Scored outcome of running a scenario to completion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScenarioResult {
    /// Net cost in the tariff's currency unit (import cost minus export credit).
    pub total_cost: f64,
    /// Energy demanded but not met by generation, battery, or (if
    /// `grid_available` was false) the grid, kWh.
    pub unserved_energy_kwh: f64,
    /// Battery equivalent full cycles accumulated over the scenario.
    pub battery_cycles: f64,
}

/// Run a scenario from `start_hour` for `total_hours` in steps of
/// `dt_hours`, calling `policy` each step to decide charge/discharge
/// setpoints (kW, clamped to the battery's power rating before being
/// applied). `start_hour` is applied consistently to `load_profile`,
/// `generation_profile`, `tariff`'s time-of-use lookup, AND `policy` --
/// they all see the same absolute clock. (An earlier version of this
/// scenario harness let callers shift `load_profile`/`generation_profile`
/// via wrapper closures while `tariff.import_price` kept reading the
/// unshifted internal loop counter; that silently priced every import at
/// whatever tariff bracket `[0, total_hours)` happened to fall into,
/// which is why a "start at 11:00, run to 21:00" scenario was still being
/// priced as if hour 17-21 never occurred. `start_hour` fixes this at the
/// root instead of requiring every caller to reimplement the shift
/// correctly by hand.)
///
/// `grid_available`: if true, any shortfall not covered by generation +
/// battery is imported at `tariff`'s price (adds to `total_cost`, not
/// `unserved_energy_kwh`); any surplus not stored is exported at the
/// tariff's export price. If false (islanded), shortfall becomes
/// `unserved_energy_kwh` instead and surplus is simply curtailed (no cost
/// either way -- there's no grid to sell it to).
#[allow(clippy::too_many_arguments)]
/// Failures returned by the validated energy-scheduling scenario runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioError {
    InvalidTimeStep,
    InvalidHorizon,
    InvalidStartHour,
    InvalidTariff,
    InvalidBatteryConfiguration,
    InvalidProfile,
    InvalidSetpoint,
    SimultaneousChargeAndDischarge,
    NonFiniteResult,
}

/// Compatibility wrapper for callers that already supply trusted, valid
/// scenarios. Untrusted or generated scenario inputs should use
/// try_run_scenario so invalid data is reported instead of panicking or
/// hanging the simulator.
#[allow(clippy::too_many_arguments)]
pub fn run_scenario(
    battery: &mut Battery,
    tariff: &TariffSchedule,
    load_profile: impl Fn(f64) -> f64,
    generation_profile: impl Fn(f64) -> f64,
    dt_hours: f64,
    total_hours: f64,
    start_hour: f64,
    grid_available: bool,
    policy: impl FnMut(f64, f64, f64, &Battery) -> (f64, f64),
) -> ScenarioResult {
    try_run_scenario(
        battery,
        tariff,
        load_profile,
        generation_profile,
        dt_hours,
        total_hours,
        start_hour,
        grid_available,
        policy,
    )
    .unwrap_or_else(|error| panic!("invalid energy scheduling scenario: {error:?}"))
}

/// Run a validated scenario atomically with respect to battery state.
///
/// The runner rejects invalid time steps, invalid profiles and non-finite
/// control outputs; it prevents a zero timestep from creating an infinite
/// loop; it uses a shorter final step when the horizon is not divisible by
/// dt; and it reports only the equivalent-full-cycle increment caused by
/// this scenario. Battery state is committed back to the caller only when
/// the complete scenario succeeds.
#[allow(clippy::too_many_arguments)]
pub fn try_run_scenario(
    battery: &mut Battery,
    tariff: &TariffSchedule,
    load_profile: impl Fn(f64) -> f64,
    generation_profile: impl Fn(f64) -> f64,
    dt_hours: f64,
    total_hours: f64,
    start_hour: f64,
    grid_available: bool,
    mut policy: impl FnMut(f64, f64, f64, &Battery) -> (f64, f64),
) -> Result<ScenarioResult, ScenarioError> {
    if !dt_hours.is_finite() || dt_hours <= 0.0 {
        return Err(ScenarioError::InvalidTimeStep);
    }
    if !total_hours.is_finite() || total_hours < 0.0 {
        return Err(ScenarioError::InvalidHorizon);
    }
    if !start_hour.is_finite() {
        return Err(ScenarioError::InvalidStartHour);
    }

    let tariff_values = [
        tariff.off_peak_price_per_kwh,
        tariff.peak_price_per_kwh,
        tariff.peak_start_hour,
        tariff.peak_end_hour,
        tariff.export_price_per_kwh,
    ];
    if !tariff_values.iter().all(|value| value.is_finite())
        || !(0.0..24.0).contains(&tariff.peak_start_hour)
        || !(0.0..24.0).contains(&tariff.peak_end_hour)
    {
        return Err(ScenarioError::InvalidTariff);
    }

    let battery_values = [
        battery.capacity_kwh,
        battery.power_rating_kw,
        battery.round_trip_efficiency,
        battery.soc(),
        battery.state_of_health(),
        battery.effective_capacity_kwh(),
        battery.equivalent_full_cycles(),
    ];
    if !battery_values.iter().all(|value| value.is_finite())
        || battery.capacity_kwh < 0.0
        || battery.power_rating_kw < 0.0
        || !(0.0..=1.0).contains(&battery.round_trip_efficiency)
        || !(0.0..=1.0).contains(&battery.soc())
        || !(0.0..=1.0).contains(&battery.state_of_health())
        || battery.effective_capacity_kwh() < 0.0
        || battery.equivalent_full_cycles() < 0.0
    {
        return Err(ScenarioError::InvalidBatteryConfiguration);
    }

    // Work on a clone so a later bad profile or policy output cannot leave
    // the caller's battery partially mutated by an unsuccessful scenario.
    let mut working_battery = battery.clone();
    let initial_cycles = working_battery.equivalent_full_cycles();
    let mut elapsed = 0.0;
    let mut total_cost = 0.0;
    let mut unserved_energy_kwh = 0.0;

    while elapsed < total_hours {
        let remaining_hours = total_hours - elapsed;
        let step_hours = dt_hours.min(remaining_hours);
        let final_step = dt_hours >= remaining_hours;
        if !final_step && elapsed + step_hours <= elapsed {
            // Extremely small dt relative to a very large horizon can stop
            // floating-point time from advancing, which would otherwise loop
            // forever. The clone keeps the caller's state unchanged on error.
            return Err(ScenarioError::InvalidTimeStep);
        }

        let t = start_hour + elapsed;
        if !t.is_finite() {
            return Err(ScenarioError::InvalidStartHour);
        }
        let load_kw = load_profile(t);
        let generation_kw = generation_profile(t);
        if !load_kw.is_finite()
            || load_kw < 0.0
            || !generation_kw.is_finite()
            || generation_kw < 0.0
        {
            return Err(ScenarioError::InvalidProfile);
        }

        let (charge_cmd_kw, discharge_cmd_kw) =
            policy(t, load_kw, generation_kw, &working_battery);
        if !charge_cmd_kw.is_finite()
            || charge_cmd_kw < 0.0
            || !discharge_cmd_kw.is_finite()
            || discharge_cmd_kw < 0.0
        {
            return Err(ScenarioError::InvalidSetpoint);
        }
        let charge_kw = charge_cmd_kw.clamp(0.0, working_battery.power_rating_kw);
        let discharge_kw = discharge_cmd_kw.clamp(0.0, working_battery.power_rating_kw);
        if charge_kw > 0.0 && discharge_kw > 0.0 {
            return Err(ScenarioError::SimultaneousChargeAndDischarge);
        }

        let charge_accepted_dc_kwh = working_battery
            .charge(charge_kw, step_hours)
            .map_err(|_| ScenarioError::InvalidBatteryConfiguration)?;
        let one_way_efficiency = working_battery.round_trip_efficiency.sqrt();
        let actual_charge_kw = if step_hours > 0.0 && one_way_efficiency > 0.0 {
            charge_accepted_dc_kwh / one_way_efficiency / step_hours
        } else {
            0.0
        };
        let discharge_delivered_ac_kwh = working_battery
            .discharge(discharge_kw, step_hours)
            .map_err(|_| ScenarioError::InvalidBatteryConfiguration)?;
        let served_kw_from_battery = discharge_delivered_ac_kwh / step_hours;

        // Use accepted charge energy, not the requested setpoint, in the
        // balance. A full battery must not appear to consume its requested
        // charging power when it accepted no energy.
        let net_kw = (load_kw + actual_charge_kw) - (generation_kw + served_kw_from_battery);
        if !net_kw.is_finite() {
            return Err(ScenarioError::NonFiniteResult);
        }
        if net_kw > 0.0 {
            if grid_available {
                total_cost += net_kw * step_hours * tariff.import_price(t);
            } else {
                unserved_energy_kwh += net_kw * step_hours;
            }
        } else if grid_available {
            total_cost -= (-net_kw) * step_hours * tariff.export_price_per_kwh;
        }
        if !total_cost.is_finite() || !unserved_energy_kwh.is_finite() {
            return Err(ScenarioError::NonFiniteResult);
        }

        elapsed = if final_step {
            total_hours
        } else {
            elapsed + step_hours
        };
    }

    let result = ScenarioResult {
        total_cost,
        unserved_energy_kwh,
        battery_cycles: working_battery.equivalent_full_cycles() - initial_cycles,
    };
    *battery = working_battery;
    Ok(result)
}

/// Baseline: greedy self-consumption with no look-ahead and no reserve
/// management. Charges with any solar surplus, discharges to cover any
/// shortfall, always -- even if that strands the battery empty right
/// before a predictable evening peak.
pub fn naive_greedy_policy(
    _t: f64,
    load_kw: f64,
    generation_kw: f64,
    _battery: &Battery,
) -> (f64, f64) {
    let net = generation_kw - load_kw;
    if net > 0.0 { (net, 0.0) } else { (0.0, -net) }
}

/// First concrete advisor policy: same greedy self-consumption, but holds
/// back a minimum state of charge during daylight hours so it isn't
/// stranded without storage for the (predictable) evening peak.
#[derive(Debug, Clone, Copy)]
pub struct ReserveAwarePolicy {
    pub daytime_reserve_soc: f64,
    pub day_start_hour: f64,
    pub day_end_hour: f64,
}

impl ReserveAwarePolicy {
    pub fn decide(
        &self,
        t: f64,
        load_kw: f64,
        generation_kw: f64,
        battery: &Battery,
    ) -> (f64, f64) {
        let net = generation_kw - load_kw;
        if net > 0.0 {
            return (net, 0.0);
        }
        let hour = t.rem_euclid(24.0);
        let is_daytime = hour >= self.day_start_hour && hour < self.day_end_hour;
        if is_daytime && battery.soc() <= self.daytime_reserve_soc {
            (0.0, 0.0) // hold reserve for the evening peak
        } else {
            (0.0, -net)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tariff() -> TariffSchedule {
        TariffSchedule {
            off_peak_price_per_kwh: 0.10,
            peak_price_per_kwh: 0.35,
            peak_start_hour: 17.0,
            peak_end_hour: 21.0,
            export_price_per_kwh: 0.05,
        }
    }

    /// Scenario: a brief midday generation dip (simulating a passing cloud)
    /// tempts the naive policy into partially draining the battery on a
    /// cheap off-peak shortfall, leaving less reserve for the (much more
    /// expensive) evening peak a few hours later. The reserve-aware policy
    /// holds back exactly for this reason. This is the presence-of-
    /// improvement test: the two policies must produce DIFFERENT, ranked
    /// outcomes on the same scenario, not merely both "work".
    fn load_profile(t: f64) -> f64 {
        let hour = t.rem_euclid(24.0);
        let evening_peak = if hour >= 17.0 && hour < 21.0 {
            30.0
        } else {
            0.0
        };
        10.0 + evening_peak
    }

    fn generation_profile_with_midday_dip(t: f64) -> f64 {
        let hour = t.rem_euclid(24.0);
        let base: f64 = if hour >= 8.0 && hour < 16.0 {
            25.0
        } else {
            0.0
        };
        // Cloud passes over 11:00-12:00: generation collapses to near zero
        // right when the naive policy would otherwise be idly floating.
        let cloud: f64 = if hour >= 11.0 && hour < 12.0 {
            20.0
        } else {
            0.0
        };
        (base - cloud).max(0.0)
    }

    #[test]
    fn test_reserve_aware_policy_beats_naive_on_cost_with_predictable_evening_peak() {
        let tariff = default_tariff();
        let dt_hours = 0.25; // 15-minute steps

        // Scenario window starts right at the 11:00 cloud dip (rather than
        // at midnight) and runs through the 21:00 end of the evening peak.
        // Starting at midnight was tried first and rejected: with this
        // load/battery sizing, BOTH policies fully drain the battery
        // overnight before the reserve mechanism's daytime window (6:00-
        // 17:00) even begins, so by the time the cloud dip arrives both
        // policies are already converged to the same (empty) state and
        // never actually diverge. Starting the clock at the dip itself,
        // with a battery state of charge already below the reserve
        // threshold, isolates the mechanism this test is actually about:
        // whether holding a cheap reserve now measurably pays off at the
        // expensive peak later.
        //
        // NOTE: an earlier version of this test tried to achieve the same
        // effect by wrapping `load_profile`/`generation_profile_with_midday_dip`
        // in `|t| profile(t + 11.0)` closures and passing `total_hours: 10.0`
        // starting from an internal t=0. That was a real bug, not just an
        // alternate style: `run_scenario`'s tariff lookup used its own
        // unshifted loop counter, so the "peak" 17:00-21:00 window (checked
        // against the RAW 0..10 range) never matched and every import was
        // silently priced off-peak regardless of the shifted profiles. Using
        // `run_scenario`'s `start_hour` parameter instead applies the shift
        // consistently to load, generation, tariff, and policy.
        const SCENARIO_START_HOUR: f64 = 11.0;
        let total_hours = 10.0; // 11:00 -> 21:00

        let mut naive_battery = Battery::new(100.0, 25.0, 0.90).with_soc(0.35);
        let naive_result = run_scenario(
            &mut naive_battery,
            &tariff,
            load_profile,
            generation_profile_with_midday_dip,
            dt_hours,
            total_hours,
            SCENARIO_START_HOUR,
            true,
            naive_greedy_policy,
        );

        let mut advisor_battery = Battery::new(100.0, 25.0, 0.90).with_soc(0.35);
        let advisor_policy = ReserveAwarePolicy {
            daytime_reserve_soc: 0.4,
            day_start_hour: 6.0,
            day_end_hour: 17.0,
        };
        let advisor_result = run_scenario(
            &mut advisor_battery,
            &tariff,
            load_profile,
            generation_profile_with_midday_dip,
            dt_hours,
            total_hours,
            SCENARIO_START_HOUR,
            true,
            |t, load_kw, generation_kw, battery| {
                advisor_policy.decide(t, load_kw, generation_kw, battery)
            },
        );

        assert!(
            advisor_result.total_cost < naive_result.total_cost,
            "reserve-aware policy should cost less than naive greedy: advisor={:?} naive={:?}",
            advisor_result,
            naive_result
        );
    }

    #[test]
    fn test_islanded_scenario_tracks_unserved_energy_not_cost() {
        let tariff = default_tariff();
        let mut battery = Battery::new(10.0, 5.0, 0.9).with_soc(0.1); // small, mostly-depleted battery
        let result = run_scenario(
            &mut battery,
            &tariff,
            |_t| 50.0, // load far exceeds any plausible generation+battery capacity
            |_t| 0.0,  // no generation at all
            1.0,
            5.0,
            0.0,
            false, // islanded -- no grid to fall back on
            naive_greedy_policy,
        );
        assert_eq!(
            result.total_cost, 0.0,
            "islanded scenarios accrue no grid cost"
        );
        assert!(
            result.unserved_energy_kwh > 0.0,
            "an undersized islanded system must show unserved demand, got {}",
            result.unserved_energy_kwh
        );
    }

    #[test]
    fn test_grid_tied_never_shows_unserved_energy() {
        let tariff = default_tariff();
        let mut battery = Battery::new(10.0, 5.0, 0.9).with_soc(0.1);
        let result = run_scenario(
            &mut battery,
            &tariff,
            |_t| 50.0,
            |_t| 0.0,
            1.0,
            5.0,
            0.0,
            true, // grid-tied -- infinite-bus assumption covers any shortfall
            naive_greedy_policy,
        );
        assert_eq!(result.unserved_energy_kwh, 0.0);
        assert!(
            result.total_cost > 0.0,
            "shortfall should show up as import cost instead"
        );
    }

    #[test]
    fn test_naive_policy_charges_on_surplus_and_discharges_on_shortfall() {
        let battery = Battery::new(50.0, 25.0, 0.9).with_soc(0.5);
        let (charge, discharge) = naive_greedy_policy(0.0, 10.0, 15.0, &battery);
        assert_eq!(charge, 5.0);
        assert_eq!(discharge, 0.0);

        let (charge, discharge) = naive_greedy_policy(0.0, 15.0, 10.0, &battery);
        assert_eq!(charge, 0.0);
        assert_eq!(discharge, 5.0);
    }

    #[test]
    fn test_reserve_aware_policy_withholds_discharge_below_reserve_during_day() {
        let policy = ReserveAwarePolicy {
            daytime_reserve_soc: 0.4,
            day_start_hour: 6.0,
            day_end_hour: 17.0,
        };
        let low_battery = Battery::new(50.0, 25.0, 0.9).with_soc(0.3); // below reserve
        let (charge, discharge) = policy.decide(10.0, 15.0, 10.0, &low_battery); // daytime, shortfall
        assert_eq!(charge, 0.0);
        assert_eq!(
            discharge, 0.0,
            "should hold reserve, not discharge, during the day below threshold"
        );
    }

    #[test]
    fn test_reserve_aware_policy_discharges_freely_at_night_regardless_of_reserve() {
        let policy = ReserveAwarePolicy {
            daytime_reserve_soc: 0.4,
            day_start_hour: 6.0,
            day_end_hour: 17.0,
        };
        let low_battery = Battery::new(50.0, 25.0, 0.9).with_soc(0.3); // below reserve
        let (charge, discharge) = policy.decide(20.0, 15.0, 0.0, &low_battery); // 20:00, nighttime, shortfall
        assert_eq!(charge, 0.0);
        assert_eq!(discharge, 15.0, "no reserve cap outside daytime hours");
    }
    #[test]
    fn test_tariff_supports_peak_windows_crossing_midnight() {
        let tariff = TariffSchedule {
            peak_start_hour: 22.0,
            peak_end_hour: 6.0,
            ..default_tariff()
        };
        assert_eq!(tariff.import_price(23.0), tariff.peak_price_per_kwh);
        assert_eq!(tariff.import_price(2.0), tariff.peak_price_per_kwh);
        assert_eq!(tariff.import_price(12.0), tariff.off_peak_price_per_kwh);
    }

    #[test]
    fn test_try_runner_rejects_invalid_time_steps_without_mutating_battery() {
        let tariff = default_tariff();
        for dt_hours in [0.0, -0.25, f64::NAN, f64::INFINITY] {
            let mut battery = Battery::new(10.0, 5.0, 0.9).with_soc(0.6);
            let before_soc = battery.soc();
            let result = try_run_scenario(
                &mut battery,
                &tariff,
                |_t| 1.0,
                |_t| 0.0,
                dt_hours,
                1.0,
                0.0,
                true,
                |_t, _load, _generation, _battery| (0.0, 0.0),
            );
            assert_eq!(result, Err(ScenarioError::InvalidTimeStep));
            assert_eq!(battery.soc(), before_soc);
        }
    }

    #[test]
    fn test_try_runner_rejects_invalid_horizon() {
        let tariff = default_tariff();
        for total_hours in [-1.0, f64::NAN, f64::INFINITY] {
            let mut battery = Battery::new(10.0, 5.0, 0.9);
            let result = try_run_scenario(
                &mut battery,
                &tariff,
                |_t| 1.0,
                |_t| 0.0,
                0.25,
                total_hours,
                0.0,
                true,
                |_t, _load, _generation, _battery| (0.0, 0.0),
            );
            assert_eq!(result, Err(ScenarioError::InvalidHorizon));
        }
    }

    #[test]
    fn test_non_divisible_horizon_uses_a_short_final_step() {
        let tariff = default_tariff();
        let mut battery = Battery::new(0.0, 0.0, 1.0);
        let result = run_scenario(
            &mut battery,
            &tariff,
            |_t| 1.0,
            |_t| 0.0,
            0.6,
            1.0,
            0.0,
            true,
            |_t, _load, _generation, _battery| (0.0, 0.0),
        );
        assert!(
            (result.total_cost - 0.1).abs() < 1e-9,
            "1 kW for exactly 1 hour should cost 0.10, got {}",
            result.total_cost
        );
    }

    #[test]
    fn test_full_battery_does_not_draw_unaccepted_charge_power() {
        let tariff = default_tariff();
        let mut battery = Battery::new(10.0, 5.0, 1.0).with_soc(1.0);
        let result = run_scenario(
            &mut battery,
            &tariff,
            |_t| 0.0,
            |_t| 5.0,
            1.0,
            1.0,
            0.0,
            true,
            |_t, _load, _generation, _battery| (5.0, 0.0),
        );
        assert!(
            (result.total_cost + 0.25).abs() < 1e-9,
            "surplus from a full battery should be exportable, got {}",
            result.total_cost
        );
    }

    #[test]
    fn test_result_reports_only_cycles_accumulated_in_this_scenario() {
        let tariff = default_tariff();
        let mut battery = Battery::new(10.0, 5.0, 1.0).with_soc(0.5);
        battery.charge(5.0, 1.0).unwrap();
        battery.discharge(5.0, 1.0).unwrap();
        assert!(battery.equivalent_full_cycles() > 0.0);

        let result = run_scenario(
            &mut battery,
            &tariff,
            |_t| 0.0,
            |_t| 0.0,
            1.0,
            1.0,
            0.0,
            true,
            |_t, _load, _generation, _battery| (0.0, 0.0),
        );
        assert_eq!(result.battery_cycles, 0.0);
    }

    #[test]
    fn test_invalid_profile_is_rejected_without_committing_partial_battery_state() {
        let tariff = default_tariff();
        let mut battery = Battery::new(10.0, 5.0, 0.9).with_soc(0.5);
        let before_soc = battery.soc();
        let before_cycles = battery.equivalent_full_cycles();
        let result = try_run_scenario(
            &mut battery,
            &tariff,
            |_t| f64::NAN,
            |_t| 0.0,
            0.25,
            1.0,
            0.0,
            true,
            |_t, _load, _generation, _battery| (0.0, 1.0),
        );
        assert_eq!(result, Err(ScenarioError::InvalidProfile));
        assert_eq!(battery.soc(), before_soc);
        assert_eq!(battery.equivalent_full_cycles(), before_cycles);
    }

    #[test]
    fn test_simultaneous_charge_and_discharge_is_rejected_atomically() {
        let tariff = default_tariff();
        let mut battery = Battery::new(10.0, 5.0, 0.9).with_soc(0.5);
        let before_soc = battery.soc();
        let result = try_run_scenario(
            &mut battery,
            &tariff,
            |_t| 1.0,
            |_t| 1.0,
            0.25,
            1.0,
            0.0,
            true,
            |_t, _load, _generation, _battery| (1.0, 1.0),
        );
        assert_eq!(
            result,
            Err(ScenarioError::SimultaneousChargeAndDischarge)
        );
        assert_eq!(battery.soc(), before_soc);
    }

}
