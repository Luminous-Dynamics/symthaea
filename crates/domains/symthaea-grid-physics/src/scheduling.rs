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
        if hour >= self.peak_start_hour && hour < self.peak_end_hour {
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
pub fn run_scenario(
    battery: &mut Battery,
    tariff: &TariffSchedule,
    load_profile: impl Fn(f64) -> f64,
    generation_profile: impl Fn(f64) -> f64,
    dt_hours: f64,
    total_hours: f64,
    start_hour: f64,
    grid_available: bool,
    mut policy: impl FnMut(f64, f64, f64, &Battery) -> (f64, f64),
) -> ScenarioResult {
    let mut elapsed = 0.0;
    let mut total_cost = 0.0;
    let mut unserved_energy_kwh = 0.0;
    while elapsed < total_hours {
        let t = start_hour + elapsed;
        let load_kw = load_profile(t);
        let generation_kw = generation_profile(t);
        let (charge_cmd_kw, discharge_cmd_kw) = policy(t, load_kw, generation_kw, battery);
        let charge_kw = charge_cmd_kw.clamp(0.0, battery.power_rating_kw);
        let discharge_kw = discharge_cmd_kw.clamp(0.0, battery.power_rating_kw);
        let _ = battery.charge(charge_kw, dt_hours);
        let discharge_delivered_ac_kwh = battery.discharge(discharge_kw, dt_hours).unwrap_or(0.0);
        let served_kw_from_battery = discharge_delivered_ac_kwh / dt_hours;
        // Power balance: local demand is load PLUS whatever the battery is
        // drawing to charge; local supply is generation PLUS battery
        // discharge. Omitting charge_kw here previously double-counted
        // surplus generation as export credit even while that same surplus
        // was simultaneously being routed into the battery.
        let net_kw = (load_kw + charge_kw) - (generation_kw + served_kw_from_battery);
        if net_kw > 0.0 {
            if grid_available {
                total_cost += net_kw * dt_hours * tariff.import_price(t);
            } else {
                unserved_energy_kwh += net_kw * dt_hours;
            }
        } else if grid_available {
            total_cost -= (-net_kw) * dt_hours * tariff.export_price_per_kwh;
        }
        elapsed += dt_hours;
    }
    ScenarioResult {
        total_cost,
        unserved_energy_kwh,
        battery_cycles: battery.equivalent_full_cycles(),
    }
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
}
