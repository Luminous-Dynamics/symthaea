// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trajectory planning as a cognitive task (Phase 3 of
//! `SPACE_AUTOMATION_PLAN_2026-07-06.md`) -- the honest "grav-craft":
//! optimizing WHEN and HOW to burn within known physics, never a claim
//! about modifying gravity or propulsion itself.
//!
//! # A documentation trap this module deliberately avoids
//!
//! `orbital_mechanics::LambertSolution::delta_v_total` is `|v1| + |v2|` --
//! the raw speed magnitudes of the transfer orbit's own endpoints, NOT the
//! maneuver cost relative to whatever orbit you're actually on. Naively
//! minimizing that field would silently plan the wrong transfer for any
//! real rendezvous (it's only correct for departure/arrival from/to rest).
//! Every search here computes the real delta-v explicitly:
//! `(v1 - v_departure_actual).norm() + (v2 - v_arrival_actual).norm()`,
//! the same thing this crate's own `test_lambert_hohmann_agreement`
//! computes rather than trusting `delta_v_total`. See that field's doc
//! comment in `orbital-mechanics` for the full story.
//!
//! Public API here uses plain `[f64; 3]`, converting to
//! `nalgebra::Vector3` only internally at the `solve_lambert()` call site
//! -- same "convert at the boundary" pattern used elsewhere in this crate,
//! since `nalgebra` is pinned to 0.32 here specifically to match
//! `orbital-mechanics`'s own resolved version (symthaea's workspace default
//! is 0.34, a different, incompatible `Vector3` type despite the name).

use nalgebra::Vector3;
use orbital_mechanics::coordinates::wgs84::MU;
use orbital_mechanics::solve_lambert;

#[derive(Debug, Clone, Copy)]
pub struct TransferCandidate {
    pub tof_s: f64,
    pub delta_v_kms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TransferPlanResult {
    pub best: TransferCandidate,
    pub candidates_evaluated: usize,
    pub candidates_solved: usize,
}

/// Grid-search over time-of-flight to find the Lambert transfer between
/// two fixed position vectors minimizing REAL delta-v (see module docs) --
/// the caller supplies the actual departure/arrival velocities (e.g. from
/// circular-orbit speeds, or any other actual orbital state), not just
/// positions. Skips TOF values Lambert can't solve (e.g. too short for the
/// geometry) rather than failing the whole search.
#[allow(clippy::too_many_arguments)]
pub fn plan_min_delta_v_transfer(
    r1_km: [f64; 3],
    v1_actual_kms: [f64; 3],
    r2_km: [f64; 3],
    v2_actual_kms: [f64; 3],
    tof_min_s: f64,
    tof_max_s: f64,
    tof_step_s: f64,
    clockwise: bool,
) -> Option<TransferPlanResult> {
    let r1 = Vector3::new(r1_km[0], r1_km[1], r1_km[2]);
    let r2 = Vector3::new(r2_km[0], r2_km[1], r2_km[2]);
    let v1_actual = Vector3::new(v1_actual_kms[0], v1_actual_kms[1], v1_actual_kms[2]);
    let v2_actual = Vector3::new(v2_actual_kms[0], v2_actual_kms[1], v2_actual_kms[2]);

    let mut best: Option<TransferCandidate> = None;
    let mut evaluated = 0usize;
    let mut solved = 0usize;
    let mut tof = tof_min_s;

    while tof <= tof_max_s {
        evaluated += 1;
        if let Ok(sol) = solve_lambert(&r1, &r2, tof, MU, clockwise) {
            solved += 1;
            let delta_v = (sol.v1 - v1_actual).norm() + (sol.v2 - v2_actual).norm();
            if best.map(|b| delta_v < b.delta_v_kms).unwrap_or(true) {
                best = Some(TransferCandidate {
                    tof_s: tof,
                    delta_v_kms: delta_v,
                });
            }
        }
        tof += tof_step_s;
    }

    best.map(|best| TransferPlanResult {
        best,
        candidates_evaluated: evaluated,
        candidates_solved: solved,
    })
}

use orbital_mechanics::solar_system::{get_body, mu_km3_s2};
use orbital_mechanics::{
    Planet, apply_gravity_assist, heliocentric_position_km, heliocentric_velocity_kms, julian_day,
};

#[derive(Debug, Clone, Copy)]
pub struct GravityAssistCandidate {
    pub tof_days: f64,
    /// Delta-v to leave Earth's actual heliocentric velocity and enter the
    /// transfer orbit, km/s -- the only propellant actually spent; the
    /// flyby itself costs nothing.
    pub departure_delta_v_kms: f64,
    pub incoming_heliocentric_speed_kms: f64,
    pub outgoing_heliocentric_speed_kms: f64,
    pub speed_gained_kms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct GravityAssistPlanResult {
    pub departure_jd: f64,
    pub best: GravityAssistCandidate,
    pub candidates_evaluated: usize,
    pub candidates_solved: usize,
}

/// Real Earth -> Jupiter transfer (via Lambert, using REAL VSOP87 ephemeris
/// positions for departure and each candidate arrival date -- Jupiter's
/// position genuinely depends on which arrival date is tried, unlike the
/// fixed-endpoint LEO->GEO case above) followed by a real patched-conic
/// Jupiter flyby. Grid-searches time-of-flight, picking the candidate that
/// minimizes the departure burn -- the only propellant actually spent,
/// since the gravity assist itself is free regardless of which TOF is
/// chosen.
///
/// This is the honest "grav-craft" the space-automation plan asks for:
/// real ephemeris + real Lambert + real patched-conic flyby physics,
/// demonstrating that a trailing-side Jupiter encounter genuinely boosts
/// heliocentric speed using gravity that already exists there, at zero
/// extra propellant cost. It is a SINGLE-flyby demo, not a full multi-leg
/// mission-design search (jointly optimizing periapsis + TOF across a
/// chain of several flybys to reach a specific final target, the way real
/// missions like Voyager were designed) -- see module docs for that scope
/// boundary. The departure date is an arbitrary real calendar date, not a
/// claim about matching any actual historical mission's launch window.
#[allow(clippy::too_many_arguments)]
pub fn plan_earth_jupiter_gravity_assist(
    departure_year: i16,
    departure_month: u8,
    departure_day: f64,
    tof_min_days: f64,
    tof_max_days: f64,
    tof_step_days: f64,
    periapsis_radius_km: f64,
    leading_side: bool,
) -> Option<GravityAssistPlanResult> {
    let jd_dep = julian_day(departure_year, departure_month, departure_day);
    let r_earth = heliocentric_position_km(&Planet::Earth, jd_dep);
    let v_earth = heliocentric_velocity_kms(&Planet::Earth, jd_dep);

    let mu_sun = mu_km3_s2(&get_body("Sun").expect("Sun is always in the catalog"));
    let mu_jupiter = mu_km3_s2(&get_body("Jupiter").expect("Jupiter is always in the catalog"));

    let mut best: Option<GravityAssistCandidate> = None;
    let mut evaluated = 0usize;
    let mut solved = 0usize;
    let mut tof_days = tof_min_days;

    while tof_days <= tof_max_days {
        evaluated += 1;
        let jd_arrival = jd_dep + tof_days;
        let r_jupiter = heliocentric_position_km(&Planet::Jupiter, jd_arrival);
        let v_jupiter = heliocentric_velocity_kms(&Planet::Jupiter, jd_arrival);
        let tof_s = tof_days * 86400.0;

        // Try both the short-way and long-way transfer at this TOF and keep
        // whichever is cheaper -- standard practice for real Lambert-based
        // mission planning. Which sense is actually optimal depends on the
        // real geometric transfer angle between Earth's and Jupiter's actual
        // positions on this specific date, which varies date-to-date; fixing
        // this to a single sense (as an earlier version of this function did)
        // silently plans a wildly suboptimal transfer whenever the real
        // geometry doesn't happen to favor that one sense.
        let mut solved_this_tof = false;
        for clockwise in [false, true] {
            if let Ok(sol) = solve_lambert(&r_earth, &r_jupiter, tof_s, mu_sun, clockwise) {
                solved_this_tof = true;
                let departure_delta_v = (sol.v1 - v_earth).norm();
                let incoming_speed = sol.v2.norm();
                let v_out = apply_gravity_assist(
                    sol.v2,
                    v_jupiter,
                    periapsis_radius_km,
                    mu_jupiter,
                    leading_side,
                );
                let outgoing_speed = v_out.norm();

                let candidate = GravityAssistCandidate {
                    tof_days,
                    departure_delta_v_kms: departure_delta_v,
                    incoming_heliocentric_speed_kms: incoming_speed,
                    outgoing_heliocentric_speed_kms: outgoing_speed,
                    speed_gained_kms: outgoing_speed - incoming_speed,
                };
                if best
                    .map(|b| candidate.departure_delta_v_kms < b.departure_delta_v_kms)
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }
        }
        if solved_this_tof {
            solved += 1;
        }
        tof_days += tof_step_days;
    }

    best.map(|best| GravityAssistPlanResult {
        departure_jd: jd_dep,
        best,
        candidates_evaluated: evaluated,
        candidates_solved: solved,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct PeriapsisCandidate {
    pub periapsis_radius_km: f64,
    pub speed_gained_kms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct PeriapsisSearchResult {
    pub best: PeriapsisCandidate,
    /// Jupiter's actual physical radius, km -- the hard floor below which
    /// "periapsis" means intersecting the planet. Not a safety margin;
    /// the caller supplies that by choosing `periapsis_min_km` above this.
    pub safety_floor_km: f64,
    pub candidates_evaluated: usize,
}

/// For a FIXED Earth->Jupiter transfer (a departure date + TOF -- e.g. the
/// optimum found by `plan_earth_jupiter_gravity_assist`), grid-search
/// periapsis radius to find the value that maximizes heliocentric speed
/// gained from the flyby.
///
/// This is the real "closer periapsis buys more free delta-v, but costs
/// safety margin and radiation exposure" trade actual mission designers
/// navigate: Galileo and Juno both chose distant periapsis specifically to
/// survive Jupiter's radiation belts, even though a closer pass would have
/// bent the trajectory (and thus the heliocentric speed change) more. A
/// pure speed-gain search has no such constraint built in, so expect (and
/// this is a real, honest finding, not a bug) that the "best" result lands
/// at or near `periapsis_min_km` every time -- the turn angle in
/// `gravity_assist::turn_angle_rad` strictly increases as periapsis
/// decreases, for any fixed incoming v-infinity. `periapsis_min_km` is
/// clamped up to Jupiter's actual physical radius (`safety_floor_km`) --
/// a hard floor, not a preference; the caller is responsible for adding
/// whatever real safety/radiation margin above that they actually want
/// searched.
#[allow(clippy::too_many_arguments)]
pub fn search_periapsis_for_max_speed_gain(
    departure_year: i16,
    departure_month: u8,
    departure_day: f64,
    tof_days: f64,
    periapsis_min_km: f64,
    periapsis_max_km: f64,
    periapsis_step_km: f64,
    leading_side: bool,
) -> Option<PeriapsisSearchResult> {
    let jd_dep = julian_day(departure_year, departure_month, departure_day);
    let r_earth = heliocentric_position_km(&Planet::Earth, jd_dep);
    let jd_arrival = jd_dep + tof_days;
    let r_jupiter = heliocentric_position_km(&Planet::Jupiter, jd_arrival);
    let v_jupiter = heliocentric_velocity_kms(&Planet::Jupiter, jd_arrival);

    let mu_sun = mu_km3_s2(&get_body("Sun").expect("Sun is always in the catalog"));
    let jupiter_body = get_body("Jupiter").expect("Jupiter is always in the catalog");
    let mu_jupiter = mu_km3_s2(&jupiter_body);
    let safety_floor_km = jupiter_body.radius_m / 1000.0;

    let tof_s = tof_days * 86400.0;
    let sol = solve_lambert(&r_earth, &r_jupiter, tof_s, mu_sun, false).ok()?;
    let incoming_speed = sol.v2.norm();

    let mut best: Option<PeriapsisCandidate> = None;
    let mut evaluated = 0usize;
    let mut periapsis_km = periapsis_min_km.max(safety_floor_km);

    while periapsis_km <= periapsis_max_km {
        evaluated += 1;
        let v_out = apply_gravity_assist(sol.v2, v_jupiter, periapsis_km, mu_jupiter, leading_side);
        let candidate = PeriapsisCandidate {
            periapsis_radius_km: periapsis_km,
            speed_gained_kms: v_out.norm() - incoming_speed,
        };
        if best
            .map(|b| candidate.speed_gained_kms > b.speed_gained_kms)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
        periapsis_km += periapsis_step_km;
    }

    best.map(|best| PeriapsisSearchResult {
        best,
        safety_floor_km,
        candidates_evaluated: evaluated,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct ChainedFlybyCandidate {
    pub periapsis_radius_km: f64,
    pub leading_side: bool,
    pub achieved_outgoing_speed_kms: f64,
    pub required_departure_speed_kms: f64,
    /// Vector-space gap (km/s) between the velocity the free (unpowered)
    /// flyby actually delivers and the velocity the second-leg Lambert
    /// transfer requires. This is the delta-v a deep-space maneuver at
    /// Jupiter would still need to supply to perfectly connect the two
    /// legs -- zero would mean the free flyby alone threads the needle.
    pub connection_gap_kms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct EarthJupiterSaturnChainResult {
    pub departure_jd: f64,
    pub jupiter_arrival_jd: f64,
    pub saturn_arrival_jd: f64,
    pub leg1_departure_delta_v_kms: f64,
    pub leg2_tof_days: f64,
    pub best: ChainedFlybyCandidate,
    pub candidates_evaluated: usize,
}

/// Earth -> Jupiter (gravity assist) -> Saturn: the two-leg chain the
/// space-automation plan names as future scope ("rediscover a
/// Voyager-class route"), scoped honestly to what a single free flyby can
/// actually achieve.
///
/// Leg 1 (Earth->Jupiter) is solved exactly as in
/// `plan_earth_jupiter_gravity_assist`, at a FIXED departure date + TOF
/// the caller supplies (e.g. the optimum that function already found).
/// Leg 2 (Jupiter->Saturn) is solved as a SEPARATE real Lambert transfer,
/// grid-searched over TOF2, giving the velocity a spacecraft would need to
/// depart Jupiter with to reach Saturn's real position at the resulting
/// arrival date.
///
/// The free (unpowered) flyby modeled by `apply_gravity_assist` can only
/// deliver ONE outgoing velocity per (periapsis, leading/trailing) choice
/// -- it rotates the incoming v-infinity within a single plane fixed by
/// the incoming-relative-velocity and Jupiter's own velocity vectors (see
/// `gravity_assist` module docs), NOT full 3D b-plane targeting the way a
/// real mission's flyby is aimed. It cannot be steered to hit an
/// arbitrary target velocity. This function grid-searches periapsis
/// (real physical floor: must clear Jupiter's actual radius) x
/// leading/trailing side x leg-2 TOF, and reports the combination that
/// gets the ACHIEVED flyby velocity closest to the REQUIRED leg-2
/// velocity. A small `connection_gap_kms` means a free flyby alone
/// threads the needle close to a real mission design; a large gap is an
/// honest signal that this simplified single-plane flyby model cannot
/// freely target Saturn the way a real b-plane-targeted flyby (optionally
/// plus a deep-space maneuver) would -- it does not mean the chain
/// "failed," only that connecting it perfectly for free is not always
/// possible with two degrees of freedom (periapsis + side).
#[allow(clippy::too_many_arguments)]
pub fn plan_earth_jupiter_saturn_chain(
    departure_year: i16,
    departure_month: u8,
    departure_day: f64,
    leg1_tof_days: f64,
    leg2_tof_min_days: f64,
    leg2_tof_max_days: f64,
    leg2_tof_step_days: f64,
    periapsis_min_km: f64,
    periapsis_max_km: f64,
    periapsis_step_km: f64,
) -> Option<EarthJupiterSaturnChainResult> {
    let jd_dep = julian_day(departure_year, departure_month, departure_day);
    let r_earth = heliocentric_position_km(&Planet::Earth, jd_dep);
    let v_earth = heliocentric_velocity_kms(&Planet::Earth, jd_dep);

    let jd_jupiter = jd_dep + leg1_tof_days;
    let r_jupiter = heliocentric_position_km(&Planet::Jupiter, jd_jupiter);
    let v_jupiter = heliocentric_velocity_kms(&Planet::Jupiter, jd_jupiter);

    let mu_sun = mu_km3_s2(&get_body("Sun").expect("Sun is always in the catalog"));
    let jupiter_body = get_body("Jupiter").expect("Jupiter is always in the catalog");
    let mu_jupiter = mu_km3_s2(&jupiter_body);
    let jupiter_radius_km = jupiter_body.radius_m / 1000.0;

    // Leg 1: Earth -> Jupiter at the caller-supplied fixed departure/TOF.
    let leg1_tof_s = leg1_tof_days * 86400.0;
    let sol1 = solve_lambert(&r_earth, &r_jupiter, leg1_tof_s, mu_sun, false).ok()?;
    let leg1_departure_delta_v = (sol1.v1 - v_earth).norm();
    let v_incoming_jupiter = sol1.v2;

    // Hard physical floor: periapsis cannot go below Jupiter's actual
    // radius. This does NOT enforce a real safety/radiation margin -- the
    // caller supplies that by choosing periapsis_min_km appropriately.
    let periapsis_min_km = periapsis_min_km.max(jupiter_radius_km);

    let mut best: Option<(f64, ChainedFlybyCandidate)> = None;
    let mut evaluated = 0usize;
    let mut leg2_tof_days = leg2_tof_min_days;

    while leg2_tof_days <= leg2_tof_max_days {
        let jd_saturn = jd_jupiter + leg2_tof_days;
        let r_saturn = heliocentric_position_km(&Planet::Saturn, jd_saturn);
        let leg2_tof_s = leg2_tof_days * 86400.0;

        if let Ok(sol2) = solve_lambert(&r_jupiter, &r_saturn, leg2_tof_s, mu_sun, false) {
            let required_v = sol2.v1;
            let required_speed = required_v.norm();

            let mut periapsis_km = periapsis_min_km;
            while periapsis_km <= periapsis_max_km {
                for leading_side in [false, true] {
                    evaluated += 1;
                    let v_out = apply_gravity_assist(
                        v_incoming_jupiter,
                        v_jupiter,
                        periapsis_km,
                        mu_jupiter,
                        leading_side,
                    );
                    let gap = (v_out - required_v).norm();
                    let candidate = ChainedFlybyCandidate {
                        periapsis_radius_km: periapsis_km,
                        leading_side,
                        achieved_outgoing_speed_kms: v_out.norm(),
                        required_departure_speed_kms: required_speed,
                        connection_gap_kms: gap,
                    };
                    if best
                        .as_ref()
                        .map(|(_, b)| candidate.connection_gap_kms < b.connection_gap_kms)
                        .unwrap_or(true)
                    {
                        best = Some((leg2_tof_days, candidate));
                    }
                }
                periapsis_km += periapsis_step_km;
            }
        }
        leg2_tof_days += leg2_tof_step_days;
    }

    best.map(|(leg2_tof_days, best)| EarthJupiterSaturnChainResult {
        departure_jd: jd_dep,
        jupiter_arrival_jd: jd_jupiter,
        saturn_arrival_jd: jd_jupiter + leg2_tof_days,
        leg1_departure_delta_v_kms: leg1_departure_delta_v,
        leg2_tof_days,
        best,
        candidates_evaluated: evaluated,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct EarthJupiterSaturnUranusChainResult {
    pub departure_jd: f64,
    pub jupiter_arrival_jd: f64,
    pub saturn_arrival_jd: f64,
    pub uranus_arrival_jd: f64,
    pub leg1_departure_delta_v_kms: f64,
    pub leg2_tof_days: f64,
    pub leg3_tof_days: f64,
    pub jupiter_flyby: ChainedFlybyCandidate,
    pub saturn_flyby: ChainedFlybyCandidate,
    pub candidates_evaluated: usize,
}

/// Earth -> Jupiter (flyby) -> Saturn (flyby) -> Uranus: a genuine TWO-FLYBY
/// chain (Voyager 2's actual "grand tour" sequence visited Jupiter, Saturn,
/// Uranus, and Neptune in that order), extending
/// `plan_earth_jupiter_saturn_chain` by one more real gravity assist.
///
/// This is a GREEDY, leg-by-leg composition, NOT a jointly-optimized
/// multi-flyby trajectory: it first finds the Jupiter flyby (periapsis +
/// side) and Jupiter->Saturn TOF that minimizes the Jupiter connection gap
/// -- exactly what `plan_earth_jupiter_saturn_chain` does -- then, GIVEN
/// that choice, assumes a deep-space maneuver perfectly corrects the
/// trajectory onto the intended Jupiter->Saturn Lambert arc (the standard
/// patched-conic assumption real interplanetary missions like Cassini and
/// Galileo actually rely on small trajectory-correction maneuvers for), and
/// independently searches the Saturn flyby (periapsis + side) and
/// Saturn->Uranus TOF that minimizes the SECOND connection gap. A truly
/// joint optimization across both flybys simultaneously could in principle
/// do better than this greedy composition -- that is real, explicitly
/// out-of-scope future work, not something this function claims to already
/// do. J2000.0 (or whatever date the caller supplies) is an arbitrary real
/// calendar date, not a claim about matching Voyager 2's actual 1977
/// launch window or trajectory.
#[allow(clippy::too_many_arguments)]
pub fn plan_earth_jupiter_saturn_uranus_chain(
    departure_year: i16,
    departure_month: u8,
    departure_day: f64,
    leg1_tof_days: f64,
    leg2_tof_min_days: f64,
    leg2_tof_max_days: f64,
    leg2_tof_step_days: f64,
    leg3_tof_min_days: f64,
    leg3_tof_max_days: f64,
    leg3_tof_step_days: f64,
    jupiter_periapsis_min_km: f64,
    jupiter_periapsis_max_km: f64,
    jupiter_periapsis_step_km: f64,
    saturn_periapsis_min_km: f64,
    saturn_periapsis_max_km: f64,
    saturn_periapsis_step_km: f64,
) -> Option<EarthJupiterSaturnUranusChainResult> {
    let jd_dep = julian_day(departure_year, departure_month, departure_day);
    let r_earth = heliocentric_position_km(&Planet::Earth, jd_dep);
    let v_earth = heliocentric_velocity_kms(&Planet::Earth, jd_dep);

    let jd_jupiter = jd_dep + leg1_tof_days;
    let r_jupiter = heliocentric_position_km(&Planet::Jupiter, jd_jupiter);
    let v_jupiter = heliocentric_velocity_kms(&Planet::Jupiter, jd_jupiter);

    let mu_sun = mu_km3_s2(&get_body("Sun").expect("Sun is always in the catalog"));
    let jupiter_body = get_body("Jupiter").expect("Jupiter is always in the catalog");
    let mu_jupiter = mu_km3_s2(&jupiter_body);
    let jupiter_radius_km = jupiter_body.radius_m / 1000.0;
    let saturn_body = get_body("Saturn").expect("Saturn is always in the catalog");
    let mu_saturn = mu_km3_s2(&saturn_body);
    let saturn_radius_km = saturn_body.radius_m / 1000.0;

    // Leg 1: Earth -> Jupiter (fixed, caller-supplied).
    let leg1_tof_s = leg1_tof_days * 86400.0;
    let sol1 = solve_lambert(&r_earth, &r_jupiter, leg1_tof_s, mu_sun, false).ok()?;
    let leg1_departure_delta_v = (sol1.v1 - v_earth).norm();
    let v_incoming_jupiter = sol1.v2;

    // Stage A: Jupiter flyby -- search leg2 TOF x periapsis x side to
    // minimize the connection gap to the Jupiter->Saturn Lambert arc.
    let jupiter_periapsis_min_km = jupiter_periapsis_min_km.max(jupiter_radius_km);
    let mut best_a: Option<(f64, ChainedFlybyCandidate, Vector3<f64>, Vector3<f64>)> = None;
    let mut evaluated = 0usize;
    let mut leg2_tof_days = leg2_tof_min_days;

    while leg2_tof_days <= leg2_tof_max_days {
        let jd_saturn = jd_jupiter + leg2_tof_days;
        let r_saturn = heliocentric_position_km(&Planet::Saturn, jd_saturn);
        let leg2_tof_s = leg2_tof_days * 86400.0;

        if let Ok(sol2) = solve_lambert(&r_jupiter, &r_saturn, leg2_tof_s, mu_sun, false) {
            let required_v = sol2.v1;
            let required_speed = required_v.norm();

            let mut periapsis_km = jupiter_periapsis_min_km;
            while periapsis_km <= jupiter_periapsis_max_km {
                for leading_side in [false, true] {
                    evaluated += 1;
                    let v_out = apply_gravity_assist(
                        v_incoming_jupiter,
                        v_jupiter,
                        periapsis_km,
                        mu_jupiter,
                        leading_side,
                    );
                    let gap = (v_out - required_v).norm();
                    let candidate = ChainedFlybyCandidate {
                        periapsis_radius_km: periapsis_km,
                        leading_side,
                        achieved_outgoing_speed_kms: v_out.norm(),
                        required_departure_speed_kms: required_speed,
                        connection_gap_kms: gap,
                    };
                    if best_a
                        .as_ref()
                        .map(|(_, b, _, _)| candidate.connection_gap_kms < b.connection_gap_kms)
                        .unwrap_or(true)
                    {
                        best_a = Some((leg2_tof_days, candidate, r_saturn, sol2.v2));
                    }
                }
                periapsis_km += jupiter_periapsis_step_km;
            }
        }
        leg2_tof_days += leg2_tof_step_days;
    }
    let (leg2_tof_days, jupiter_flyby, r_saturn, v_incoming_saturn) = best_a?;
    let jd_saturn = jd_jupiter + leg2_tof_days;
    let v_saturn = heliocentric_velocity_kms(&Planet::Saturn, jd_saturn);

    // Stage B: Saturn flyby -- search leg3 TOF x periapsis x side to
    // minimize the connection gap to the Saturn->Uranus Lambert arc. Uses
    // v_incoming_saturn (the leg-2 Lambert arc's arrival velocity), per the
    // deep-space-maneuver assumption documented above.
    let saturn_periapsis_min_km = saturn_periapsis_min_km.max(saturn_radius_km);
    let mut best_b: Option<(f64, ChainedFlybyCandidate)> = None;
    let mut leg3_tof_days = leg3_tof_min_days;

    while leg3_tof_days <= leg3_tof_max_days {
        let jd_uranus = jd_saturn + leg3_tof_days;
        let r_uranus = heliocentric_position_km(&Planet::Uranus, jd_uranus);
        let leg3_tof_s = leg3_tof_days * 86400.0;

        if let Ok(sol3) = solve_lambert(&r_saturn, &r_uranus, leg3_tof_s, mu_sun, false) {
            let required_v = sol3.v1;
            let required_speed = required_v.norm();

            let mut periapsis_km = saturn_periapsis_min_km;
            while periapsis_km <= saturn_periapsis_max_km {
                for leading_side in [false, true] {
                    evaluated += 1;
                    let v_out = apply_gravity_assist(
                        v_incoming_saturn,
                        v_saturn,
                        periapsis_km,
                        mu_saturn,
                        leading_side,
                    );
                    let gap = (v_out - required_v).norm();
                    let candidate = ChainedFlybyCandidate {
                        periapsis_radius_km: periapsis_km,
                        leading_side,
                        achieved_outgoing_speed_kms: v_out.norm(),
                        required_departure_speed_kms: required_speed,
                        connection_gap_kms: gap,
                    };
                    if best_b
                        .as_ref()
                        .map(|(_, b)| candidate.connection_gap_kms < b.connection_gap_kms)
                        .unwrap_or(true)
                    {
                        best_b = Some((leg3_tof_days, candidate));
                    }
                }
                periapsis_km += saturn_periapsis_step_km;
            }
        }
        leg3_tof_days += leg3_tof_step_days;
    }
    let (leg3_tof_days, saturn_flyby) = best_b?;

    Some(EarthJupiterSaturnUranusChainResult {
        departure_jd: jd_dep,
        jupiter_arrival_jd: jd_jupiter,
        saturn_arrival_jd: jd_saturn,
        uranus_arrival_jd: jd_saturn + leg3_tof_days,
        leg1_departure_delta_v_kms: leg1_departure_delta_v,
        leg2_tof_days,
        leg3_tof_days,
        jupiter_flyby,
        saturn_flyby,
        candidates_evaluated: evaluated,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct JointTwoFlybyResult {
    pub departure_jd: f64,
    pub jupiter_arrival_jd: f64,
    pub saturn_arrival_jd: f64,
    pub uranus_arrival_jd: f64,
    pub leg1_departure_delta_v_kms: f64,
    pub leg2_tof_days: f64,
    pub leg3_tof_days: f64,
    pub jupiter_flyby: ChainedFlybyCandidate,
    pub saturn_flyby: ChainedFlybyCandidate,
    /// jupiter_flyby.connection_gap_kms + saturn_flyby.connection_gap_kms
    /// -- the JOINT objective this function actually minimizes, unlike
    /// `plan_earth_jupiter_saturn_uranus_chain`'s greedy composition (which
    /// minimizes each flyby's own gap independently and lives with
    /// whatever that choice of leg-2 TOF leaves for the Saturn leg).
    pub total_connection_gap_kms: f64,
    pub candidates_evaluated: usize,
}

/// Same Earth->Jupiter->Saturn->Uranus route as
/// `plan_earth_jupiter_saturn_uranus_chain`, but JOINTLY optimizing the
/// choice of Jupiter->Saturn time-of-flight (leg 2) against the TOTAL
/// connection gap (Jupiter's + Saturn's combined), instead of picking leg
/// 2's TOF to minimize only the Jupiter flyby's own gap and then living
/// with whatever Saturn connection that choice happens to leave.
///
/// # Why this is tractable without a full 6-dimensional grid search
///
/// The two flybys' connection gaps are coupled ONLY through the shared
/// choice of leg-2 TOF: under the deep-space-maneuver assumption (see
/// `plan_earth_jupiter_saturn_uranus_chain`'s docs), the velocity handed
/// off to the Saturn flyby is always the leg-2 Lambert arc's own arrival
/// velocity, `sol2.v2` -- which depends on leg-2 TOF but NOT on which
/// periapsis/side the Jupiter flyby itself used. This means, for any FIXED
/// leg-2 TOF, the Jupiter periapsis/side search and the Saturn
/// periapsis/side/leg-3-TOF search are fully independent and can each be
/// minimized separately -- exactly what this function does, once per
/// candidate leg-2 TOF -- and then leg-2 TOF itself is searched to
/// minimize the SUM of both independently-minimized gaps. This is the true
/// joint optimum over all six real degrees of freedom (2 periapsides, 2
/// sides, 2 downstream TOFs), reached without ever needing a naive nested
/// product over all of them at once.
#[allow(clippy::too_many_arguments)]
pub fn plan_earth_jupiter_saturn_uranus_chain_jointly_optimized(
    departure_year: i16,
    departure_month: u8,
    departure_day: f64,
    leg1_tof_days: f64,
    leg2_tof_min_days: f64,
    leg2_tof_max_days: f64,
    leg2_tof_step_days: f64,
    leg3_tof_min_days: f64,
    leg3_tof_max_days: f64,
    leg3_tof_step_days: f64,
    jupiter_periapsis_min_km: f64,
    jupiter_periapsis_max_km: f64,
    jupiter_periapsis_step_km: f64,
    saturn_periapsis_min_km: f64,
    saturn_periapsis_max_km: f64,
    saturn_periapsis_step_km: f64,
) -> Option<JointTwoFlybyResult> {
    let jd_dep = julian_day(departure_year, departure_month, departure_day);
    let r_earth = heliocentric_position_km(&Planet::Earth, jd_dep);
    let v_earth = heliocentric_velocity_kms(&Planet::Earth, jd_dep);

    let jd_jupiter = jd_dep + leg1_tof_days;
    let r_jupiter = heliocentric_position_km(&Planet::Jupiter, jd_jupiter);
    let v_jupiter = heliocentric_velocity_kms(&Planet::Jupiter, jd_jupiter);

    let mu_sun = mu_km3_s2(&get_body("Sun").expect("Sun is always in the catalog"));
    let jupiter_body = get_body("Jupiter").expect("Jupiter is always in the catalog");
    let mu_jupiter = mu_km3_s2(&jupiter_body);
    let jupiter_periapsis_min_km = jupiter_periapsis_min_km.max(jupiter_body.radius_m / 1000.0);
    let saturn_body = get_body("Saturn").expect("Saturn is always in the catalog");
    let mu_saturn = mu_km3_s2(&saturn_body);
    let saturn_periapsis_min_km = saturn_periapsis_min_km.max(saturn_body.radius_m / 1000.0);

    let leg1_tof_s = leg1_tof_days * 86400.0;
    let sol1 = solve_lambert(&r_earth, &r_jupiter, leg1_tof_s, mu_sun, false).ok()?;
    let leg1_departure_delta_v = (sol1.v1 - v_earth).norm();
    let v_incoming_jupiter = sol1.v2;

    let mut best: Option<(f64, ChainedFlybyCandidate, f64, ChainedFlybyCandidate, f64)> = None;
    let mut evaluated = 0usize;
    let mut leg2_tof_days = leg2_tof_min_days;

    while leg2_tof_days <= leg2_tof_max_days {
        let jd_saturn = jd_jupiter + leg2_tof_days;
        let r_saturn = heliocentric_position_km(&Planet::Saturn, jd_saturn);
        let leg2_tof_s = leg2_tof_days * 86400.0;

        if let Ok(sol2) = solve_lambert(&r_jupiter, &r_saturn, leg2_tof_s, mu_sun, false) {
            // Jupiter flyby: independent minimum for THIS leg-2 TOF.
            let required_v_jupiter = sol2.v1;
            let mut best_jupiter: Option<ChainedFlybyCandidate> = None;
            let mut periapsis_km = jupiter_periapsis_min_km;
            while periapsis_km <= jupiter_periapsis_max_km {
                for leading_side in [false, true] {
                    evaluated += 1;
                    let v_out = apply_gravity_assist(
                        v_incoming_jupiter,
                        v_jupiter,
                        periapsis_km,
                        mu_jupiter,
                        leading_side,
                    );
                    let gap = (v_out - required_v_jupiter).norm();
                    let candidate = ChainedFlybyCandidate {
                        periapsis_radius_km: periapsis_km,
                        leading_side,
                        achieved_outgoing_speed_kms: v_out.norm(),
                        required_departure_speed_kms: required_v_jupiter.norm(),
                        connection_gap_kms: gap,
                    };
                    if best_jupiter
                        .map(|b| candidate.connection_gap_kms < b.connection_gap_kms)
                        .unwrap_or(true)
                    {
                        best_jupiter = Some(candidate);
                    }
                }
                periapsis_km += jupiter_periapsis_step_km;
            }

            // Saturn flyby: independent minimum for THIS leg-2 TOF's
            // resulting arrival velocity at Saturn (sol2.v2), searched
            // over leg-3 TOF x periapsis x side.
            if let Some(best_jupiter) = best_jupiter {
                let v_incoming_saturn = sol2.v2;
                let v_saturn = heliocentric_velocity_kms(&Planet::Saturn, jd_saturn);
                let mut best_saturn: Option<(f64, ChainedFlybyCandidate)> = None;
                let mut leg3_tof_days = leg3_tof_min_days;

                while leg3_tof_days <= leg3_tof_max_days {
                    let jd_uranus = jd_saturn + leg3_tof_days;
                    let r_uranus = heliocentric_position_km(&Planet::Uranus, jd_uranus);
                    let leg3_tof_s = leg3_tof_days * 86400.0;

                    if let Ok(sol3) = solve_lambert(&r_saturn, &r_uranus, leg3_tof_s, mu_sun, false)
                    {
                        let required_v = sol3.v1;
                        let required_speed = required_v.norm();
                        let mut periapsis_km = saturn_periapsis_min_km;
                        while periapsis_km <= saturn_periapsis_max_km {
                            for leading_side in [false, true] {
                                evaluated += 1;
                                let v_out = apply_gravity_assist(
                                    v_incoming_saturn,
                                    v_saturn,
                                    periapsis_km,
                                    mu_saturn,
                                    leading_side,
                                );
                                let gap = (v_out - required_v).norm();
                                let candidate = ChainedFlybyCandidate {
                                    periapsis_radius_km: periapsis_km,
                                    leading_side,
                                    achieved_outgoing_speed_kms: v_out.norm(),
                                    required_departure_speed_kms: required_speed,
                                    connection_gap_kms: gap,
                                };
                                if best_saturn
                                    .as_ref()
                                    .map(|(_, b)| {
                                        candidate.connection_gap_kms < b.connection_gap_kms
                                    })
                                    .unwrap_or(true)
                                {
                                    best_saturn = Some((leg3_tof_days, candidate));
                                }
                            }
                            periapsis_km += saturn_periapsis_step_km;
                        }
                    }
                    leg3_tof_days += leg3_tof_step_days;
                }

                if let Some((leg3_tof_days, best_saturn)) = best_saturn {
                    let total_gap =
                        best_jupiter.connection_gap_kms + best_saturn.connection_gap_kms;
                    if best
                        .as_ref()
                        .map(|(_, _, _, _, b_total)| total_gap < *b_total)
                        .unwrap_or(true)
                    {
                        best = Some((
                            leg2_tof_days,
                            best_jupiter,
                            leg3_tof_days,
                            best_saturn,
                            total_gap,
                        ));
                    }
                }
            }
        }
        leg2_tof_days += leg2_tof_step_days;
    }

    best.map(
        |(leg2_tof_days, jupiter_flyby, leg3_tof_days, saturn_flyby, total_connection_gap_kms)| {
            let jd_saturn = jd_jupiter + leg2_tof_days;
            JointTwoFlybyResult {
                departure_jd: jd_dep,
                jupiter_arrival_jd: jd_jupiter,
                saturn_arrival_jd: jd_saturn,
                uranus_arrival_jd: jd_saturn + leg3_tof_days,
                leg1_departure_delta_v_kms: leg1_departure_delta_v,
                leg2_tof_days,
                leg3_tof_days,
                jupiter_flyby,
                saturn_flyby,
                total_connection_gap_kms,
                candidates_evaluated: evaluated,
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbital_mechanics::keplerian::hohmann_transfer;
    use std::f64::consts::PI;

    /// Shared setup: LEO->GEO coplanar transfer, departure/arrival points
    /// placed near (but not exactly at) 180° apart. Exactly antiparallel
    /// position vectors are a known Lambert-solver singularity (the
    /// transfer plane is undefined) -- this crate's own
    /// `test_lambert_hohmann_agreement` uses the same near-180° approach
    /// for the same reason.
    fn leo_to_geo_near_hohmann() -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3], f64) {
        let r1_mag = 6778.0; // LEO, ~400km altitude
        let r2_mag = 42164.0; // GEO
        let angle = 179.0_f64.to_radians(); // near-antiparallel, avoids the singularity
        let r1 = [r1_mag, 0.0, 0.0];
        let r2 = [r2_mag * angle.cos(), r2_mag * angle.sin(), 0.0];

        let v_circ1 = (MU / r1_mag).sqrt();
        let v_circ2 = (MU / r2_mag).sqrt();
        // Circular velocity is perpendicular to position, in-plane.
        let v1_actual = [0.0, v_circ1, 0.0];
        let v2_actual = [-v_circ2 * angle.sin(), v_circ2 * angle.cos(), 0.0];

        let a_transfer = (r1_mag + r2_mag) / 2.0;
        let hohmann_tof = PI * (a_transfer.powi(3) / MU).sqrt();

        (r1, v1_actual, r2, v2_actual, hohmann_tof)
    }

    #[test]
    fn test_search_finds_a_solution() {
        let (r1, v1, r2, v2, hohmann_tof) = leo_to_geo_near_hohmann();
        let result = plan_min_delta_v_transfer(
            r1,
            v1,
            r2,
            v2,
            hohmann_tof * 0.5,
            hohmann_tof * 1.5,
            hohmann_tof * 0.02,
            false,
        );
        let result = result.expect("expected at least one solvable TOF in range");
        assert!(result.candidates_solved > 0);
        assert!(result.best.delta_v_kms > 0.0);
    }

    #[test]
    fn test_best_tof_is_near_hohmann_tof() {
        // The minimum-delta-v TOF for a coplanar transfer should land near
        // the analytic Hohmann half-period -- a real physics cross-check,
        // not just "the search ran without crashing".
        let (r1, v1, r2, v2, hohmann_tof) = leo_to_geo_near_hohmann();
        let result = plan_min_delta_v_transfer(
            r1,
            v1,
            r2,
            v2,
            hohmann_tof * 0.5,
            hohmann_tof * 1.5,
            hohmann_tof * 0.01,
            false,
        )
        .expect("expected a solution");
        let ratio = result.best.tof_s / hohmann_tof;
        assert!(
            (0.8..1.2).contains(&ratio),
            "expected best TOF near the Hohmann half-period, got ratio {ratio} \
             (best_tof={}, hohmann_tof={hohmann_tof})",
            result.best.tof_s
        );
    }

    #[test]
    fn test_best_delta_v_is_close_to_analytic_hohmann() {
        // Cross-validate the grid-searched minimum against the closed-form
        // Hohmann transfer delta-v. Not exact -- our departure/arrival
        // angle is 179 deg, not exactly 180 deg, per the singularity note
        // above -- but should be within a modest tolerance, tighter than
        // this crate's own test_lambert_hohmann_agreement's very loose
        // 0.3x-3x band (we're searching for the true minimum over TOF,
        // which that test doesn't do -- it only checks one fixed TOF).
        let (r1, v1, r2, v2, hohmann_tof) = leo_to_geo_near_hohmann();
        let result = plan_min_delta_v_transfer(
            r1,
            v1,
            r2,
            v2,
            hohmann_tof * 0.5,
            hohmann_tof * 1.5,
            hohmann_tof * 0.01,
            false,
        )
        .expect("expected a solution");

        let (dv1, dv2) = hohmann_transfer(6778.0, 42164.0);
        let analytic_total = dv1 + dv2;

        let ratio = result.best.delta_v_kms / analytic_total;
        assert!(
            (0.9..1.3).contains(&ratio),
            "expected searched delta-v within ~30% of analytic Hohmann, got \
             ratio {ratio} (searched={}, analytic={analytic_total})",
            result.best.delta_v_kms
        );
    }

    #[test]
    fn test_very_short_tof_has_prohibitive_delta_v() {
        // Lambert's problem is a pure geometric boundary-value solve -- it
        // does NOT refuse a too-short time-of-flight the way one might
        // expect (verified empirically: it happily returns a valid, if
        // absurd, conic connecting LEO and GEO in 1-10 seconds). The real
        // "infeasible" signal a planner cares about is prohibitive delta-v,
        // not solver failure -- this is exactly why the search matters:
        // Lambert alone can't tell you a TOF is a bad idea, only its cost
        // can.
        let (r1, v1, r2, v2, hohmann_tof) = leo_to_geo_near_hohmann();
        let result = plan_min_delta_v_transfer(r1, v1, r2, v2, 1.0, 10.0, 1.0, false)
            .expect("Lambert solves even absurdly short TOFs -- see comment above");
        let (dv1, dv2) = hohmann_transfer(6778.0, 42164.0);
        let analytic_hohmann_total = dv1 + dv2;
        assert!(
            result.best.delta_v_kms > 100.0 * analytic_hohmann_total,
            "expected a ~10s LEO->GEO transfer to cost orders of magnitude \
             more delta-v than the {hohmann_tof:.0}s Hohmann transfer, got \
             {} km/s vs analytic {analytic_hohmann_total} km/s",
            result.best.delta_v_kms
        );
    }

    // Gravity-assist tests below use the J2000.0 epoch (2000-01-01) purely
    // as a clean, deterministic real calendar date -- NOT a claim that this
    // models any specific historical mission's actual launch window.
    //
    // Real finding from building these tests: the idealized coplanar-Hohmann
    // half-period estimate (~997 days for a=3.1 AU, the same formula that
    // correctly predicted the LEO->GEO tests' best TOF above) badly
    // mispredicts the real minimum-energy TOF for an ARBITRARY real
    // departure date. An initial 800-1200 day window (chosen from that
    // Hohmann estimate) found only a ~40 km/s "best" departure delta-v --
    // implausibly high. A wide empirical scan (200-6000 days) found the
    // real minimum near TOF=3000 days at ~8.7-9.2 km/s, matching known
    // real Earth->Jupiter mission-class values (Galileo/Juno-era C3 figures
    // correspond to v_infinity ~9 km/s). The idealized formula assumes
    // Earth and Jupiter are already at the optimal 180 deg relative phase
    // at departure; for an arbitrary real date they generally aren't, so
    // the real low-energy transfer takes a longer, less direct path to
    // reach the correct arrival phase -- this is genuine synodic-alignment
    // behavior real mission designers search porkchop plots for, not a
    // code bug. (Also ruled out a simpler explanation first: adding a
    // short-way/long-way Lambert retry per TOF, the standard practice for
    // arbitrary real geometry, left the ~40 km/s result unchanged --
    // short-way was already cheaper here, so the window itself was wrong,
    // not the transfer sense.)

    #[test]
    fn test_earth_jupiter_search_finds_solutions() {
        let result = plan_earth_jupiter_gravity_assist(
            2000, 1, 1.0, // J2000.0 epoch, an arbitrary real date
            2800.0, 3200.0, 20.0, 200_000.0, false,
        );
        let result = result.expect("expected at least one solvable TOF in range");
        assert!(result.candidates_solved > 0);
        assert!(result.candidates_solved <= result.candidates_evaluated);
        // A real interplanetary departure burn is substantial but not
        // absurd -- sanity-bound it against known real mission delta-v
        // budgets (Galileo/Juno-class Earth departures are ~9 km/s
        // hyperbolic excess, not tens of km/s).
        assert!(
            result.best.departure_delta_v_kms > 0.0 && result.best.departure_delta_v_kms < 15.0,
            "expected a physically plausible departure delta-v, got {}",
            result.best.departure_delta_v_kms
        );
    }

    #[test]
    fn test_trailing_side_flyby_boosts_heliocentric_speed() {
        // The real-physics claim: a trailing-side Jupiter encounter (the
        // spacecraft passes behind Jupiter in its orbital direction)
        // genuinely increases heliocentric speed, using gravity that
        // already exists there -- zero extra propellant for the speed
        // change itself. This exercises the full real pipeline: VSOP87
        // ephemeris -> Lambert (Sun's mu) -> patched-conic flyby
        // (Jupiter's mu), not just the isolated `apply_gravity_assist`
        // unit test in `orbital-mechanics`.
        let result = plan_earth_jupiter_gravity_assist(
            2000, 1, 1.0, 2800.0, 3200.0, 20.0, 200_000.0, false, // trailing side
        )
        .expect("expected a solution");

        assert!(
            result.best.speed_gained_kms > 0.0,
            "expected trailing-side flyby to increase heliocentric speed, \
             got incoming={} outgoing={} gained={}",
            result.best.incoming_heliocentric_speed_kms,
            result.best.outgoing_heliocentric_speed_kms,
            result.best.speed_gained_kms
        );
    }

    #[test]
    fn test_trailing_side_gains_more_speed_than_leading_side() {
        // Cross-check consistent with `gravity_assist::apply_gravity_assist`'s
        // own unit tests, but run end-to-end through real ephemeris-derived
        // trajectories rather than synthetic vectors -- confirms the sign
        // convention holds all the way through this module's plumbing, not
        // just in the isolated physics function.
        //
        // NOT named "opposite effect": at this real geometry and periapsis,
        // BOTH senses net a heliocentric speed GAIN (verified via the demo
        // example, real numbers ~+10.8 km/s trailing vs ~+10.3 km/s
        // leading) -- the isolated library-level test's "opposite" framing
        // (a synthetic setup specifically chosen to show a sign flip) does
        // not generalize to every real geometry/periapsis combination. Only
        // the ordering (trailing > leading) is asserted here, not a sign.
        let trailing =
            plan_earth_jupiter_gravity_assist(2000, 1, 1.0, 2800.0, 3200.0, 20.0, 200_000.0, false)
                .expect("expected a solution");
        let leading =
            plan_earth_jupiter_gravity_assist(2000, 1, 1.0, 2800.0, 3200.0, 20.0, 200_000.0, true)
                .expect("expected a solution");

        assert!(
            trailing.best.speed_gained_kms > leading.best.speed_gained_kms,
            "expected trailing-side to gain more heliocentric speed than \
             leading-side, got trailing={} leading={}",
            trailing.best.speed_gained_kms,
            leading.best.speed_gained_kms
        );
    }

    // TOF=3060 days (the empirically-found best from the tests above) is
    // reused as the fixed leg-1 TOF for both the periapsis search and the
    // Jupiter->Saturn chain tests below.

    #[test]
    fn test_periapsis_search_respects_physical_floor() {
        // Ask for a periapsis range that starts BELOW Jupiter's actual
        // radius (~71,492 km) -- the search must clamp up to the real
        // floor, not silently accept an inside-the-planet "periapsis".
        let result = search_periapsis_for_max_speed_gain(
            2000, 1, 1.0, 3060.0, 10_000.0, 500_000.0, 20_000.0, false,
        )
        .expect("expected a solution");

        assert!(
            result.safety_floor_km > 60_000.0 && result.safety_floor_km < 80_000.0,
            "expected Jupiter's real radius (~71,492 km), got {}",
            result.safety_floor_km
        );
        assert!(
            result.best.periapsis_radius_km >= result.safety_floor_km,
            "periapsis {} must not be below the physical floor {}",
            result.best.periapsis_radius_km,
            result.safety_floor_km
        );
    }

    #[test]
    fn test_periapsis_search_prefers_closer_approach() {
        // Real physics claim: for a fixed incoming v-infinity, turn angle
        // (and thus speed gain) strictly increases as periapsis decreases
        // -- see `gravity_assist::turn_angle_rad`. A PURE speed-gain
        // optimizer with no safety/radiation constraint should therefore
        // always land at (or very near) the search's minimum periapsis --
        // this is the real, honest finding the module docs call out (real
        // missions add a safety margin specifically because this pure
        // optimum is too close for comfort), not a search bug.
        let result = search_periapsis_for_max_speed_gain(
            2000,
            1,
            1.0,
            3060.0,
            100_000.0,
            1_000_000.0,
            50_000.0,
            false,
        )
        .expect("expected a solution");

        assert!(
            (result.best.periapsis_radius_km - 100_000.0).abs() < 1.0,
            "expected the unconstrained optimum to land at the search's own \
             minimum periapsis (100,000 km), got {}",
            result.best.periapsis_radius_km
        );

        // Cross-check the monotonicity claim directly: a far periapsis
        // must gain strictly less speed than the search's chosen minimum.
        let far = search_periapsis_for_max_speed_gain(
            2000,
            1,
            1.0,
            3060.0,
            900_000.0,
            1_000_000.0,
            50_000.0,
            false,
        )
        .expect("expected a solution");
        assert!(
            result.best.speed_gained_kms > far.best.speed_gained_kms,
            "expected closer periapsis to gain more speed: close={} far={}",
            result.best.speed_gained_kms,
            far.best.speed_gained_kms
        );
    }

    #[test]
    fn test_earth_jupiter_saturn_chain_finds_a_connection() {
        // The "rediscover a Voyager-class route" two-leg chain. Real
        // Jupiter->Saturn transfer times are on the order of several years
        // (Voyager took roughly 20 months Jupiter-to-Saturn on a fast
        // trajectory that traded fuel-free time for a specific arrival
        // geometry; a slower, more direct Hohmann-like leg can take
        // longer) -- search a broad multi-year window rather than assume
        // a specific value, the same lesson learned from leg 1's TOF
        // window above.
        //
        // Periapsis range 100,000-3,000,000 km, NOT 100,000-1,000,000: an
        // earlier version of this test used the narrower range and its
        // "best" landed exactly at the 1,000,000 km upper boundary --
        // always a red flag for a grid search (means the true optimum may
        // lie outside the searched range, the same lesson learned from
        // the TOF window above). Confirmed via a temporary wider scan:
        // connection_gap_kms vs periapsis is NOT monotonic here -- it has
        // a genuine INTERIOR minimum near periapsis=1,455,000 km
        // (gap=3.53 km/s), then gets worse again at larger periapsis. This
        // makes physical sense: the achieved outgoing velocity traces a
        // circular arc in velocity space as periapsis (thus turn angle)
        // varies, and the distance from a point sweeping along a circle to
        // a fixed external target point generically has one interior
        // minimum, not a monotonic trend. The corrected range now brackets
        // that real interior optimum instead of clipping it at a boundary.
        let result = plan_earth_jupiter_saturn_chain(
            2000,
            1,
            1.0,
            3060.0,
            500.0,
            3000.0,
            100.0,
            100_000.0,
            3_000_000.0,
            20_000.0,
        )
        .expect("expected at least one Jupiter->Saturn Lambert solution");

        assert!(result.candidates_evaluated > 0);
        assert!(result.saturn_arrival_jd > result.jupiter_arrival_jd);
        assert!(result.jupiter_arrival_jd > result.departure_jd);
        // The connection gap is a real, unconstrained delta-v figure (km/s)
        // -- assert it's finite and non-negative, not a specific value,
        // since how well an unpowered flyby can thread a specific Saturn
        // arrival is a genuine open question this function answers, not
        // something to assume in advance.
        assert!(result.best.connection_gap_kms >= 0.0);
        assert!(result.best.connection_gap_kms.is_finite());
        assert!(result.best.periapsis_radius_km >= 71_000.0);
        // The interior optimum should NOT sit exactly on either search
        // boundary -- that would again signal the range doesn't bracket
        // the true minimum. A small margin (not touching 100_000 or
        // 3_000_000 exactly) confirms this is a real interior result.
        assert!(
            result.best.periapsis_radius_km > 150_000.0
                && result.best.periapsis_radius_km < 2_900_000.0,
            "expected an interior optimum, got periapsis {} (too close to a \
             search boundary -- range may need widening)",
            result.best.periapsis_radius_km
        );
    }

    #[test]
    fn test_earth_jupiter_saturn_uranus_chain_finds_two_flybys() {
        // The genuine two-flyby extension: Jupiter AND Saturn each get
        // their own independent periapsis/side/TOF search. Real
        // Saturn->Uranus transfer times are on the order of several years
        // (comparable to or longer than the Jupiter->Saturn leg above) --
        // search a broad window rather than assume a specific value, same
        // lesson as every TOF window in this module.
        let result = plan_earth_jupiter_saturn_uranus_chain(
            2000,
            1,
            1.0,
            3060.0,
            2400.0,
            2800.0,
            50.0,
            3000.0,
            8000.0,
            100.0,
            100_000.0,
            3_000_000.0,
            50_000.0,
            100_000.0,
            3_000_000.0,
            50_000.0,
        )
        .expect("expected a full two-flyby connection");

        assert!(result.candidates_evaluated > 0);
        assert!(result.uranus_arrival_jd > result.saturn_arrival_jd);
        assert!(result.saturn_arrival_jd > result.jupiter_arrival_jd);
        assert!(result.jupiter_arrival_jd > result.departure_jd);

        // Both flybys must respect their own planet's real physical floor.
        assert!(result.jupiter_flyby.periapsis_radius_km >= 69_000.0);
        assert!(result.saturn_flyby.periapsis_radius_km >= 58_000.0);

        // Both connection gaps are real, finite, non-negative figures --
        // not asserted to any specific value, since how well two
        // independent free flybys can thread this specific route is a
        // genuine open question this function answers empirically.
        assert!(result.jupiter_flyby.connection_gap_kms.is_finite());
        assert!(result.jupiter_flyby.connection_gap_kms >= 0.0);
        assert!(result.saturn_flyby.connection_gap_kms.is_finite());
        assert!(result.saturn_flyby.connection_gap_kms >= 0.0);
    }

    #[test]
    fn test_joint_optimizer_never_worse_than_greedy() {
        // The defining property a joint optimizer over a superset of what
        // the greedy composition considers MUST have: its total connection
        // gap can never be worse than the greedy version's, for the exact
        // same search ranges -- greedy is a restriction of the same search
        // space (it only ever considers ONE leg-2 TOF, whichever minimizes
        // the Jupiter gap alone), so the joint optimum is always <= it.
        // Smaller ranges than the full example demo, to keep this test fast.
        let greedy = plan_earth_jupiter_saturn_uranus_chain(
            2000,
            1,
            1.0,
            3060.0,
            2400.0,
            2800.0,
            100.0,
            3000.0,
            8000.0,
            500.0,
            100_000.0,
            2_000_000.0,
            100_000.0,
            100_000.0,
            2_000_000.0,
            100_000.0,
        )
        .expect("expected a greedy connection");
        let greedy_total =
            greedy.jupiter_flyby.connection_gap_kms + greedy.saturn_flyby.connection_gap_kms;

        let joint = plan_earth_jupiter_saturn_uranus_chain_jointly_optimized(
            2000,
            1,
            1.0,
            3060.0,
            2400.0,
            2800.0,
            100.0,
            3000.0,
            8000.0,
            500.0,
            100_000.0,
            2_000_000.0,
            100_000.0,
            100_000.0,
            2_000_000.0,
            100_000.0,
        )
        .expect("expected a joint connection");

        assert!(
            joint.total_connection_gap_kms <= greedy_total + 1e-9,
            "joint optimizer (total={}) must never be worse than greedy \
             (jupiter={} + saturn={} = {}) over the same search ranges",
            joint.total_connection_gap_kms,
            greedy.jupiter_flyby.connection_gap_kms,
            greedy.saturn_flyby.connection_gap_kms,
            greedy_total
        );
        assert!(joint.candidates_evaluated > 0);
        assert!(joint.jupiter_flyby.periapsis_radius_km >= 69_000.0);
        assert!(joint.saturn_flyby.periapsis_radius_km >= 58_000.0);
    }
}
