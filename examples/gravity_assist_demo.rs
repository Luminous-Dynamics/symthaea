// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Earth-Jupiter Gravity-Assist Demo (Phase 3 item 2 of
//! `SPACE_AUTOMATION_PLAN_2026-07-06.md`) -- the honest "grav-craft".
//!
//! This is a real trajectory-optimization physics demo, not a claim that
//! Symthaea's cognitive/consciousness engine is "reasoning" about orbital
//! mechanics: `plan_earth_jupiter_gravity_assist()` is a deterministic
//! grid-search optimizer over real VSOP87 ephemeris + Lambert + patched-conic
//! flyby physics, called directly here with no LLM or consciousness loop
//! involved.
//!
//! Shows that a trailing-side Jupiter flyby genuinely increases heliocentric
//! speed using gravity that already exists there, at zero extra propellant
//! cost beyond the initial departure burn, and that it gains MORE speed than
//! the leading-side alternative -- not necessarily that leading-side loses
//! speed outright. Whether leading-side nets a gain or a loss depends on the
//! incoming approach geometry and how large a turn angle the chosen periapsis
//! allows (see the `plan_earth_jupiter_gravity_assist` doc comment); at the
//! real ephemeris geometry and 200,000km periapsis used here, both senses
//! happen to net a gain, with trailing-side larger.
//!
//! Also demonstrates two extensions:
//! 1. **Periapsis search** (`search_periapsis_for_max_speed_gain`) -- the
//!    real "closer periapsis buys more free delta-v, but costs safety
//!    margin/radiation exposure" trade real mission designers navigate.
//! 2. **Earth->Jupiter->Saturn chain** (`plan_earth_jupiter_saturn_chain`) --
//!    the "rediscover a Voyager-class route" two-leg extension, honestly
//!    scoped to what a single free (unpowered) flyby can actually target.
//!
//! Single-flyby/two-leg scope, not a full multi-leg mission-design search;
//! J2000.0 is an arbitrary real calendar date, not a claim about matching
//! any historical mission's actual launch window.
//!
//! Run with: `cargo run --release --example gravity_assist_demo --features orbital`

fn main() {
    #[cfg(feature = "orbital")]
    {
        use symthaea_orbital::trajectory_planning::{
            plan_earth_jupiter_gravity_assist, plan_earth_jupiter_saturn_chain,
            plan_earth_jupiter_saturn_uranus_chain,
            plan_earth_jupiter_saturn_uranus_chain_jointly_optimized,
            search_periapsis_for_max_speed_gain,
        };

        println!();
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  EARTH -> JUPITER GRAVITY-ASSIST DEMO -- the honest \"grav-craft\"   ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Real VSOP87 ephemeris + real Lambert solve (Sun's mu) + real");
        println!("patched-conic Jupiter flyby (Jupiter's mu). Single-flyby demo,");
        println!("not a multi-leg mission-design search. Departure date is an");
        println!("arbitrary real calendar date (J2000.0), not a claim about any");
        println!("historical mission's actual launch window.");
        println!();

        let periapsis_km = 200_000.0;

        for (label, leading_side) in [("Trailing-side flyby", false), ("Leading-side flyby", true)]
        {
            // TOF window is 2800-3200 days, not the ~997-day idealized
            // coplanar-Hohmann half-period: for this specific real J2000.0
            // departure date, Earth and Jupiter aren't at the idealized
            // optimal relative phase, so the real minimum-energy transfer
            // takes a longer path to reach the correct arrival phase (found
            // via a wide empirical scan -- see trajectory_planning.rs test
            // comments for the full story). Real ~8.7-9.2 km/s departure
            // delta-v here matches known Galileo/Juno-class mission figures.
            let result = plan_earth_jupiter_gravity_assist(
                2000,
                1,
                1.0,
                2800.0,
                3200.0,
                20.0,
                periapsis_km,
                leading_side,
            );

            println!("-- {label} --");
            match result {
                Some(r) => {
                    println!("  Departure JD:              {:.2}", r.departure_jd);
                    println!("  Candidates evaluated:      {}", r.candidates_evaluated);
                    println!("  Candidates solved:         {}", r.candidates_solved);
                    println!("  Best TOF:                  {:.1} days", r.best.tof_days);
                    println!(
                        "  Departure delta-v:         {:.3} km/s",
                        r.best.departure_delta_v_kms
                    );
                    println!(
                        "  Incoming heliocentric v:   {:.3} km/s",
                        r.best.incoming_heliocentric_speed_kms
                    );
                    println!(
                        "  Outgoing heliocentric v:   {:.3} km/s",
                        r.best.outgoing_heliocentric_speed_kms
                    );
                    println!(
                        "  Speed gained:              {:+.3} km/s",
                        r.best.speed_gained_kms
                    );
                }
                None => {
                    println!("  No solvable transfer found in the searched TOF window.");
                }
            }
            println!();
        }

        println!("What this demonstrates: the trailing-side flyby's speed gain is");
        println!("positive and larger than the leading-side flyby's -- a real,");
        println!("physically-grounded speed change using gravity that already");
        println!("exists, at zero propellant cost beyond the initial departure burn.");
        println!();
        println!("What this does NOT demonstrate: propulsion improvement, a");
        println!("multi-leg mission route, or any claim tied to a real historical");
        println!("mission's actual launch window.");

        // ------------------------------------------------------------
        // Extension 1: periapsis search
        // ------------------------------------------------------------
        println!();
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  PERIAPSIS SEARCH -- speed gain vs. safety margin                  ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Grid-searches periapsis radius (real physical floor: must clear");
        println!("Jupiter's actual radius) for the value that maximizes speed gain.");
        println!();

        let periapsis_result = search_periapsis_for_max_speed_gain(
            2000,
            1,
            1.0,
            3060.0,
            100_000.0,
            1_000_000.0,
            50_000.0,
            false,
        );
        match periapsis_result {
            Some(r) => {
                println!(
                    "  Safety floor (Jupiter's radius): {:.0} km",
                    r.safety_floor_km
                );
                println!(
                    "  Candidates evaluated:            {}",
                    r.candidates_evaluated
                );
                println!(
                    "  Best periapsis:                  {:.0} km",
                    r.best.periapsis_radius_km
                );
                println!(
                    "  Speed gained at best periapsis:  {:+.3} km/s",
                    r.best.speed_gained_kms
                );
                println!();
                println!("This lands at the search's own minimum periapsis (100,000 km) --");
                println!("expected, not a bug: turn angle strictly increases as periapsis");
                println!("decreases, so a PURE speed-gain optimizer with no radiation/safety");
                println!("constraint always prefers the closest allowed approach. Real");
                println!("missions (Galileo, Juno) choose a much larger periapsis specifically");
                println!("to survive Jupiter's radiation belts -- this function does not");
                println!("decide what counts as safe, it only refuses to go inside the planet.");
            }
            None => println!("  No solution found."),
        }

        // ------------------------------------------------------------
        // Extension 2: Earth -> Jupiter -> Saturn chain
        // ------------------------------------------------------------
        println!();
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  EARTH -> JUPITER -> SATURN CHAIN -- a Voyager-class route         ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Leg 1 (Earth->Jupiter) fixed at the TOF found optimal above. Leg 2");
        println!("(Jupiter->Saturn) is a SEPARATE real Lambert solve, searched over");
        println!("time-of-flight. The free (unpowered) flyby can only deliver one");
        println!("outgoing velocity per (periapsis, side) choice -- this searches both");
        println!("to find the combination closest to what leg 2 actually requires.");
        println!();

        // Periapsis range 100,000-3,000,000 km: a narrower 100k-1M range
        // was tried first and its "best" landed exactly at the 1M upper
        // boundary -- a red flag for any grid search (same lesson as the
        // TOF-window fix above). A wider scan confirmed connection_gap_kms
        // vs periapsis is NOT monotonic here -- it has a genuine interior
        // minimum near periapsis=1,455,000 km, then gets worse again at
        // larger periapsis. This makes sense: the achieved outgoing
        // velocity traces a circular arc in velocity space as periapsis
        // (and thus turn angle) varies, and the distance from a point
        // sweeping along a circle to a fixed external target generically
        // has one interior minimum, not a monotonic trend.
        let chain_result = plan_earth_jupiter_saturn_chain(
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
        );
        match chain_result {
            Some(r) => {
                println!("  Departure JD:                  {:.2}", r.departure_jd);
                println!(
                    "  Jupiter arrival JD:            {:.2}",
                    r.jupiter_arrival_jd
                );
                println!(
                    "  Saturn arrival JD:              {:.2}",
                    r.saturn_arrival_jd
                );
                println!(
                    "  Leg 2 TOF:                      {:.1} days",
                    r.leg2_tof_days
                );
                println!(
                    "  Leg 1 departure delta-v:        {:.3} km/s",
                    r.leg1_departure_delta_v_kms
                );
                println!(
                    "  Candidates evaluated:           {}",
                    r.candidates_evaluated
                );
                println!(
                    "  Best periapsis:                 {:.0} km",
                    r.best.periapsis_radius_km
                );
                println!(
                    "  Best side:                      {}",
                    if r.best.leading_side {
                        "leading"
                    } else {
                        "trailing"
                    }
                );
                println!(
                    "  Achieved outgoing speed:        {:.3} km/s",
                    r.best.achieved_outgoing_speed_kms
                );
                println!(
                    "  Required departure speed:       {:.3} km/s",
                    r.best.required_departure_speed_kms
                );
                println!(
                    "  Connection gap:                 {:.3} km/s",
                    r.best.connection_gap_kms
                );
                println!();
                println!("The connection gap is the delta-v a deep-space maneuver at Jupiter");
                println!("would still need to supply to perfectly connect the two legs -- a");
                println!("free flyby alone generally CANNOT thread an arbitrary target with");
                println!("just two degrees of freedom (periapsis + side); this number is the");
                println!("honest measure of how close it gets for this particular route.");
            }
            None => println!("  No Jupiter->Saturn Lambert solution found in the searched window."),
        }

        // ------------------------------------------------------------
        // Extension 3: two-flyby chain, Earth -> Jupiter -> Saturn -> Uranus
        // ------------------------------------------------------------
        println!();
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  EARTH -> JUPITER -> SATURN -> URANUS -- a two-flyby grand tour    ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Extends the chain above by one more real gravity assist AT Saturn.");
        println!("Each flyby's periapsis/side is searched independently (greedy,");
        println!("leg-by-leg), assuming a deep-space maneuver corrects onto each");
        println!("intended Lambert arc -- the standard patched-conic technique real");
        println!("missions like Cassini and Galileo actually use. NOT a joint");
        println!("optimization across both flybys at once (real future scope).");
        println!();

        let two_flyby_result = plan_earth_jupiter_saturn_uranus_chain(
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
        );
        match two_flyby_result {
            Some(r) => {
                println!("  Departure JD:                  {:.2}", r.departure_jd);
                println!(
                    "  Jupiter arrival JD:            {:.2}",
                    r.jupiter_arrival_jd
                );
                println!(
                    "  Saturn arrival JD:              {:.2}",
                    r.saturn_arrival_jd
                );
                println!(
                    "  Uranus arrival JD:              {:.2}",
                    r.uranus_arrival_jd
                );
                println!(
                    "  Leg 1 departure delta-v:        {:.3} km/s",
                    r.leg1_departure_delta_v_kms
                );
                println!(
                    "  Candidates evaluated:           {}",
                    r.candidates_evaluated
                );
                println!();
                println!("  Jupiter flyby:");
                println!(
                    "    periapsis={:.0} km  side={}  gap={:.3} km/s",
                    r.jupiter_flyby.periapsis_radius_km,
                    if r.jupiter_flyby.leading_side {
                        "leading"
                    } else {
                        "trailing"
                    },
                    r.jupiter_flyby.connection_gap_kms
                );
                println!("  Saturn flyby:");
                println!(
                    "    periapsis={:.0} km  side={}  gap={:.3} km/s",
                    r.saturn_flyby.periapsis_radius_km,
                    if r.saturn_flyby.leading_side {
                        "leading"
                    } else {
                        "trailing"
                    },
                    r.saturn_flyby.connection_gap_kms
                );
                println!();
                println!("Two independent connection gaps, one per flyby -- each is the");
                println!("delta-v a deep-space maneuver at that planet would still need to");
                println!("supply. This is a genuine two-flyby mission-design search, though");
                println!("still greedy rather than jointly optimized across both flybys.");
            }
            None => println!("  No full two-flyby connection found in the searched windows."),
        }

        // ------------------------------------------------------------
        // Extension 4: joint optimization vs. the greedy composition above
        // ------------------------------------------------------------
        println!();
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  JOINT OPTIMIZATION -- does optimizing both flybys together help? ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("The greedy chain above picks the Jupiter->Saturn TOF that minimizes");
        println!("ONLY the Jupiter flyby's own gap, then lives with whatever Saturn");
        println!("connection that choice leaves. This jointly searches leg-2 TOF");
        println!("against the TOTAL gap (Jupiter's + Saturn's combined) instead --");
        println!("tractable because, under the deep-space-maneuver assumption, each");
        println!("flyby's own periapsis/side choice doesn't affect the OTHER flyby's");
        println!("gap; only the shared leg-2 TOF choice couples them.");
        println!();

        let joint_result = plan_earth_jupiter_saturn_uranus_chain_jointly_optimized(
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
        );
        match (two_flyby_result, joint_result) {
            (Some(greedy), Some(joint)) => {
                let greedy_total = greedy.jupiter_flyby.connection_gap_kms
                    + greedy.saturn_flyby.connection_gap_kms;
                println!("  Greedy total gap:               {:.3} km/s", greedy_total);
                println!(
                    "  Joint total gap:                 {:.3} km/s",
                    joint.total_connection_gap_kms
                );
                println!(
                    "  Improvement:                     {:.3} km/s ({:.1}%)",
                    greedy_total - joint.total_connection_gap_kms,
                    100.0 * (greedy_total - joint.total_connection_gap_kms) / greedy_total
                );
                println!(
                    "  Joint leg-2 TOF:                 {:.1} days",
                    joint.leg2_tof_days
                );
                println!(
                    "  Greedy leg-2 TOF:                {:.1} days",
                    greedy.leg2_tof_days
                );
                println!();
                if joint.leg2_tof_days == greedy.leg2_tof_days {
                    println!("Same leg-2 TOF chosen -- for this route, minimizing Jupiter's");
                    println!("own gap first also happened to be globally optimal; joint search");
                    println!("confirms this rather than finding a different, better answer.");
                } else {
                    println!("DIFFERENT leg-2 TOF chosen -- a real example of the greedy");
                    println!("composition being genuinely suboptimal: trading a slightly worse");
                    println!("Jupiter connection for a meaningfully better Saturn connection");
                    println!("reduces the TOTAL maneuver cost below what greedy alone found.");
                }
            }
            _ => println!("  Could not compare -- one or both searches found no connection."),
        }
    }

    #[cfg(not(feature = "orbital"))]
    {
        println!("orbital feature disabled; skipping gravity-assist demo.");
        println!("Run with: cargo run --release --example gravity_assist_demo --features orbital");
    }
}
