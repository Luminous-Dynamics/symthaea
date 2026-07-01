// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consumer Economics Optimization
//!
//! Pathway analysis for achieving cost-competitive consumer fusion.
//!
//! Run with: cargo run --example consumer_economics

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::{
    CapitalCosts, CoupledPhysicsEngine, EconomicEngine, EnergyComparison, FuelCosts, OmCosts,
    OperatingConditions,
};

fn main() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              CONSUMER ECONOMICS OPTIMIZATION                         ║");
    println!("║              Pathway to Cost-Competitive Fusion                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let genesis = GenesisSeed::from_phrase("Consumer Economics 2024");
    let physics = CoupledPhysicsEngine::from_genesis(&genesis);

    // ═══════════════════════════════════════════════════════════════════════════
    // BASELINE: Current Consumer Design (5 kW D-D)
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BASELINE: 5 kW D-D Consumer Unit");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let conditions = OperatingConditions::consumer();
    let result = physics.simulate(&conditions);

    let econ = EconomicEngine::consumer(conditions.power_kw);
    let capital = CapitalCosts::consumer_5kw();
    let om = OmCosts::consumer();
    let fuel = FuelCosts::dd_fusion();

    let baseline_lcoe = econ.calculate_lcoe(&capital, &om, &fuel);

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  BASELINE DESIGN                                                   │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Power Output:        {:>8.0} kW                                  │",
        conditions.power_kw
    );
    println!(
        "│  System Mass:         {:>8.0} kg                                  │",
        result.geometry_shielding.total_mass_kg
    );
    println!(
        "│  Shielding:           {:>8.2} m                                   │",
        result.geometry_shielding.shielding.thickness_m
    );
    println!(
        "│  Shell Material:      {}                              │",
        result.shell_material
    );
    println!(
        "│  Feasible:            {}                                       │",
        if result.feasible { "YES" } else { "NO " }
    );
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│  ECONOMICS                                                         │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Capital Cost:        ${:>9.0}                                │",
        capital.total()
    );
    println!(
        "│  Cost per kW:         ${:>9.0}/kW                             │",
        capital.cost_per_kw(conditions.power_kw)
    );
    println!(
        "│  Annual O&M:          ${:>9.0}/year                           │",
        om.annual_cost(conditions.power_kw, econ.capacity_factor)
    );
    println!(
        "│  LCOE:                ${:>9.1}/MWh (${:.3}/kWh)             │",
        baseline_lcoe.lcoe_usd_mwh, baseline_lcoe.lcoe_usd_kwh
    );
    println!(
        "│  Payback Period:      {:>9.1} years                           │",
        baseline_lcoe.payback_years
    );
    println!("└────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPARISON WITH ALTERNATIVES
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  COMPARISON WITH ALTERNATIVES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let grid = EnergyComparison::grid_electricity();
    let solar = EnergyComparison::solar_pv();

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  Source                   LCOE ($/MWh)   CF      Dispatch  CO2    │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  ★ Spark Fusion (5 kW)    {:>8.1}       {:>4.0}%   Yes       0     │",
        baseline_lcoe.lcoe_usd_mwh,
        econ.capacity_factor * 100.0
    );
    println!(
        "│    Grid Electricity       {:>8.1}       {:>4.0}%   Yes      400    │",
        grid.lcoe_usd_mwh,
        grid.capacity_factor * 100.0
    );
    println!(
        "│    Residential Solar      {:>8.1}       {:>4.0}%   No        40    │",
        solar.lcoe_usd_mwh * 3.0,
        solar.capacity_factor * 100.0
    ); // Residential premium
    println!(
        "│    Solar + Battery        {:>8.1}       {:>4.0}%   Yes       50    │",
        solar.lcoe_usd_mwh * 3.0 + 150.0,
        50.0
    ); // With storage
    println!("└────────────────────────────────────────────────────────────────────┘");

    let premium = baseline_lcoe.lcoe_usd_mwh / grid.lcoe_usd_mwh;
    println!(
        "\n  Current premium over grid: {:.1}× (${:.2}/kWh vs ${:.2}/kWh)",
        premium,
        baseline_lcoe.lcoe_usd_kwh,
        grid.lcoe_usd_mwh / 1000.0
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // SENSITIVITY ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SENSITIVITY ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let sensitivities = econ.sensitivity_analysis(&capital, &om, &fuel);

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  Factor                    Impact on LCOE      New LCOE ($/MWh)  │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    for (param, change_pct, new_lcoe) in &sensitivities {
        let direction = if *change_pct > 0.0 { "↑" } else { "↓" };
        println!(
            "│  {:25} {} {:>5.1}%          {:>8.1}            │",
            param,
            direction,
            change_pct.abs(),
            new_lcoe
        );
    }
    println!("└────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // COST REDUCTION SCENARIOS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  COST REDUCTION SCENARIOS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Scenario 1: Near-term (3-5 years) - Learning curve reductions
    let near_term_capital = CapitalCosts {
        fusion_core_usd: capital.fusion_core_usd * 0.7, // 30% reduction
        shielding_usd: capital.shielding_usd * 0.8,     // 20% reduction
        power_conversion_usd: capital.power_conversion_usd * 0.9,
        balance_of_plant_usd: capital.balance_of_plant_usd * 0.85,
        installation_usd: capital.installation_usd * 0.8,
        contingency_fraction: 0.15, // Lower contingency with experience
    };
    let near_term_lcoe = econ.calculate_lcoe(&near_term_capital, &om, &fuel);

    // Scenario 2: Mid-term (5-10 years) - Mass production
    let mid_term_capital = CapitalCosts {
        fusion_core_usd: capital.fusion_core_usd * 0.4, // 60% reduction
        shielding_usd: capital.shielding_usd * 0.5,     // 50% reduction
        power_conversion_usd: capital.power_conversion_usd * 0.6,
        balance_of_plant_usd: capital.balance_of_plant_usd * 0.5,
        installation_usd: capital.installation_usd * 0.5,
        contingency_fraction: 0.10,
    };
    let mid_term_om = OmCosts {
        fixed_usd_kw_year: om.fixed_usd_kw_year * 0.7,
        variable_usd_mwh: om.variable_usd_mwh * 0.7,
        overhaul_cost_usd: om.overhaul_cost_usd * 0.6,
        overhaul_interval_years: 15.0, // Longer between overhauls
    };
    let mid_term_lcoe = econ.calculate_lcoe(&mid_term_capital, &mid_term_om, &fuel);

    // Scenario 3: Long-term (10+ years) - Full optimization
    let long_term_capital = CapitalCosts {
        fusion_core_usd: capital.fusion_core_usd * 0.2, // 80% reduction
        shielding_usd: capital.shielding_usd * 0.3,     // 70% reduction
        power_conversion_usd: capital.power_conversion_usd * 0.4,
        balance_of_plant_usd: capital.balance_of_plant_usd * 0.3,
        installation_usd: capital.installation_usd * 0.3,
        contingency_fraction: 0.05,
    };
    let long_term_om = OmCosts {
        fixed_usd_kw_year: om.fixed_usd_kw_year * 0.5,
        variable_usd_mwh: om.variable_usd_mwh * 0.5,
        overhaul_cost_usd: om.overhaul_cost_usd * 0.4,
        overhaul_interval_years: 20.0,
    };
    let long_term_lcoe = econ.calculate_lcoe(&long_term_capital, &long_term_om, &fuel);

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  Scenario              Capital ($)    LCOE ($/MWh)  vs Grid      │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Baseline (Today)      {:>10.0}     {:>8.1}       {:.1}×         │",
        capital.total(),
        baseline_lcoe.lcoe_usd_mwh,
        baseline_lcoe.lcoe_usd_mwh / grid.lcoe_usd_mwh
    );
    println!(
        "│  Near-term (3-5 yr)    {:>10.0}     {:>8.1}       {:.1}×         │",
        near_term_capital.total(),
        near_term_lcoe.lcoe_usd_mwh,
        near_term_lcoe.lcoe_usd_mwh / grid.lcoe_usd_mwh
    );
    println!(
        "│  Mid-term (5-10 yr)    {:>10.0}     {:>8.1}       {:.1}×         │",
        mid_term_capital.total(),
        mid_term_lcoe.lcoe_usd_mwh,
        mid_term_lcoe.lcoe_usd_mwh / grid.lcoe_usd_mwh
    );
    println!(
        "│  Long-term (10+ yr)    {:>10.0}     {:>8.1}       {:.1}×         │",
        long_term_capital.total(),
        long_term_lcoe.lcoe_usd_mwh,
        long_term_lcoe.lcoe_usd_mwh / grid.lcoe_usd_mwh
    );
    println!(
        "│  Grid parity target    {:>10}     {:>8.1}       1.0×         │",
        "-", grid.lcoe_usd_mwh
    );
    println!("└────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // POWER LEVEL OPTIMIZATION
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  POWER LEVEL OPTIMIZATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│  Power (kW)  Capital ($)     $/kW      LCOE ($/MWh)   Payback    │");
    println!("├────────────────────────────────────────────────────────────────────┤");

    for power in [2.0_f64, 5.0, 10.0, 15.0, 20.0, 25.0] {
        // Scale capital costs (sub-linear scaling)
        let scale: f64 = power / 5.0;
        let scaled_capital = CapitalCosts {
            fusion_core_usd: capital.fusion_core_usd * scale.powf(0.75),
            shielding_usd: capital.shielding_usd * scale.powf(0.8),
            power_conversion_usd: capital.power_conversion_usd * scale.powf(0.7),
            balance_of_plant_usd: capital.balance_of_plant_usd * scale.powf(0.6),
            installation_usd: capital.installation_usd * scale.powf(0.5),
            contingency_fraction: capital.contingency_fraction,
        };

        let scaled_econ = EconomicEngine::consumer(power);
        let scaled_lcoe = scaled_econ.calculate_lcoe(&scaled_capital, &om, &fuel);

        println!(
            "│  {:>6.0}       {:>10.0}     {:>6.0}     {:>8.1}        {:>6.1}     │",
            power,
            scaled_capital.total(),
            scaled_capital.cost_per_kw(power),
            scaled_lcoe.lcoe_usd_mwh,
            scaled_lcoe.payback_years.min(99.0)
        );
    }
    println!("└────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // VALUE PROPOSITION ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  VALUE PROPOSITION: Why Pay Premium?");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  QUANTIFIABLE BENEFITS                                             │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  ✓ Energy independence:  No grid reliance                          │");
    println!("│  ✓ Power outage immunity: ~$200/event value (1-2 events/yr)        │");
    println!("│  ✓ No utility price hikes: 3%/yr inflation protection              │");
    println!("│  ✓ Zero emissions:        Carbon credit value ~$50/tCO2            │");
    println!("│  ✓ Property value:        +5-10% home value                        │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    // Calculate effective LCOE with benefits
    let annual_energy_kwh = conditions.power_kw * econ.capacity_factor * 8760.0;
    let annual_co2_avoided_tons = annual_energy_kwh * 0.0004; // 400 g/kWh grid average
    let carbon_credit_value = annual_co2_avoided_tons * 50.0; // $50/ton
    let outage_value = 300.0; // $300/year
    let inflation_hedge_value = 0.03 * grid.lcoe_usd_mwh / 1000.0 * annual_energy_kwh; // 3% inflation

    let total_benefit_value = carbon_credit_value + outage_value + inflation_hedge_value;
    let effective_lcoe = (baseline_lcoe.lcoe_usd_mwh * annual_energy_kwh / 1000.0
        - total_benefit_value)
        / (annual_energy_kwh / 1000.0);

    println!("\n┌────────────────────────────────────────────────────────────────────┐");
    println!("│  ADJUSTED ECONOMICS (Including Benefits)                          │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Carbon credits:        ${:>6.0}/year                              │",
        carbon_credit_value
    );
    println!(
        "│  Outage protection:     ${:>6.0}/year                              │",
        outage_value
    );
    println!(
        "│  Inflation hedge:       ${:>6.0}/year                              │",
        inflation_hedge_value
    );
    println!("│  ─────────────────────────────────────────────                     │");
    println!(
        "│  Total benefit value:   ${:>6.0}/year                              │",
        total_benefit_value
    );
    println!(
        "│  Effective LCOE:        ${:>6.1}/MWh ({:.1}× vs grid)              │",
        effective_lcoe,
        effective_lcoe / grid.lcoe_usd_mwh
    );
    println!("└────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════
    // ROADMAP TO COMPETITIVENESS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    ROADMAP TO GRID PARITY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  CRITICAL PATH TO $120/MWh (Grid Parity)                           │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  1. [CAPITAL] Reduce fusion core cost by 70%                       │");
    println!("│     → Manufacturing scale: 1000+ units/year                        │");
    println!("│     → Material cost reduction through standardization              │");
    println!("│                                                                    │");
    println!("│  2. [SHIELDING] Optimize shielding mass                            │");
    println!("│     → Advanced materials: borated polyethylene composites          │");
    println!("│     → Geometry optimization: conformal shielding                   │");
    println!("│                                                                    │");
    println!("│  3. [O&M] Extend maintenance intervals                             │");
    println!("│     → Predictive maintenance with ML                               │");
    println!("│     → Remote diagnostics and OTA updates                           │");
    println!("│                                                                    │");
    println!("│  4. [SCALE] Increase power per unit                                │");
    println!("│     → 10-15 kW optimal for residential + EV charging               │");
    println!("│     → 25-50 kW for multi-family / small commercial                 │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    let target_lcoe = grid.lcoe_usd_mwh;
    let reduction_needed = (1.0 - target_lcoe / baseline_lcoe.lcoe_usd_mwh) * 100.0;

    println!("\n┌────────────────────────────────────────────────────────────────────┐");
    println!("│  SUMMARY                                                           │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Current LCOE:          ${:.1}/MWh                              │",
        baseline_lcoe.lcoe_usd_mwh
    );
    println!(
        "│  Grid parity target:    ${:.1}/MWh                               │",
        target_lcoe
    );
    println!(
        "│  Required reduction:    {:.0}%                                     │",
        reduction_needed
    );
    println!(
        "│  With benefits credit:  ${:.1}/MWh ({:.0}% reduction needed)       │",
        effective_lcoe,
        (1.0 - target_lcoe / effective_lcoe).max(0.0) * 100.0
    );
    println!("└────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    CONSUMER ECONOMICS COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
}
