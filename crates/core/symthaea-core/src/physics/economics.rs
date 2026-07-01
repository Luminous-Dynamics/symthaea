// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Economic Viability Model: LCOE and Cost Analysis
//!
//! Calculates Levelized Cost of Energy (LCOE) and compares Spark Engine
//! economics against competing energy sources.
//!
//! ## LCOE Formula
//!
//! ```text
//! LCOE = (Capital × CRF + O&M + Fuel) / (Capacity × CF × 8760)
//!
//! where:
//!   CRF = Capital Recovery Factor = r(1+r)^n / ((1+r)^n - 1)
//!   CF = Capacity Factor
//!   8760 = hours per year
//! ```
//!
//! ## Cost Categories
//!
//! 1. **Capital**: Initial construction, equipment, installation
//! 2. **O&M**: Operations, maintenance, staffing
//! 3. **Fuel**: Deuterium, tritium (if D-T), replacement parts
//! 4. **Decommissioning**: End-of-life disposal, site restoration

/// Fuel cost parameters
#[derive(Debug, Clone)]
pub struct FuelCosts {
    /// Deuterium cost ($/kg)
    pub deuterium_usd_kg: f64,
    /// Tritium cost ($/g) - only for D-T
    pub tritium_usd_g: f64,
    /// Deuterium consumption rate (kg/MWh)
    pub deuterium_kg_per_mwh: f64,
    /// Tritium consumption rate (g/MWh) - for D-T
    pub tritium_g_per_mwh: f64,
}

impl FuelCosts {
    /// D-D fusion fuel costs
    pub fn dd_fusion() -> Self {
        Self {
            // Deuterium from seawater: ~$500-700/kg (electrolysis + isotope separation)
            deuterium_usd_kg: 600.0,
            tritium_usd_g: 0.0, // Not used in D-D
            // D-D: 3.27 MeV per reaction, 2 deuterons
            // Energy density: ~87,000 MWh/kg of D consumed
            deuterium_kg_per_mwh: 1.15e-5, // ~11.5 mg/MWh
            tritium_g_per_mwh: 0.0,
        }
    }

    /// D-T fusion fuel costs
    pub fn dt_fusion() -> Self {
        Self {
            deuterium_usd_kg: 600.0,
            // Tritium is expensive: ~$30,000/g (produced in reactors)
            tritium_usd_g: 30_000.0,
            // D-T: 17.6 MeV per reaction
            // Much better than D-D
            deuterium_kg_per_mwh: 2.1e-6,
            tritium_g_per_mwh: 3.2e-6 * 1000.0, // Convert to g
        }
    }

    /// D-He3 aneutronic fusion fuel costs
    /// He-3 is extremely rare and expensive (lunar mining required for scale)
    pub fn dhe3_fusion() -> Self {
        Self {
            deuterium_usd_kg: 600.0,
            // He-3 costs ~$15,000/g ($15M/kg) - extremely limited supply
            // Using tritium field to represent He-3 for cost calculation
            // Current sources: tritium decay, lunar regolith, gas giant atmospheres
            tritium_usd_g: 15_000.0, // He-3 at $15k/g
            // D-He3: 18.3 MeV per reaction (D + He3 → He4 + p)
            // Slightly better energy per reaction than D-T
            deuterium_kg_per_mwh: 1.9e-6,
            // He-3 consumption: ~same mass ratio as D
            tritium_g_per_mwh: 2.9e-3, // ~2.9 g He-3 per MWh
        }
    }

    /// Fuel cost per MWh
    pub fn cost_per_mwh(&self) -> f64 {
        self.deuterium_usd_kg * self.deuterium_kg_per_mwh
            + self.tritium_usd_g * self.tritium_g_per_mwh
    }
}

/// Capital cost breakdown
#[derive(Debug, Clone)]
pub struct CapitalCosts {
    /// Fusion core (HEA shell, interface, containment)
    pub fusion_core_usd: f64,
    /// Neutron shielding
    pub shielding_usd: f64,
    /// Power conversion (turbine, generator)
    pub power_conversion_usd: f64,
    /// Balance of plant (cooling, controls, etc.)
    pub balance_of_plant_usd: f64,
    /// Installation and commissioning
    pub installation_usd: f64,
    /// Engineering and contingency (% of above)
    pub contingency_fraction: f64,
}

impl CapitalCosts {
    /// Consumer unit (5 kW)
    pub fn consumer_5kw() -> Self {
        Self {
            fusion_core_usd: 150_000.0,
            shielding_usd: 50_000.0,
            power_conversion_usd: 25_000.0,
            balance_of_plant_usd: 30_000.0,
            installation_usd: 20_000.0,
            contingency_fraction: 0.20,
        }
    }

    /// Industrial unit (100 MW)
    pub fn industrial_100mw() -> Self {
        Self {
            fusion_core_usd: 50_000_000.0,
            shielding_usd: 30_000_000.0,
            power_conversion_usd: 40_000_000.0,
            balance_of_plant_usd: 25_000_000.0,
            installation_usd: 15_000_000.0,
            contingency_fraction: 0.25,
        }
    }

    /// Total capital cost
    pub fn total(&self) -> f64 {
        let base = self.fusion_core_usd
            + self.shielding_usd
            + self.power_conversion_usd
            + self.balance_of_plant_usd
            + self.installation_usd;
        base * (1.0 + self.contingency_fraction)
    }

    /// Cost per kW
    pub fn cost_per_kw(&self, power_kw: f64) -> f64 {
        self.total() / power_kw
    }
}

/// Operations and maintenance costs
#[derive(Debug, Clone)]
pub struct OmCosts {
    /// Fixed O&M ($/kW-year)
    pub fixed_usd_kw_year: f64,
    /// Variable O&M ($/MWh)
    pub variable_usd_mwh: f64,
    /// Major overhaul cost ($/event)
    pub overhaul_cost_usd: f64,
    /// Overhaul interval (years)
    pub overhaul_interval_years: f64,
}

impl OmCosts {
    /// Consumer unit O&M
    pub fn consumer() -> Self {
        Self {
            fixed_usd_kw_year: 50.0, // Annual maintenance contract
            variable_usd_mwh: 5.0,
            overhaul_cost_usd: 10_000.0, // Core replacement
            overhaul_interval_years: 10.0,
        }
    }

    /// Industrial unit O&M
    pub fn industrial() -> Self {
        Self {
            fixed_usd_kw_year: 30.0, // Economy of scale
            variable_usd_mwh: 3.0,
            overhaul_cost_usd: 5_000_000.0,
            overhaul_interval_years: 5.0,
        }
    }

    /// Annual O&M cost
    pub fn annual_cost(&self, power_kw: f64, capacity_factor: f64) -> f64 {
        let fixed = self.fixed_usd_kw_year * power_kw;
        let variable = self.variable_usd_mwh * power_kw * capacity_factor * 8.76; // MWh/year
        let overhaul_annual = self.overhaul_cost_usd / self.overhaul_interval_years;
        fixed + variable + overhaul_annual
    }
}

/// LCOE calculation result
#[derive(Debug, Clone)]
pub struct LcoeResult {
    /// Levelized cost of energy ($/MWh)
    pub lcoe_usd_mwh: f64,
    /// Same in $/kWh for consumer comparison
    pub lcoe_usd_kwh: f64,
    /// Capital contribution to LCOE ($/MWh)
    pub capital_component: f64,
    /// O&M contribution ($/MWh)
    pub om_component: f64,
    /// Fuel contribution ($/MWh)
    pub fuel_component: f64,
    /// Total lifetime cost ($)
    pub lifetime_cost_usd: f64,
    /// Total lifetime energy (MWh)
    pub lifetime_energy_mwh: f64,
    /// Simple payback period (years) vs grid
    pub payback_years: f64,
    /// Net present value ($) over lifetime
    pub npv_usd: f64,
    /// Internal rate of return (%)
    pub irr_percent: f64,
}

/// Comparison with alternative energy sources
#[derive(Debug, Clone)]
pub struct EnergyComparison {
    /// Source name
    pub name: String,
    /// LCOE ($/MWh)
    pub lcoe_usd_mwh: f64,
    /// Capacity factor
    pub capacity_factor: f64,
    /// Dispatchable (on-demand)?
    pub dispatchable: bool,
    /// CO2 emissions (kg/MWh)
    pub co2_kg_mwh: f64,
    /// Land use (m²/kW)
    pub land_use_m2_kw: f64,
}

impl EnergyComparison {
    /// Grid electricity (US average 2024)
    pub fn grid_electricity() -> Self {
        Self {
            name: "Grid Electricity (US avg)".to_string(),
            lcoe_usd_mwh: 120.0,  // ~$0.12/kWh
            capacity_factor: 1.0, // On demand
            dispatchable: true,
            co2_kg_mwh: 400.0, // Mix of sources
            land_use_m2_kw: 0.0,
        }
    }

    /// Solar PV (utility scale)
    pub fn solar_pv() -> Self {
        Self {
            name: "Solar PV (utility)".to_string(),
            lcoe_usd_mwh: 35.0, // Cheapest new generation
            capacity_factor: 0.25,
            dispatchable: false,
            co2_kg_mwh: 40.0, // Lifecycle
            land_use_m2_kw: 20.0,
        }
    }

    /// Wind (onshore)
    pub fn wind_onshore() -> Self {
        Self {
            name: "Wind (onshore)".to_string(),
            lcoe_usd_mwh: 40.0,
            capacity_factor: 0.35,
            dispatchable: false,
            co2_kg_mwh: 15.0,
            land_use_m2_kw: 50.0,
        }
    }

    /// Natural gas combined cycle
    pub fn natural_gas_cc() -> Self {
        Self {
            name: "Natural Gas CC".to_string(),
            lcoe_usd_mwh: 55.0,
            capacity_factor: 0.55,
            dispatchable: true,
            co2_kg_mwh: 400.0,
            land_use_m2_kw: 0.5,
        }
    }

    /// Nuclear fission
    pub fn nuclear_fission() -> Self {
        Self {
            name: "Nuclear Fission".to_string(),
            lcoe_usd_mwh: 90.0, // New construction
            capacity_factor: 0.92,
            dispatchable: true,
            co2_kg_mwh: 12.0,
            land_use_m2_kw: 0.3,
        }
    }

    /// Battery storage (for comparison)
    pub fn battery_storage() -> Self {
        Self {
            name: "Battery Storage (4hr)".to_string(),
            lcoe_usd_mwh: 150.0,   // Adds to generation cost
            capacity_factor: 0.15, // Limited by cycles
            dispatchable: true,
            co2_kg_mwh: 50.0, // Manufacturing
            land_use_m2_kw: 1.0,
        }
    }
}

/// Economic analysis engine
pub struct EconomicEngine {
    /// Discount rate (real, after inflation)
    pub discount_rate: f64,
    /// System lifetime (years)
    pub lifetime_years: f64,
    /// Capacity factor
    pub capacity_factor: f64,
    /// Power output (kW)
    pub power_kw: f64,
}

impl EconomicEngine {
    /// Create for consumer unit
    pub fn consumer(power_kw: f64) -> Self {
        Self {
            discount_rate: 0.05, // 5% real
            lifetime_years: 25.0,
            capacity_factor: 0.90, // High availability
            power_kw,
        }
    }

    /// Create for industrial unit
    pub fn industrial(power_kw: f64) -> Self {
        Self {
            discount_rate: 0.08, // 8% real (higher risk)
            lifetime_years: 40.0,
            capacity_factor: 0.85,
            power_kw,
        }
    }

    /// Capital recovery factor
    pub fn capital_recovery_factor(&self) -> f64 {
        let r = self.discount_rate;
        let n = self.lifetime_years;
        r * (1.0 + r).powf(n) / ((1.0 + r).powf(n) - 1.0)
    }

    /// Annual energy production (MWh)
    pub fn annual_energy_mwh(&self) -> f64 {
        self.power_kw * self.capacity_factor * 8.76 // kW * CF * hours/1000
    }

    /// Lifetime energy production (MWh)
    pub fn lifetime_energy_mwh(&self) -> f64 {
        self.annual_energy_mwh() * self.lifetime_years
    }

    /// Calculate LCOE
    pub fn calculate_lcoe(
        &self,
        capital: &CapitalCosts,
        om: &OmCosts,
        fuel: &FuelCosts,
    ) -> LcoeResult {
        let crf = self.capital_recovery_factor();
        let annual_energy = self.annual_energy_mwh();
        let lifetime_energy = self.lifetime_energy_mwh();

        // Annual costs
        let annual_capital = capital.total() * crf;
        let annual_om = om.annual_cost(self.power_kw, self.capacity_factor);
        let annual_fuel = fuel.cost_per_mwh() * annual_energy;

        // LCOE components ($/MWh)
        let capital_component = annual_capital / annual_energy;
        let om_component = annual_om / annual_energy;
        let fuel_component = fuel.cost_per_mwh();

        let lcoe_usd_mwh = capital_component + om_component + fuel_component;
        let lcoe_usd_kwh = lcoe_usd_mwh / 1000.0;

        // Lifetime costs
        let lifetime_cost = capital.total()
            + om.annual_cost(self.power_kw, self.capacity_factor) * self.lifetime_years
            + annual_fuel * self.lifetime_years;

        // Simple payback vs grid ($0.12/kWh)
        let grid_rate = 0.12; // $/kWh
        let annual_savings = annual_energy * 1000.0 * grid_rate - annual_om - annual_fuel;
        let payback_years = if annual_savings > 0.0 {
            capital.total() / annual_savings
        } else {
            f64::INFINITY
        };

        // NPV calculation
        let mut npv = -capital.total();
        for year in 1..=(self.lifetime_years as usize) {
            let cash_flow = annual_savings;
            npv += cash_flow / (1.0 + self.discount_rate).powi(year as i32);
        }

        // IRR approximation (simplified)
        let irr = if payback_years < self.lifetime_years && annual_savings > 0.0 {
            (annual_savings / capital.total() - self.discount_rate / 2.0) * 100.0
        } else {
            0.0
        };

        LcoeResult {
            lcoe_usd_mwh,
            lcoe_usd_kwh,
            capital_component,
            om_component,
            fuel_component,
            lifetime_cost_usd: lifetime_cost,
            lifetime_energy_mwh: lifetime_energy,
            payback_years,
            npv_usd: npv,
            irr_percent: irr.max(0.0),
        }
    }

    /// Sensitivity analysis on LCOE
    pub fn sensitivity_analysis(
        &self,
        capital: &CapitalCosts,
        om: &OmCosts,
        fuel: &FuelCosts,
    ) -> Vec<(String, f64, f64)> {
        let baseline = self.calculate_lcoe(capital, om, fuel);
        let mut sensitivities = Vec::new();

        // Capital cost ±20%
        let mut cap_high = capital.clone();
        cap_high.fusion_core_usd *= 1.2;
        cap_high.shielding_usd *= 1.2;
        let lcoe_cap_high = self.calculate_lcoe(&cap_high, om, fuel).lcoe_usd_mwh;
        sensitivities.push((
            "Capital +20%".to_string(),
            (lcoe_cap_high - baseline.lcoe_usd_mwh) / baseline.lcoe_usd_mwh * 100.0,
            lcoe_cap_high,
        ));

        // Discount rate +2%
        let mut engine_high_r = self.clone();
        engine_high_r.discount_rate += 0.02;
        let lcoe_r_high = engine_high_r.calculate_lcoe(capital, om, fuel).lcoe_usd_mwh;
        sensitivities.push((
            "Discount rate +2%".to_string(),
            (lcoe_r_high - baseline.lcoe_usd_mwh) / baseline.lcoe_usd_mwh * 100.0,
            lcoe_r_high,
        ));

        // Capacity factor -10%
        let mut engine_low_cf = self.clone();
        engine_low_cf.capacity_factor *= 0.9;
        let lcoe_cf_low = engine_low_cf.calculate_lcoe(capital, om, fuel).lcoe_usd_mwh;
        sensitivities.push((
            "Capacity factor -10%".to_string(),
            (lcoe_cf_low - baseline.lcoe_usd_mwh) / baseline.lcoe_usd_mwh * 100.0,
            lcoe_cf_low,
        ));

        // Lifetime -5 years
        let mut engine_short = self.clone();
        engine_short.lifetime_years -= 5.0;
        let lcoe_short = engine_short.calculate_lcoe(capital, om, fuel).lcoe_usd_mwh;
        sensitivities.push((
            "Lifetime -5 years".to_string(),
            (lcoe_short - baseline.lcoe_usd_mwh) / baseline.lcoe_usd_mwh * 100.0,
            lcoe_short,
        ));

        // O&M +50%
        let mut om_high = om.clone();
        om_high.fixed_usd_kw_year *= 1.5;
        om_high.variable_usd_mwh *= 1.5;
        let lcoe_om_high = self.calculate_lcoe(capital, &om_high, fuel).lcoe_usd_mwh;
        sensitivities.push((
            "O&M +50%".to_string(),
            (lcoe_om_high - baseline.lcoe_usd_mwh) / baseline.lcoe_usd_mwh * 100.0,
            lcoe_om_high,
        ));

        sensitivities
    }

    /// Print economic analysis
    pub fn print_analysis(
        &self,
        name: &str,
        result: &LcoeResult,
        sensitivities: &[(String, f64, f64)],
    ) {
        println!("\n");
        println!("┌────────────────────────────────────────────────────────────────────┐");
        println!("│              ECONOMIC VIABILITY ANALYSIS                           │");
        println!("│              {name}                                            │");
        println!("├────────────────────────────────────────────────────────────────────┤");
        println!("│                                                                    │");
        println!("│  LCOE BREAKDOWN                                                    │");
        println!("│  ─────────────────────────────────────────────────────────────     │");
        println!(
            "│  Capital:     ${:>8.1}/MWh  ({:>4.1}%)                            │",
            result.capital_component,
            result.capital_component / result.lcoe_usd_mwh * 100.0
        );
        println!(
            "│  O&M:         ${:>8.1}/MWh  ({:>4.1}%)                            │",
            result.om_component,
            result.om_component / result.lcoe_usd_mwh * 100.0
        );
        println!(
            "│  Fuel:        ${:>8.2}/MWh  ({:>4.1}%)                            │",
            result.fuel_component,
            result.fuel_component / result.lcoe_usd_mwh * 100.0
        );
        println!("│  ─────────────────────────────────────────────────────────────     │");
        println!(
            "│  TOTAL LCOE:  ${:>8.1}/MWh  (${:.3}/kWh)                        │",
            result.lcoe_usd_mwh, result.lcoe_usd_kwh
        );
        println!("│                                                                    │");
        println!("│  FINANCIAL METRICS                                                 │");
        println!("│  ─────────────────────────────────────────────────────────────     │");
        println!(
            "│  Lifetime cost:      ${:>12.0}                              │",
            result.lifetime_cost_usd
        );
        println!(
            "│  Lifetime energy:    {:>12.0} MWh                            │",
            result.lifetime_energy_mwh
        );
        println!(
            "│  Payback (vs grid):  {:>8.1} years                              │",
            result.payback_years
        );
        println!(
            "│  NPV:                ${:>12.0}                              │",
            result.npv_usd
        );
        println!(
            "│  IRR:                {:>8.1}%                                   │",
            result.irr_percent
        );
        println!("│                                                                    │");
        println!("│  SENSITIVITY ANALYSIS                                              │");
        println!("│  ─────────────────────────────────────────────────────────────     │");
        for (param, change_pct, new_lcoe) in sensitivities {
            let direction = if *change_pct > 0.0 { "↑" } else { "↓" };
            println!(
                "│  {:25} {} {:>5.1}% → ${:.1}/MWh           │",
                param,
                direction,
                change_pct.abs(),
                new_lcoe
            );
        }
        println!("│                                                                    │");
        println!("└────────────────────────────────────────────────────────────────────┘");
    }

    /// Compare with alternatives
    pub fn print_comparison(&self, spark_lcoe: f64) {
        let alternatives = vec![
            EnergyComparison::grid_electricity(),
            EnergyComparison::solar_pv(),
            EnergyComparison::wind_onshore(),
            EnergyComparison::natural_gas_cc(),
            EnergyComparison::nuclear_fission(),
        ];

        println!("\n");
        println!("┌────────────────────────────────────────────────────────────────────┐");
        println!("│              COMPARISON WITH ALTERNATIVES                          │");
        println!("├────────────────────────────────────────────────────────────────────┤");
        println!("│                                                                    │");
        println!(
            "│  {:25} {:>10} {:>8} {:>10} {:>8}   │",
            "Source", "LCOE", "CF", "CO2", "Dispatch"
        );
        println!(
            "│  {:25} {:>10} {:>8} {:>10} {:>8}   │",
            "", "$/MWh", "%", "kg/MWh", ""
        );
        println!("│  ─────────────────────────────────────────────────────────────     │");

        println!(
            "│  {:25} {:>10.1} {:>7.0}% {:>10.0} {:>8}   │",
            "★ Spark Engine (D-D)",
            spark_lcoe,
            self.capacity_factor * 100.0,
            0.0, // Zero CO2
            "Yes"
        );

        for alt in &alternatives {
            let dispatch = if alt.dispatchable { "Yes" } else { "No" };
            println!(
                "│  {:25} {:>10.1} {:>7.0}% {:>10.0} {:>8}   │",
                alt.name,
                alt.lcoe_usd_mwh,
                alt.capacity_factor * 100.0,
                alt.co2_kg_mwh,
                dispatch
            );
        }

        println!("│                                                                    │");
        println!("│  KEY ADVANTAGES OF SPARK ENGINE:                                   │");
        println!("│  • Zero CO2 emissions (vs 400 kg/MWh for gas/grid)                │");
        println!("│  • High capacity factor (90%) - always available                  │");
        println!("│  • Dispatchable - no storage needed                               │");
        println!("│  • Minimal land use                                               │");
        println!("│  • Fuel supply: seawater (essentially unlimited)                  │");
        println!("│                                                                    │");
        println!("└────────────────────────────────────────────────────────────────────┘");
    }
}

/// Result of break-even analysis for a single parameter.
#[derive(Debug, Clone)]
pub struct BreakEvenResult {
    /// Parameter name
    pub parameter: String,
    /// Current value
    pub current_value: f64,
    /// Break-even value (where LCOE = target)
    pub break_even_value: f64,
    /// Improvement factor needed (break_even / current)
    pub improvement_factor: f64,
    /// Whether the improvement is achievable (<10×)
    pub achievable: bool,
    /// Target LCOE used ($/MWh)
    pub target_lcoe: f64,
}

impl EconomicEngine {
    /// Break-even analysis: find the parameter values where LCOE equals a target.
    ///
    /// Uses binary search on each parameter (capital cost, capacity factor,
    /// lifetime, O&M) to find the value where LCOE = target_lcoe.
    ///
    /// Default target: $120/MWh (US grid parity).
    pub fn break_even_analysis(
        &self,
        capital: &CapitalCosts,
        om: &OmCosts,
        fuel: &FuelCosts,
        target_lcoe: f64,
    ) -> Vec<BreakEvenResult> {
        let mut results = Vec::new();
        let baseline = self.calculate_lcoe(capital, om, fuel);

        // 1. Break-even on capital cost (search for multiplier)
        if baseline.lcoe_usd_mwh > target_lcoe {
            let be = self.binary_search_break_even(0.01, 10.0, target_lcoe, |factor| {
                let mut cap = capital.clone();
                cap.fusion_core_usd *= factor;
                cap.shielding_usd *= factor;
                cap.power_conversion_usd *= factor;
                cap.balance_of_plant_usd *= factor;
                cap.installation_usd *= factor;
                self.calculate_lcoe(&cap, om, fuel).lcoe_usd_mwh
            });
            results.push(BreakEvenResult {
                parameter: "Capital cost multiplier".to_string(),
                current_value: 1.0,
                break_even_value: be,
                improvement_factor: if be < 1.0 { 1.0 / be } else { be },
                achievable: be > 0.1,
                target_lcoe,
            });
        } else {
            results.push(BreakEvenResult {
                parameter: "Capital cost multiplier".to_string(),
                current_value: 1.0,
                break_even_value: 1.0,
                improvement_factor: 1.0,
                achievable: true,
                target_lcoe,
            });
        }

        // 2. Break-even on capacity factor
        {
            let be = self.binary_search_break_even(0.1, 0.99, target_lcoe, |cf| {
                let mut eng = self.clone();
                eng.capacity_factor = cf;
                eng.calculate_lcoe(capital, om, fuel).lcoe_usd_mwh
            });
            results.push(BreakEvenResult {
                parameter: "Capacity factor".to_string(),
                current_value: self.capacity_factor,
                break_even_value: be,
                improvement_factor: be / self.capacity_factor,
                achievable: be <= 0.99,
                target_lcoe,
            });
        }

        // 3. Break-even on lifetime
        {
            let be = self.binary_search_break_even(1.0, 100.0, target_lcoe, |lt| {
                let mut eng = self.clone();
                eng.lifetime_years = lt;
                eng.calculate_lcoe(capital, om, fuel).lcoe_usd_mwh
            });
            results.push(BreakEvenResult {
                parameter: "Lifetime (years)".to_string(),
                current_value: self.lifetime_years,
                break_even_value: be,
                improvement_factor: be / self.lifetime_years,
                achievable: be < self.lifetime_years * 10.0,
                target_lcoe,
            });
        }

        // 4. Break-even on O&M multiplier
        {
            let be = self.binary_search_break_even(0.01, 10.0, target_lcoe, |factor| {
                let mut o = om.clone();
                o.fixed_usd_kw_year *= factor;
                o.variable_usd_mwh *= factor;
                o.overhaul_cost_usd *= factor;
                self.calculate_lcoe(capital, &o, fuel).lcoe_usd_mwh
            });
            results.push(BreakEvenResult {
                parameter: "O&M cost multiplier".to_string(),
                current_value: 1.0,
                break_even_value: be,
                improvement_factor: if be < 1.0 { 1.0 / be } else { be },
                achievable: be > 0.1,
                target_lcoe,
            });
        }

        results
    }

    /// Binary search for parameter value where LCOE = target.
    /// `eval_fn` maps parameter value → LCOE.
    fn binary_search_break_even<F>(&self, low: f64, high: f64, target: f64, eval_fn: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut lo = low;
        let mut hi = high;
        let lcoe_lo = eval_fn(lo);
        let lcoe_hi = eval_fn(hi);

        // Determine search direction
        let increasing = lcoe_hi > lcoe_lo;

        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let lcoe_mid = eval_fn(mid);

            if (lcoe_mid - target).abs() < 0.01 {
                return mid;
            }

            if increasing {
                if lcoe_mid > target {
                    hi = mid;
                } else {
                    lo = mid;
                }
            } else if lcoe_mid > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        (lo + hi) / 2.0
    }
}

impl Clone for EconomicEngine {
    fn clone(&self) -> Self {
        Self {
            discount_rate: self.discount_rate,
            lifetime_years: self.lifetime_years,
            capacity_factor: self.capacity_factor,
            power_kw: self.power_kw,
        }
    }
}

// =============================================================================
// Direction D: Comparative Benchmarking
// =============================================================================

/// Data point for a Ragone plot (specific power vs energy density).
#[derive(Debug, Clone)]
pub struct RagoneDataPoint {
    /// Technology or configuration name
    pub name: String,
    /// Specific power (W/kg)
    pub specific_power_w_kg: f64,
    /// Energy density (Wh/kg)
    pub energy_density_wh_kg: f64,
    /// Category for grouping in plots
    pub category: String,
}

/// Ragone plot comparison of LCF against reference technologies.
#[derive(Debug, Clone)]
pub struct RagoneComparison {
    /// All data points (LCF + reference technologies)
    pub data_points: Vec<RagoneDataPoint>,
}

impl RagoneComparison {
    /// Build a Ragone comparison including LCF data from a scaling study
    /// and 9 reference technologies.
    ///
    /// `lcf_data` is a list of (name, power_w, mass_kg, lifetime_hours) tuples
    /// from the scaling study results.
    pub fn build(lcf_data: &[(String, f64, f64, f64)]) -> Self {
        let mut data_points = Vec::new();

        // Add LCF data points from scaling study
        for (name, power_w, mass_kg, lifetime_hours) in lcf_data {
            if *mass_kg > 0.0 {
                let specific_power = power_w / mass_kg;
                let energy_density = power_w * lifetime_hours / mass_kg;
                data_points.push(RagoneDataPoint {
                    name: name.clone(),
                    specific_power_w_kg: specific_power,
                    energy_density_wh_kg: energy_density,
                    category: "LCF Reactor".to_string(),
                });
            }
        }

        // Reference technologies (representative values from literature)
        let references = vec![
            ("Li-ion Battery", 300.0, 250.0, "Electrochemical"),
            ("PEM Fuel Cell", 500.0, 1_000.0, "Electrochemical"),
            ("Diesel Generator", 1_000.0, 12_000.0, "Combustion"),
            ("Pu-238 RTG", 5.0, 500_000.0, "Nuclear (Decay)"),
            ("Solar + Battery", 100.0, 200.0, "Renewable"),
            ("SMR Fission", 10.0, 1.0e8, "Nuclear (Fission)"),
            ("Micro Gas Turbine", 2_000.0, 3_000.0, "Combustion"),
            ("Supercapacitor", 10_000.0, 10.0, "Electrochemical"),
            ("Hydrogen FC System", 300.0, 800.0, "Electrochemical"),
        ];

        for (name, sp, ed, cat) in references {
            data_points.push(RagoneDataPoint {
                name: name.to_string(),
                specific_power_w_kg: sp,
                energy_density_wh_kg: ed,
                category: cat.to_string(),
            });
        }

        Self { data_points }
    }
}

/// Scale-aware cost comparison at a specific power level.
#[derive(Debug, Clone)]
pub struct ScaleCostPoint {
    /// Technology name
    pub technology: String,
    /// Cost per watt ($/W)
    pub cost_per_w: f64,
    /// LCOE at this scale ($/MWh)
    pub lcoe_usd_mwh: f64,
}

/// Scale-aware $/W and LCOE comparison across power levels.
#[derive(Debug, Clone)]
pub struct ScaleAwareComparison {
    /// Power level (W) for this comparison
    pub power_w: f64,
    /// Cost comparisons at this scale
    pub technologies: Vec<ScaleCostPoint>,
}

impl ScaleAwareComparison {
    /// Build comparisons across multiple power levels.
    ///
    /// Uses power-law scaling for $/W for each competing technology.
    /// `lcf_points` is a list of (power_w, cost_usd, lcoe_usd_mwh) from the scaling study.
    pub fn build_all(
        power_levels: &[f64],
        lcf_points: &[(f64, f64, f64)],
    ) -> Vec<ScaleAwareComparison> {
        power_levels
            .iter()
            .map(|&power_w| {
                let mut technologies = Vec::new();

                // Find closest LCF data point
                if let Some(&(_, cost_usd, lcoe)) = lcf_points
                    .iter()
                    .min_by(|a, b| (a.0 - power_w).abs().total_cmp(&(b.0 - power_w).abs()))
                {
                    technologies.push(ScaleCostPoint {
                        technology: "LCF Reactor".to_string(),
                        cost_per_w: cost_usd / power_w,
                        lcoe_usd_mwh: lcoe,
                    });
                }

                // Competing technologies with power-law $/W scaling
                // Format: (name, ref_cost_per_w at 1kW, scaling_exponent, base_lcoe)
                let competitors: Vec<(&str, f64, f64, f64)> = vec![
                    // Solar + battery: $3/W at 1kW, improves at scale
                    ("Solar + Battery", 3.0, -0.15, 80.0),
                    // Diesel generator: $0.50/W at 1kW
                    ("Diesel Generator", 0.50, -0.10, 250.0),
                    // Grid connection: $0.01/W at 1kW (just connection cost)
                    ("Grid Connection", 0.01, -0.05, 120.0),
                    // RTG: $300/W at 1kW (extremely expensive)
                    ("Pu-238 RTG", 300.0, -0.05, 50_000.0),
                    // PEM fuel cell: $5/W at 1kW
                    ("PEM Fuel Cell", 5.0, -0.20, 200.0),
                    // Micro-nuclear (SMR): $20/W at 1MW reference
                    ("Micro-Nuclear (SMR)", 20.0, -0.30, 90.0),
                ];

                let power_kw = power_w / 1000.0;
                for (name, ref_cpw, exp, base_lcoe) in competitors {
                    // Power-law: $/W = ref × (P/1kW)^exp
                    let cost_per_w = ref_cpw * (power_kw).powf(exp);
                    // LCOE also scales (very roughly) with size
                    let lcoe = base_lcoe * (power_kw / 1.0).powf(exp * 0.5);
                    technologies.push(ScaleCostPoint {
                        technology: name.to_string(),
                        cost_per_w,
                        lcoe_usd_mwh: lcoe,
                    });
                }

                ScaleAwareComparison {
                    power_w,
                    technologies,
                }
            })
            .collect()
    }
}

/// Technology readiness assessment.
#[derive(Debug, Clone)]
pub struct ReadinessComparison {
    /// Technology name
    pub technology: String,
    /// Technology Readiness Level (1-9)
    pub trl: u8,
    /// Estimated years to deployment
    pub years_to_deployment: f64,
    /// R&D funding needed (USD)
    pub rd_funding_needed_usd: f64,
    /// Key technical risks
    pub key_risks: Vec<String>,
}

impl ReadinessComparison {
    /// Assessment for LCF (Lattice Confinement Fusion).
    pub fn lcf_assessment() -> Self {
        Self {
            technology: "Lattice Confinement Fusion".to_string(),
            trl: 3, // Experimental proof of concept (NASA 2020 results)
            years_to_deployment: 15.0,
            rd_funding_needed_usd: 500_000_000.0, // $500M estimated
            key_risks: vec![
                "Reaction rate enhancement reproducibility".to_string(),
                "Lattice degradation under sustained operation".to_string(),
                "Net energy gain (Q > 1) not yet demonstrated".to_string(),
                "Scaling from microgram to engineering scale".to_string(),
                "Neutron shielding for compact form factors".to_string(),
            ],
        }
    }

    /// Assessments for competing technologies.
    pub fn competing_technologies() -> Vec<Self> {
        vec![
            Self {
                technology: "Magnetic Confinement (Tokamak)".to_string(),
                trl: 6,
                years_to_deployment: 15.0,
                rd_funding_needed_usd: 25_000_000_000.0, // ITER-class
                key_risks: vec![
                    "Plasma disruptions".to_string(),
                    "First-wall materials lifetime".to_string(),
                    "Tritium breeding ratio > 1".to_string(),
                ],
            },
            Self {
                technology: "Inertial Confinement (NIF-type)".to_string(),
                trl: 5,
                years_to_deployment: 25.0,
                rd_funding_needed_usd: 10_000_000_000.0,
                key_risks: vec![
                    "Driver efficiency (laser → target)".to_string(),
                    "Target manufacturing at scale".to_string(),
                    "Rep-rate operation".to_string(),
                ],
            },
            Self {
                technology: "Small Modular Reactor (Fission)".to_string(),
                trl: 7,
                years_to_deployment: 5.0,
                rd_funding_needed_usd: 2_000_000_000.0,
                key_risks: vec![
                    "Regulatory approval timeline".to_string(),
                    "Public acceptance".to_string(),
                    "Supply chain for HALEU fuel".to_string(),
                ],
            },
            Self {
                technology: "Advanced Geothermal".to_string(),
                trl: 6,
                years_to_deployment: 5.0,
                rd_funding_needed_usd: 1_000_000_000.0,
                key_risks: vec![
                    "Deep drilling costs".to_string(),
                    "Reservoir sustainability".to_string(),
                ],
            },
            Self {
                technology: "Compact Fusion (Private)".to_string(),
                trl: 4,
                years_to_deployment: 10.0,
                rd_funding_needed_usd: 5_000_000_000.0,
                key_risks: vec![
                    "Plasma confinement at high beta".to_string(),
                    "Engineering Q > 10".to_string(),
                    "Neutron damage to compact structures".to_string(),
                ],
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuel_costs() {
        let dd = FuelCosts::dd_fusion();
        let dt = FuelCosts::dt_fusion();

        // D-D should have lower fuel cost (no tritium)
        assert!(dd.cost_per_mwh() < dt.cost_per_mwh());

        // Fuel cost should be very low compared to fossils
        assert!(dd.cost_per_mwh() < 1.0); // <$1/MWh
    }

    #[test]
    fn test_capital_costs() {
        let consumer = CapitalCosts::consumer_5kw();
        let industrial = CapitalCosts::industrial_100mw();

        // Industrial should have lower $/kW (economy of scale)
        let consumer_per_kw = consumer.cost_per_kw(5.0);
        let industrial_per_kw = industrial.cost_per_kw(100_000.0);

        assert!(industrial_per_kw < consumer_per_kw);
    }

    #[test]
    fn test_crf() {
        let engine = EconomicEngine::consumer(5.0);
        let crf = engine.capital_recovery_factor();

        // CRF should be reasonable (5-15% for 20-30 year life)
        assert!(crf > 0.05 && crf < 0.15);
    }

    #[test]
    fn test_lcoe_calculation() {
        let engine = EconomicEngine::consumer(5.0);
        let capital = CapitalCosts::consumer_5kw();
        let om = OmCosts::consumer();
        let fuel = FuelCosts::dd_fusion();

        let result = engine.calculate_lcoe(&capital, &om, &fuel);

        // LCOE should be positive
        assert!(result.lcoe_usd_mwh > 0.0);

        // Components should sum to total
        let sum = result.capital_component + result.om_component + result.fuel_component;
        assert!((sum - result.lcoe_usd_mwh).abs() < 0.1);

        // $/kWh should be $/MWh / 1000
        assert!((result.lcoe_usd_kwh - result.lcoe_usd_mwh / 1000.0).abs() < 0.0001);
    }

    #[test]
    fn test_comparison_sources() {
        let grid = EnergyComparison::grid_electricity();
        let solar = EnergyComparison::solar_pv();

        // Grid should be dispatchable, solar should not
        assert!(grid.dispatchable);
        assert!(!solar.dispatchable);

        // Solar should have lower CO2
        assert!(solar.co2_kg_mwh < grid.co2_kg_mwh);
    }

    // === Direction C: Break-Even Analysis Tests ===

    #[test]
    fn test_break_even_analysis_returns_four_params() {
        let engine = EconomicEngine::consumer(5.0);
        let capital = CapitalCosts::consumer_5kw();
        let om = OmCosts::consumer();
        let fuel = FuelCosts::dd_fusion();

        let results = engine.break_even_analysis(&capital, &om, &fuel, 120.0);

        assert_eq!(results.len(), 4, "Should have 4 break-even parameters");
        assert_eq!(results[0].parameter, "Capital cost multiplier");
        assert_eq!(results[1].parameter, "Capacity factor");
        assert_eq!(results[2].parameter, "Lifetime (years)");
        assert_eq!(results[3].parameter, "O&M cost multiplier");

        // All improvement factors should be positive
        for r in &results {
            assert!(
                r.improvement_factor > 0.0,
                "Improvement factor for {} should be positive, got {}",
                r.parameter,
                r.improvement_factor
            );
            assert_eq!(r.target_lcoe, 120.0);
        }
    }

    #[test]
    fn test_break_even_capacity_factor_bounded() {
        let engine = EconomicEngine::consumer(5.0);
        let capital = CapitalCosts::consumer_5kw();
        let om = OmCosts::consumer();
        let fuel = FuelCosts::dd_fusion();

        let results = engine.break_even_analysis(&capital, &om, &fuel, 120.0);
        let cf_result = &results[1];

        // Capacity factor should be between 0.1 and 0.99
        assert!(
            cf_result.break_even_value >= 0.1 && cf_result.break_even_value <= 0.99,
            "Capacity factor break-even should be in [0.1, 0.99], got {}",
            cf_result.break_even_value
        );
    }

    // === Direction D: Comparative Benchmarking Tests ===

    #[test]
    fn test_ragone_includes_lcf_and_references() {
        let lcf_data = vec![
            ("LCF 1kW".to_string(), 1_000.0, 50.0, 200_000.0),
            ("LCF 100kW".to_string(), 100_000.0, 500.0, 200_000.0),
        ];
        let ragone = RagoneComparison::build(&lcf_data);

        // Should have LCF points + 9 references = 11 total
        assert_eq!(ragone.data_points.len(), 11);

        // LCF points should have correct category
        let lcf_points: Vec<_> = ragone
            .data_points
            .iter()
            .filter(|p| p.category == "LCF Reactor")
            .collect();
        assert_eq!(lcf_points.len(), 2);

        // All points should have positive specific power and energy density
        for point in &ragone.data_points {
            assert!(
                point.specific_power_w_kg > 0.0,
                "{} should have positive specific power",
                point.name
            );
            assert!(
                point.energy_density_wh_kg > 0.0,
                "{} should have positive energy density",
                point.name
            );
        }
    }

    #[test]
    fn test_scale_aware_comparison_all_power_levels() {
        let power_levels = vec![1_000.0, 10_000.0, 1_000_000.0];
        let lcf_points = vec![
            (1_000.0, 50_000.0, 500.0),
            (10_000.0, 100_000.0, 300.0),
            (1_000_000.0, 5_000_000.0, 150.0),
        ];

        let comparisons = ScaleAwareComparison::build_all(&power_levels, &lcf_points);
        assert_eq!(comparisons.len(), 3);

        for comp in &comparisons {
            // Should have LCF + 6 competitors = 7 technologies
            assert_eq!(
                comp.technologies.len(),
                7,
                "Should have 7 technologies at {:.0}W",
                comp.power_w
            );

            // All should have positive cost per watt
            for tech in &comp.technologies {
                assert!(
                    tech.cost_per_w > 0.0,
                    "{} at {:.0}W should have positive $/W",
                    tech.technology,
                    comp.power_w
                );
                assert!(
                    tech.lcoe_usd_mwh > 0.0,
                    "{} at {:.0}W should have positive LCOE",
                    tech.technology,
                    comp.power_w
                );
            }
        }
    }

    #[test]
    fn test_readiness_trl_in_range() {
        let lcf = ReadinessComparison::lcf_assessment();
        assert!(
            lcf.trl >= 1 && lcf.trl <= 9,
            "LCF TRL should be in [1,9], got {}",
            lcf.trl
        );
        assert!(!lcf.key_risks.is_empty());
        assert!(lcf.years_to_deployment > 0.0);
        assert!(lcf.rd_funding_needed_usd > 0.0);

        let competitors = ReadinessComparison::competing_technologies();
        assert!(!competitors.is_empty());
        for comp in &competitors {
            assert!(
                comp.trl >= 1 && comp.trl <= 9,
                "{} TRL should be in [1,9], got {}",
                comp.technology,
                comp.trl
            );
            assert!(!comp.key_risks.is_empty());
        }
    }

    #[test]
    fn test_ragone_reference_values_reasonable() {
        let ragone = RagoneComparison::build(&[]);

        // Should have exactly 9 reference technologies
        assert_eq!(ragone.data_points.len(), 9);

        // RTG should have highest energy density (long lifetime)
        let rtg = ragone
            .data_points
            .iter()
            .find(|p| p.name.contains("RTG"))
            .expect("Should have RTG");
        assert!(rtg.energy_density_wh_kg > 100_000.0);

        // Supercapacitor should have highest specific power
        let supercap = ragone
            .data_points
            .iter()
            .find(|p| p.name.contains("Supercapacitor"))
            .expect("Should have Supercapacitor");
        assert!(supercap.specific_power_w_kg > 5_000.0);
    }

    #[test]
    fn test_scale_aware_cost_decreases_with_scale() {
        let power_levels = vec![1_000.0, 1_000_000.0];
        let lcf_points = vec![
            (1_000.0, 100_000.0, 500.0),
            (1_000_000.0, 5_000_000.0, 100.0),
        ];

        let comparisons = ScaleAwareComparison::build_all(&power_levels, &lcf_points);

        // For technologies with negative scaling exponents, $/W should decrease at larger scale
        let small = &comparisons[0];
        let large = &comparisons[1];

        // Solar should be cheaper per watt at larger scale
        let solar_small = small
            .technologies
            .iter()
            .find(|t| t.technology.contains("Solar"))
            .unwrap();
        let solar_large = large
            .technologies
            .iter()
            .find(|t| t.technology.contains("Solar"))
            .unwrap();
        assert!(
            solar_large.cost_per_w < solar_small.cost_per_w,
            "Solar $/W should decrease with scale: {:.2} vs {:.2}",
            solar_large.cost_per_w,
            solar_small.cost_per_w
        );
    }
}
