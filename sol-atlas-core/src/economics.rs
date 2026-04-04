// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fossil fuel economics — production analysis, ROI computation, carbon cost modeling.
//!
//! Pure Rust, no rendering dependencies. All monetary values in USD.

use crate::types::{FossilDeposit, FuelType};

// ─── Constants ──────────────────────────────────────────────────

/// EPA Social Cost of Carbon (2024 central estimate, $/ton CO2).
pub const SOCIAL_COST_OF_CARBON: f64 = 51.0;

/// CO2 emission factors by fuel type (metric tons CO2 per barrel of oil equivalent).
/// Sources: EPA, EIA, IPCC AR6.
pub fn emission_factor(fuel_type: &FuelType) -> f64 {
    match fuel_type {
        FuelType::Oil => 0.43,
        FuelType::Gas => 0.24,
        FuelType::Coal => 0.64,
        FuelType::TarSands => 0.58,
    }
}

/// Reference market price per barrel of oil equivalent by fuel type (USD, ~2024 averages).
pub fn reference_price_per_boe(fuel_type: &FuelType) -> f64 {
    match fuel_type {
        FuelType::Oil => 75.0,       // WTI/Brent average
        FuelType::Gas => 28.0,       // Henry Hub equivalent per boe
        FuelType::Coal => 18.0,      // thermal coal equivalent per boe
        FuelType::TarSands => 65.0,  // discounted heavy crude
    }
}

// ─── EROI thresholds ────────────────────────────────────────────

/// Full modern civilization with arts & culture (Hall & Lambert).
pub const EROI_CIVILIZATION: f64 = 12.0;
/// Basic sustainability — transport, healthcare, education.
pub const EROI_SUSTAINABILITY: f64 = 5.0;
/// Bare minimum — enough to drive a truck.
pub const EROI_MINIMUM: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EroiTier {
    /// EROI > 12: can sustain modern civilization
    Civilization,
    /// EROI 5-12: basic sustainability
    Sustainability,
    /// EROI 3-5: marginal, declining returns
    Marginal,
    /// EROI < 3: thermodynamically unviable
    Unviable,
}

impl EroiTier {
    pub fn from_eroi(eroi: f64) -> Self {
        if eroi >= EROI_CIVILIZATION {
            Self::Civilization
        } else if eroi >= EROI_SUSTAINABILITY {
            Self::Sustainability
        } else if eroi >= EROI_MINIMUM {
            Self::Marginal
        } else {
            Self::Unviable
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Civilization => "Civilization",
            Self::Sustainability => "Sustainable",
            Self::Marginal => "Marginal",
            Self::Unviable => "Unviable",
        }
    }
}

/// Compute EROI for a deposit. Uses the `eroi` field if set,
/// otherwise derives a proxy from reference price / extraction cost.
pub fn compute_eroi(deposit: &FossilDeposit) -> Option<f64> {
    if let Some(eroi) = deposit.eroi {
        return Some(eroi);
    }
    let cost = deposit.extraction_cost_per_boe?;
    if cost <= 0.0 {
        return None;
    }
    Some(reference_price_per_boe(&deposit.fuel_type) / cost)
}

/// EROI-based color for visualization: green → amber → red → dark red.
pub fn eroi_color(eroi: f64) -> [f32; 3] {
    match EroiTier::from_eroi(eroi) {
        EroiTier::Civilization => [0.06, 0.73, 0.51],   // teal-green
        EroiTier::Sustainability => [0.98, 0.75, 0.14], // amber
        EroiTier::Marginal => [0.94, 0.27, 0.27],       // red
        EroiTier::Unviable => [0.50, 0.10, 0.10],       // dark red
    }
}

// ─── Computed Economics ─────────────────────────────────────────

/// Full economic analysis of a fossil deposit.
#[derive(Debug, Clone)]
pub struct FossilEconomics {
    /// Annual gross revenue (price * production), USD millions.
    pub annual_revenue_m: f64,
    /// Annual extraction cost, USD millions.
    pub annual_extraction_cost_m: f64,
    /// Annual carbon cost at social cost of carbon, USD millions.
    pub annual_carbon_cost_m: f64,
    /// Annual profit excluding carbon cost, USD millions.
    pub annual_profit_no_carbon_m: f64,
    /// Annual profit including carbon cost, USD millions.
    pub annual_profit_with_carbon_m: f64,
    /// ROI excluding carbon cost (%).
    pub roi_no_carbon_pct: f64,
    /// ROI including carbon cost (%).
    pub roi_with_carbon_pct: f64,
    /// Carbon price ($/ton CO2) at which annual profit = 0.
    pub breakeven_carbon_price: f64,
    /// Years until reserves are depleted at current production rate.
    pub years_to_depletion: f64,
    /// Annual CO2 emissions (million metric tons).
    pub co2_annual_mt: f64,
}

/// Compute full economics for a fossil deposit.
/// Returns `None` if `extraction_cost_per_boe` is not set.
pub fn compute_economics(deposit: &FossilDeposit) -> Option<FossilEconomics> {
    let extraction_cost = deposit.extraction_cost_per_boe?;
    let price = reference_price_per_boe(&deposit.fuel_type);
    let ef = emission_factor(&deposit.fuel_type);
    let prod = deposit.annual_production_mboe; // million boe/yr

    let revenue = prod * price;           // $M/yr
    let cost = prod * extraction_cost;    // $M/yr
    let co2_mt = prod * ef;               // Mt CO2/yr
    let carbon_cost = co2_mt * SOCIAL_COST_OF_CARBON; // $M/yr

    let profit_no_carbon = revenue - cost;
    let profit_with_carbon = profit_no_carbon - carbon_cost;

    // ROI as profit / revenue (operating margin)
    let roi_no_carbon = if revenue > 0.0 { profit_no_carbon / revenue * 100.0 } else { 0.0 };
    let roi_with_carbon = if revenue > 0.0 { profit_with_carbon / revenue * 100.0 } else { 0.0 };

    // Breakeven: profit_no_carbon - co2_mt * breakeven_price = 0
    let breakeven = if co2_mt > 0.0 { profit_no_carbon / co2_mt } else { f64::INFINITY };

    let years = if prod > 0.0 {
        deposit.proven_reserves_mboe / prod
    } else {
        f64::INFINITY
    };

    Some(FossilEconomics {
        annual_revenue_m: revenue,
        annual_extraction_cost_m: cost,
        annual_carbon_cost_m: carbon_cost,
        annual_profit_no_carbon_m: profit_no_carbon,
        annual_profit_with_carbon_m: profit_with_carbon,
        roi_no_carbon_pct: roi_no_carbon,
        roi_with_carbon_pct: roi_with_carbon,
        breakeven_carbon_price: breakeven,
        years_to_depletion: years,
        co2_annual_mt: co2_mt,
    })
}

/// Returns true if the deposit loses money when carbon cost is included.
pub fn is_carbon_negative(deposit: &FossilDeposit) -> bool {
    compute_economics(deposit)
        .map(|e| e.annual_profit_with_carbon_m < 0.0)
        .unwrap_or(false)
}

/// Annual CO2 emissions in million metric tons (works even without extraction cost data).
pub fn annual_co2_mt(deposit: &FossilDeposit) -> f64 {
    deposit.annual_production_mboe * emission_factor(&deposit.fuel_type)
}

/// Newton-Raphson IRR computation for a cash flow series.
/// `initial` is the upfront investment (positive number).
/// `cashflows` are annual net cash flows.
/// Returns the rate at which NPV = 0.
pub fn compute_irr(initial: f64, cashflows: &[f64]) -> Option<f64> {
    if cashflows.is_empty() {
        return None;
    }

    let mut rate = 0.10; // initial guess: 10%
    for _ in 0..100 {
        let mut npv = -initial;
        let mut d_npv = 0.0;
        for (t, cf) in cashflows.iter().enumerate() {
            let year = (t + 1) as f64;
            let discount = (1.0_f64 + rate).powf(year);
            npv += cf / discount;
            d_npv -= year * cf / ((1.0_f64 + rate).powf(year + 1.0));
        }

        if d_npv.abs() < 1e-12 {
            break;
        }

        let delta = npv / d_npv;
        rate -= delta;

        if delta.abs() < 1e-8 {
            return Some(rate);
        }
    }

    // Check convergence
    if rate.is_finite() && rate > -1.0 {
        Some(rate)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ghawar() -> FossilDeposit {
        FossilDeposit {
            name: "Ghawar Field".into(),
            lat: 25.4,
            lon: 49.1,
            fuel_type: FuelType::Oil,
            proven_reserves_mboe: 75_000.0,
            annual_production_mboe: 1_825.0,
            status: "declining".into(),
            country: "Saudi Arabia".into(),
            discovery_year: 1948,
            extraction_cost_per_boe: Some(10.0),
            decommission_cost_m: None,
            eroi: None,
        }
    }

    #[test]
    fn test_ghawar_economics() {
        let econ = compute_economics(&ghawar()).unwrap();
        // Revenue: 1825 Mboe * $75/boe = $136,875M
        assert!((econ.annual_revenue_m - 136_875.0).abs() < 1.0);
        // Extraction: 1825 * $10 = $18,250M
        assert!((econ.annual_extraction_cost_m - 18_250.0).abs() < 1.0);
        // CO2: 1825 * 0.43 = 784.75 Mt
        assert!((econ.co2_annual_mt - 784.75).abs() < 0.1);
        // Years to depletion: 75000 / 1825 ≈ 41.1
        assert!((econ.years_to_depletion - 41.1).abs() < 0.1);
        // Profit without carbon should be large and positive
        assert!(econ.annual_profit_no_carbon_m > 100_000.0);
        // Breakeven carbon price should be high for cheap Saudi oil
        assert!(econ.breakeven_carbon_price > 100.0);
    }

    #[test]
    fn test_tar_sands_carbon_negative() {
        let deposit = FossilDeposit {
            name: "Expensive Tar Sands".into(),
            lat: 57.0,
            lon: -111.5,
            fuel_type: FuelType::TarSands,
            proven_reserves_mboe: 1000.0,
            annual_production_mboe: 50.0,
            status: "producing".into(),
            country: "Canada".into(),
            discovery_year: 1970,
            extraction_cost_per_boe: Some(55.0), // expensive extraction
            decommission_cost_m: None,
            eroi: None,
        };
        let econ = compute_economics(&deposit).unwrap();
        // Revenue: 50 * $65 = $3,250M
        // Extraction: 50 * $55 = $2,750M
        // Profit no carbon: $500M
        // Carbon cost: 50 * 0.58 * $51 = $1,479M
        // Profit with carbon: $500 - $1,479 = -$979M (negative!)
        assert!(econ.annual_profit_with_carbon_m < 0.0);
        assert!(is_carbon_negative(&deposit));
    }

    #[test]
    fn test_no_extraction_cost_returns_none() {
        let mut deposit = ghawar();
        deposit.extraction_cost_per_boe = None;
        assert!(compute_economics(&deposit).is_none());
    }

    #[test]
    fn test_annual_co2_mt() {
        let deposit = ghawar();
        let co2 = annual_co2_mt(&deposit);
        assert!((co2 - 784.75).abs() < 0.1);
    }

    #[test]
    fn test_irr_simple() {
        // Invest $1000, get $500/yr for 3 years → IRR ≈ 23.4%
        let irr = compute_irr(1000.0, &[500.0, 500.0, 500.0]).unwrap();
        assert!((irr - 0.234).abs() < 0.01, "IRR was {irr}");
    }

    #[test]
    fn test_irr_breakeven() {
        // Invest $1000, get $1000 back in year 1 → IRR = 0%
        let irr = compute_irr(1000.0, &[1000.0]).unwrap();
        assert!(irr.abs() < 0.001, "IRR was {irr}");
    }

    #[test]
    fn test_eroi_uses_field_when_set() {
        let mut d = ghawar();
        d.eroi = Some(25.0);
        assert_eq!(compute_eroi(&d), Some(25.0));
    }

    #[test]
    fn test_eroi_derives_from_cost() {
        let mut d = ghawar();
        d.eroi = None;
        d.extraction_cost_per_boe = Some(10.0);
        // Oil ref price = 75, so proxy = 75/10 = 7.5
        let eroi = compute_eroi(&d).unwrap();
        assert!((eroi - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_eroi_none_without_data() {
        let mut d = ghawar();
        d.eroi = None;
        d.extraction_cost_per_boe = None;
        assert!(compute_eroi(&d).is_none());
    }

    #[test]
    fn test_eroi_tier_classification() {
        assert_eq!(EroiTier::from_eroi(25.0), EroiTier::Civilization);
        assert_eq!(EroiTier::from_eroi(7.0), EroiTier::Sustainability);
        assert_eq!(EroiTier::from_eroi(4.0), EroiTier::Marginal);
        assert_eq!(EroiTier::from_eroi(2.0), EroiTier::Unviable);
    }

    #[test]
    fn test_eroi_color_thresholds() {
        let green = eroi_color(25.0);
        let amber = eroi_color(7.0);
        let red = eroi_color(4.0);
        assert!(green[1] > green[0], "green should be greenish");
        assert!(amber[0] > amber[2], "amber should be warm");
        assert!(red[0] > red[1], "red should be reddish");
    }

    #[test]
    fn test_emission_factors() {
        assert!((emission_factor(&FuelType::Oil) - 0.43).abs() < 0.001);
        assert!((emission_factor(&FuelType::Gas) - 0.24).abs() < 0.001);
        assert!((emission_factor(&FuelType::Coal) - 0.64).abs() < 0.001);
        assert!((emission_factor(&FuelType::TarSands) - 0.58).abs() < 0.001);
    }
}
