// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cumulative-emissions warming via a central TCRE estimate.
//!
//! These functions implement the near-linear relationship between cumulative
//! CO₂ emissions and warming. They are transparent central-estimate arithmetic,
//! not a complete assessed remaining-carbon-budget calculation: probability
//! level, non-CO₂ forcing, zero-emissions commitment, and unrepresented
//! Earth-system feedbacks are outside this module's current scope.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

/// TCRE central estimate: warming per gigatonne of *carbon* emitted
/// (≈ 1.65 °C per 1000 GtC; assessed range roughly 1.0–2.3).
pub const TCRE_C_PER_GTC: f64 = 1.65e-3;

/// Mass ratio CO₂ : C (44.009 / 12.011 ≈ 3.664).
pub const CO2_TO_C_MASS_RATIO: f64 = 44.009 / 12.011;

/// Warming (°C) from cumulative carbon emissions (GtC), using the central TCRE.
pub fn warming_from_cumulative_carbon(gt_carbon: f64) -> f64 {
    TCRE_C_PER_GTC * gt_carbon
}

/// Warming (°C) from cumulative CO₂ emissions (GtCO₂), using the central TCRE.
pub fn warming_from_cumulative_co2(gt_co2: f64) -> f64 {
    warming_from_cumulative_carbon(gt_co2 / CO2_TO_C_MASS_RATIO)
}

/// Central-TCRE carbon headroom (GtC) between current and target warming.
///
/// This is intentionally not named an assessed “remaining carbon budget”:
/// it includes none of the probability or non-CO₂ adjustments used in formal
/// carbon-budget assessments. The result is clamped to zero once the target is
/// exceeded.
pub fn central_tcre_budget_headroom_carbon(target_warming: f64, warming_so_far: f64) -> f64 {
    let headroom = target_warming - warming_so_far;
    if headroom <= 0.0 {
        0.0
    } else {
        headroom / TCRE_C_PER_GTC
    }
}

/// Central-TCRE CO₂ headroom (GtCO₂) between current and target warming.
pub fn central_tcre_budget_headroom_co2(target_warming: f64, warming_so_far: f64) -> f64 {
    central_tcre_budget_headroom_carbon(target_warming, warming_so_far) * CO2_TO_C_MASS_RATIO
}

/// Backward-compatible alias for [`central_tcre_budget_headroom_carbon`].
#[deprecated(
    since = "0.1.1",
    note = "this is central-TCRE headroom, not a complete assessed budget; use central_tcre_budget_headroom_carbon"
)]
pub fn remaining_carbon_budget(target_warming: f64, warming_so_far: f64) -> f64 {
    central_tcre_budget_headroom_carbon(target_warming, warming_so_far)
}

/// Backward-compatible alias for [`central_tcre_budget_headroom_co2`].
#[deprecated(
    since = "0.1.1",
    note = "this is central-TCRE headroom, not a complete assessed budget; use central_tcre_budget_headroom_co2"
)]
pub fn remaining_co2_budget(target_warming: f64, warming_so_far: f64) -> f64 {
    central_tcre_budget_headroom_co2(target_warming, warming_so_far)
}

/// Ordered low/central/high TCRE values in °C per GtC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcreEstimate {
    pub low_c_per_gtc: f64,
    pub central_c_per_gtc: f64,
    pub high_c_per_gtc: f64,
}

/// The range already documented by this module, represented explicitly.
pub const DEFAULT_TCRE_ESTIMATE: TcreEstimate = TcreEstimate {
    low_c_per_gtc: 1.0e-3,
    central_c_per_gtc: TCRE_C_PER_GTC,
    high_c_per_gtc: 2.3e-3,
};

/// Ordered low/central/high warming values, °C.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarmingRange {
    pub low: f64,
    pub central: f64,
    pub high: f64,
}

/// Ordered low/central/high emissions headroom.
///
/// The ordering is inverted relative to TCRE: high TCRE gives the low budget,
/// and low TCRE gives the high budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetHeadroomRange {
    pub low: f64,
    pub central: f64,
    pub high: f64,
}

impl TcreEstimate {
    pub fn try_new(
        low_c_per_gtc: f64,
        central_c_per_gtc: f64,
        high_c_per_gtc: f64,
    ) -> Result<Self, ModelError> {
        let estimate = Self {
            low_c_per_gtc,
            central_c_per_gtc,
            high_c_per_gtc,
        };
        estimate.validate()?;
        Ok(estimate)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("low_c_per_gtc", self.low_c_per_gtc)?;
        require_positive("central_c_per_gtc", self.central_c_per_gtc)?;
        require_positive("high_c_per_gtc", self.high_c_per_gtc)?;
        if self.low_c_per_gtc > self.central_c_per_gtc {
            return Err(ModelError::InvalidOrdering {
                lower: "low_c_per_gtc",
                lower_value: self.low_c_per_gtc,
                upper: "central_c_per_gtc",
                upper_value: self.central_c_per_gtc,
            });
        }
        if self.central_c_per_gtc > self.high_c_per_gtc {
            return Err(ModelError::InvalidOrdering {
                lower: "central_c_per_gtc",
                lower_value: self.central_c_per_gtc,
                upper: "high_c_per_gtc",
                upper_value: self.high_c_per_gtc,
            });
        }
        Ok(())
    }

    pub fn warming_from_carbon(&self, gt_carbon: f64) -> Result<WarmingRange, ModelError> {
        self.validate()?;
        require_non_negative("gt_carbon", gt_carbon)?;
        Ok(WarmingRange {
            low: self.low_c_per_gtc * gt_carbon,
            central: self.central_c_per_gtc * gt_carbon,
            high: self.high_c_per_gtc * gt_carbon,
        })
    }

    pub fn warming_from_co2(&self, gt_co2: f64) -> Result<WarmingRange, ModelError> {
        require_non_negative("gt_co2", gt_co2)?;
        self.warming_from_carbon(gt_co2 / CO2_TO_C_MASS_RATIO)
    }

    pub fn carbon_headroom(
        &self,
        target_warming: f64,
        warming_so_far: f64,
    ) -> Result<BudgetHeadroomRange, ModelError> {
        self.validate()?;
        require_finite("target_warming", target_warming)?;
        require_finite("warming_so_far", warming_so_far)?;
        let headroom = (target_warming - warming_so_far).max(0.0);
        if headroom == 0.0 {
            return Ok(BudgetHeadroomRange {
                low: 0.0,
                central: 0.0,
                high: 0.0,
            });
        }
        Ok(BudgetHeadroomRange {
            low: headroom / self.high_c_per_gtc,
            central: headroom / self.central_c_per_gtc,
            high: headroom / self.low_c_per_gtc,
        })
    }

    pub fn co2_headroom(
        &self,
        target_warming: f64,
        warming_so_far: f64,
    ) -> Result<BudgetHeadroomRange, ModelError> {
        let carbon = self.carbon_headroom(target_warming, warming_so_far)?;
        Ok(BudgetHeadroomRange {
            low: carbon.low * CO2_TO_C_MASS_RATIO,
            central: carbon.central * CO2_TO_C_MASS_RATIO,
            high: carbon.high * CO2_TO_C_MASS_RATIO,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousand_gtc_gives_canonical_warming() {
        assert!((warming_from_cumulative_carbon(1000.0) - 1.65).abs() < 1e-9);
    }

    #[test]
    fn co2_and_carbon_pathways_are_consistent() {
        let via_c = warming_from_cumulative_carbon(1000.0);
        let via_co2 = warming_from_cumulative_co2(1000.0 * CO2_TO_C_MASS_RATIO);
        assert!((via_c - via_co2).abs() < 1e-9);
    }

    #[test]
    fn warming_is_linear_in_emissions() {
        let a = warming_from_cumulative_carbon(500.0);
        let b = warming_from_cumulative_carbon(1000.0);
        assert!((b / a - 2.0).abs() < 1e-9);
    }

    #[test]
    fn central_headroom_for_1p5_from_1p1() {
        let budget = central_tcre_budget_headroom_carbon(1.5, 1.1);
        assert!((budget - 242.4).abs() < 1.0, "budget={budget}");
        let co2 = central_tcre_budget_headroom_co2(1.5, 1.1);
        assert!((co2 - 888.0).abs() < 5.0, "co2={co2}");
    }

    #[test]
    fn exceeded_target_gives_zero_headroom() {
        assert_eq!(central_tcre_budget_headroom_carbon(1.5, 1.6), 0.0);
        assert_eq!(central_tcre_budget_headroom_co2(1.5, 1.6), 0.0);
    }

    #[test]
    fn explicit_tcre_range_preserves_ordering() {
        let warming = DEFAULT_TCRE_ESTIMATE.warming_from_carbon(1000.0).unwrap();
        assert!((warming.low - 1.0).abs() < 1e-12);
        assert!((warming.central - 1.65).abs() < 1e-12);
        assert!((warming.high - 2.3).abs() < 1e-12);
    }

    #[test]
    fn budget_range_inverts_tcre_ordering() {
        let budget = DEFAULT_TCRE_ESTIMATE.carbon_headroom(1.5, 1.1).unwrap();
        assert!(budget.low < budget.central && budget.central < budget.high);
        assert!((budget.central - central_tcre_budget_headroom_carbon(1.5, 1.1)).abs() < 1e-12);
    }

    #[test]
    fn invalid_tcre_ordering_is_rejected() {
        assert!(TcreEstimate::try_new(2.0e-3, 1.5e-3, 2.3e-3).is_err());
        assert!(TcreEstimate::try_new(1.0e-3, 2.5e-3, 2.3e-3).is_err());
    }
}
