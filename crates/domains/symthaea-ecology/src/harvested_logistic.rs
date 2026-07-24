// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Logistic growth under a constant harvest pressure.
//!
//! `dN/dt = rN(1-N/K) - H` provides a compact ecological tipping-point oracle:
//! the two equilibria merge at the maximum sustainable yield `rK/4` and vanish
//! under larger harvest.

use crate::error::{ModelError, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarvestedLogistic {
    pub intrinsic_growth_rate: f64,
    pub carrying_capacity: f64,
    pub harvest_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarvestRegime {
    Sustainable,
    MaximumSustainableYield,
    Overharvest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarvestEquilibriumStability {
    Stable,
    Unstable,
    Semistable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarvestEquilibrium {
    pub population: f64,
    pub stability: HarvestEquilibriumStability,
}

impl HarvestedLogistic {
    pub fn try_new(
        intrinsic_growth_rate: f64,
        carrying_capacity: f64,
        harvest_rate: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            intrinsic_growth_rate,
            carrying_capacity,
            harvest_rate,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("intrinsic_growth_rate", self.intrinsic_growth_rate)?;
        require_positive("carrying_capacity", self.carrying_capacity)?;
        require_non_negative("harvest_rate", self.harvest_rate)?;
        Ok(())
    }

    pub fn tendency(&self, population: f64) -> f64 {
        self.intrinsic_growth_rate * population * (1.0 - population / self.carrying_capacity)
            - self.harvest_rate
    }

    pub fn tendency_derivative(&self, population: f64) -> f64 {
        self.intrinsic_growth_rate * (1.0 - 2.0 * population / self.carrying_capacity)
    }

    pub fn recovery_diagnostic(
        &self,
        population: f64,
    ) -> Result<crate::recovery::RecoveryDiagnostic, ModelError> {
        self.validate()?;
        require_non_negative("population", population)?;
        crate::recovery::scalar_recovery_diagnostic(self.tendency_derivative(population))
    }

    pub fn maximum_sustainable_harvest(&self) -> f64 {
        self.intrinsic_growth_rate * self.carrying_capacity / 4.0
    }

    pub fn regime(&self) -> HarvestRegime {
        let critical = self.maximum_sustainable_harvest();
        let tolerance = 1e-12 * critical.max(1.0);
        if self.harvest_rate > critical + tolerance {
            HarvestRegime::Overharvest
        } else if (self.harvest_rate - critical).abs() <= tolerance {
            HarvestRegime::MaximumSustainableYield
        } else {
            HarvestRegime::Sustainable
        }
    }

    /// Non-negative equilibria, ordered from lower to upper population.
    pub fn equilibria(&self) -> Result<Vec<HarvestEquilibrium>, ModelError> {
        self.validate()?;
        match self.regime() {
            HarvestRegime::Overharvest => Ok(Vec::new()),
            HarvestRegime::MaximumSustainableYield => Ok(vec![HarvestEquilibrium {
                population: 0.5 * self.carrying_capacity,
                stability: HarvestEquilibriumStability::Semistable,
            }]),
            HarvestRegime::Sustainable => {
                let discriminant = 1.0
                    - 4.0 * self.harvest_rate
                        / (self.intrinsic_growth_rate * self.carrying_capacity);
                let root = discriminant.max(0.0).sqrt();
                Ok(vec![
                    HarvestEquilibrium {
                        population: 0.5 * self.carrying_capacity * (1.0 - root),
                        stability: HarvestEquilibriumStability::Unstable,
                    },
                    HarvestEquilibrium {
                        population: 0.5 * self.carrying_capacity * (1.0 + root),
                        stability: HarvestEquilibriumStability::Stable,
                    },
                ])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_harvest_recovers_extinction_and_capacity_equilibria() {
        let model = HarvestedLogistic::try_new(0.5, 100.0, 0.0).unwrap();
        let equilibria = model.equilibria().unwrap();
        assert_eq!(equilibria.len(), 2);
        assert_eq!(equilibria[0].population, 0.0);
        assert_eq!(
            equilibria[0].stability,
            HarvestEquilibriumStability::Unstable
        );
        assert_eq!(equilibria[1].population, 100.0);
        assert_eq!(equilibria[1].stability, HarvestEquilibriumStability::Stable);
    }

    #[test]
    fn maximum_sustainable_yield_is_a_tangent_equilibrium() {
        let model = HarvestedLogistic::try_new(0.5, 100.0, 12.5).unwrap();
        assert_eq!(model.regime(), HarvestRegime::MaximumSustainableYield);
        let equilibrium = model.equilibria().unwrap()[0];
        assert_eq!(equilibrium.population, 50.0);
        assert_eq!(
            equilibrium.stability,
            HarvestEquilibriumStability::Semistable
        );
        assert!(model.tendency(equilibrium.population).abs() < 1e-12);
    }

    #[test]
    fn sustainable_harvest_has_two_exact_roots() {
        let model = HarvestedLogistic::try_new(0.5, 100.0, 10.0).unwrap();
        let equilibria = model.equilibria().unwrap();
        assert_eq!(equilibria.len(), 2);
        for equilibrium in equilibria {
            assert!(model.tendency(equilibrium.population).abs() < 1e-12);
        }
    }

    #[test]
    fn overharvest_removes_non_negative_equilibria() {
        let model = HarvestedLogistic::try_new(0.5, 100.0, 13.0).unwrap();
        assert_eq!(model.regime(), HarvestRegime::Overharvest);
        assert!(model.equilibria().unwrap().is_empty());
        assert!(model.tendency(50.0) < 0.0);
    }

    #[test]
    fn recovery_time_diverges_at_maximum_sustainable_yield() {
        let ordinary = HarvestedLogistic::try_new(0.5, 100.0, 0.0).unwrap();
        let ordinary_upper = ordinary.equilibria().unwrap()[1].population;
        let ordinary_recovery = ordinary
            .recovery_diagnostic(ordinary_upper)
            .unwrap()
            .e_folding_time
            .unwrap();

        let near_fold = HarvestedLogistic::try_new(0.5, 100.0, 12.49).unwrap();
        let upper = near_fold.equilibria().unwrap()[1].population;
        let near_recovery = near_fold
            .recovery_diagnostic(upper)
            .unwrap()
            .e_folding_time
            .unwrap();
        assert!(near_recovery > ordinary_recovery);

        let critical = HarvestedLogistic::try_new(0.5, 100.0, 12.5).unwrap();
        let fold = critical.equilibria().unwrap()[0].population;
        assert_eq!(
            critical.recovery_diagnostic(fold).unwrap().stability,
            crate::recovery::LinearStability::Critical
        );
    }
}
