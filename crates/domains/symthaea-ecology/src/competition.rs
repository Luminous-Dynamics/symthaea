// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The two-species competitive Lotka-Volterra model.

use crate::error::{ModelError, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Competition {
    pub k1: f64,
    pub k2: f64,
    pub a12: f64,
    pub a21: f64,
}

/// Qualitative outcome from invasion analysis of the boundary equilibria.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompetitionOutcome {
    StableCoexistence,
    Species1Excludes2,
    Species2Excludes1,
    BistableExclusion,
    DegenerateBoundary,
    InvalidParameters,
}

/// Competitive Lotka-Volterra dynamics with explicit intrinsic growth rates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompetitionDynamics {
    pub competition: Competition,
    pub growth_rate_1: f64,
    pub growth_rate_2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompetitionStability {
    pub trace: f64,
    pub determinant: f64,
    pub locally_stable: bool,
}

impl Competition {
    pub fn try_new(k1: f64, k2: f64, a12: f64, a21: f64) -> Result<Self, ModelError> {
        let model = Self { k1, k2, a12, a21 };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("k1", self.k1)?;
        require_positive("k2", self.k2)?;
        require_non_negative("a12", self.a12)?;
        require_non_negative("a21", self.a21)?;
        Ok(())
    }

    pub fn coexistence_equilibrium(&self) -> Option<(f64, f64)> {
        if self.validate().is_err() {
            return None;
        }
        let denom = 1.0 - self.a12 * self.a21;
        if denom.abs() < 1e-12 {
            return None;
        }
        let n1 = (self.k1 - self.a12 * self.k2) / denom;
        let n2 = (self.k2 - self.a21 * self.k1) / denom;
        (n1 > 0.0 && n2 > 0.0).then_some((n1, n2))
    }

    /// Classify the long-run phase portrait from mutual invasion conditions.
    pub fn outcome(&self) -> CompetitionOutcome {
        if self.validate().is_err() {
            return CompetitionOutcome::InvalidParameters;
        }

        let invasion1 = self.k1 - self.a12 * self.k2;
        let invasion2 = self.k2 - self.a21 * self.k1;
        let scale = self.k1.max(self.k2).max(1.0);
        let tolerance = 1e-12 * scale;

        if invasion1.abs() <= tolerance || invasion2.abs() <= tolerance {
            CompetitionOutcome::DegenerateBoundary
        } else if invasion1 > 0.0 && invasion2 > 0.0 {
            CompetitionOutcome::StableCoexistence
        } else if invasion1 > 0.0 {
            CompetitionOutcome::Species1Excludes2
        } else if invasion2 > 0.0 {
            CompetitionOutcome::Species2Excludes1
        } else {
            CompetitionOutcome::BistableExclusion
        }
    }

    pub fn stable_coexistence(&self) -> bool {
        self.outcome() == CompetitionOutcome::StableCoexistence
    }
}

impl CompetitionDynamics {
    pub fn try_new(
        competition: Competition,
        growth_rate_1: f64,
        growth_rate_2: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            competition,
            growth_rate_1,
            growth_rate_2,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.competition.validate()?;
        require_positive("growth_rate_1", self.growth_rate_1)?;
        require_positive("growth_rate_2", self.growth_rate_2)?;
        Ok(())
    }

    pub fn derivatives(&self, population_1: f64, population_2: f64) -> (f64, f64) {
        let pressure_1 = population_1 + self.competition.a12 * population_2;
        let pressure_2 = population_2 + self.competition.a21 * population_1;
        (
            self.growth_rate_1 * population_1 * (1.0 - pressure_1 / self.competition.k1),
            self.growth_rate_2 * population_2 * (1.0 - pressure_2 / self.competition.k2),
        )
    }

    pub fn jacobian(&self, population_1: f64, population_2: f64) -> [[f64; 2]; 2] {
        [
            [
                self.growth_rate_1
                    * (1.0
                        - (2.0 * population_1 + self.competition.a12 * population_2)
                            / self.competition.k1),
                -self.growth_rate_1 * self.competition.a12 * population_1 / self.competition.k1,
            ],
            [
                -self.growth_rate_2 * self.competition.a21 * population_2 / self.competition.k2,
                self.growth_rate_2
                    * (1.0
                        - (2.0 * population_2 + self.competition.a21 * population_1)
                            / self.competition.k2),
            ],
        ]
    }

    pub fn coexistence_stability(&self) -> Option<CompetitionStability> {
        self.validate().ok()?;
        let (population_1, population_2) = self.competition.coexistence_equilibrium()?;
        let jacobian = self.jacobian(population_1, population_2);
        let trace = jacobian[0][0] + jacobian[1][1];
        let determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
        Some(CompetitionStability {
            trace,
            determinant,
            locally_stable: trace < 0.0 && determinant > 0.0,
        })
    }

    /// Guarded RK4 trajectory including the initial state.
    pub fn try_simulate_timestamped(
        &self,
        initial_population_1: f64,
        initial_population_2: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<crate::integration::PopulationPairSample>, ModelError> {
        self.validate()?;
        require_positive("initial_population_1", initial_population_1)?;
        require_positive("initial_population_2", initial_population_2)?;
        require_positive("dt", dt)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        crate::integration::simulate_positive_pair(
            initial_population_1,
            initial_population_2,
            dt,
            steps,
            |population_1, population_2| self.derivatives(population_1, population_2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_weak_competition_coexists() {
        let c = Competition::try_new(100.0, 100.0, 0.5, 0.5).unwrap();
        let (n1, n2) = c.coexistence_equilibrium().unwrap();
        assert!((n1 - 200.0 / 3.0).abs() < 1e-9);
        assert!((n2 - 200.0 / 3.0).abs() < 1e-9);
        assert_eq!(c.outcome(), CompetitionOutcome::StableCoexistence);
    }

    #[test]
    fn strong_competition_is_bistable() {
        let c = Competition::try_new(100.0, 100.0, 1.5, 1.5).unwrap();
        assert!(c.coexistence_equilibrium().is_some());
        assert_eq!(c.outcome(), CompetitionOutcome::BistableExclusion);
    }

    #[test]
    fn asymmetric_competition_identifies_winner() {
        let species1 = Competition::try_new(100.0, 100.0, 0.5, 1.5).unwrap();
        assert_eq!(species1.outcome(), CompetitionOutcome::Species1Excludes2);

        let species2 = Competition::try_new(100.0, 100.0, 1.5, 0.5).unwrap();
        assert_eq!(species2.outcome(), CompetitionOutcome::Species2Excludes1);
    }
    fn dynamics(competition: Competition) -> CompetitionDynamics {
        CompetitionDynamics::try_new(competition, 1.0, 1.0).unwrap()
    }

    #[test]
    fn stable_coexistence_is_a_dynamic_attractor() {
        let competition = Competition::try_new(100.0, 100.0, 0.5, 0.5).unwrap();
        let model = dynamics(competition);
        let equilibrium = competition.coexistence_equilibrium().unwrap();
        let stability = model.coexistence_stability().unwrap();
        assert!(stability.trace < 0.0);
        assert!(stability.determinant > 0.0);
        assert!(stability.locally_stable);

        let trajectory = model
            .try_simulate_timestamped(10.0, 20.0, 0.01, 5_000)
            .unwrap();
        let final_state = trajectory.last().unwrap();
        assert!((final_state.first - equilibrium.0).abs() < 1e-5);
        assert!((final_state.second - equilibrium.1).abs() < 1e-5);
    }

    #[test]
    fn exclusion_outcome_matches_dynamic_winner() {
        let competition = Competition::try_new(100.0, 100.0, 0.5, 1.5).unwrap();
        let trajectory = dynamics(competition)
            .try_simulate_timestamped(10.0, 10.0, 0.01, 5_000)
            .unwrap();
        let final_state = trajectory.last().unwrap();
        assert!(final_state.first > 99.0);
        assert!(final_state.second < 1e-6);
    }

    #[test]
    fn bistable_competition_depends_on_initial_abundance() {
        let competition = Competition::try_new(100.0, 100.0, 1.5, 1.5).unwrap();
        let model = dynamics(competition);
        let first_favored = model
            .try_simulate_timestamped(80.0, 20.0, 0.01, 5_000)
            .unwrap();
        let second_favored = model
            .try_simulate_timestamped(20.0, 80.0, 0.01, 5_000)
            .unwrap();
        let first_final = first_favored.last().unwrap();
        let second_final = second_favored.last().unwrap();
        assert!(first_final.first > 99.0 && first_final.second < 1e-6);
        assert!(second_final.second > 99.0 && second_final.first < 1e-6);
        assert!(!model.coexistence_stability().unwrap().locally_stable);
    }
}
