// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Safety-gated resource optimization for EVA assistance.
//!
//! The optimizer ranks only candidates that have already passed deterministic
//! safety/life-support constraints. It can never convert an unsafe candidate
//! into an executable one by assigning it a favorable score.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvaResourceWeights {
    pub electrical_energy: f64,
    pub oxygen: f64,
    pub co2: f64,
    pub thermal_load: f64,
    pub fatigue: f64,
}

impl Default for EvaResourceWeights {
    fn default() -> Self {
        Self {
            electrical_energy: 1.0,
            oxygen: 1.0,
            co2: 1.0,
            thermal_load: 1.0,
            fatigue: 1.0,
        }
    }
}

impl EvaResourceWeights {
    pub fn is_valid(&self) -> bool {
        [
            self.electrical_energy,
            self.oxygen,
            self.co2,
            self.thermal_load,
            self.fatigue,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v >= 0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaResourceCandidate {
    pub id: String,
    /// Energy consumed over the decision horizon, Wh.
    pub electrical_energy_wh: f64,
    /// O2 consumed over the horizon, standard litres.
    pub oxygen_l: f64,
    /// CO2 produced over the horizon, standard litres.
    pub co2_l: f64,
    /// Net heat added to the suit thermal system over the horizon, Wh-thermal.
    pub thermal_load_wh: f64,
    /// Dimensionless accumulated fatigue proxy, >= 0.
    pub fatigue_cost: f64,
    /// Must come from deterministic assist/survival preflight.
    pub safety_permitted: bool,
    /// PLSS still has an acceptable survival path.
    pub life_support_available: bool,
    /// Protected survival power remains satisfied.
    pub survival_power_satisfied: bool,
}

impl EvaResourceCandidate {
    pub fn is_valid(&self) -> bool {
        !self.id.trim().is_empty()
            && [
                self.electrical_energy_wh,
                self.oxygen_l,
                self.co2_l,
                self.thermal_load_wh,
                self.fatigue_cost,
            ]
            .into_iter()
            .all(|v| v.is_finite() && v >= 0.0)
    }

    pub fn eligible(&self) -> bool {
        self.is_valid()
            && self.safety_permitted
            && self.life_support_available
            && self.survival_power_satisfied
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvaNormalization {
    pub electrical_energy_wh: f64,
    pub oxygen_l: f64,
    pub co2_l: f64,
    pub thermal_load_wh: f64,
    pub fatigue_cost: f64,
}

impl EvaNormalization {
    pub fn is_valid(&self) -> bool {
        [
            self.electrical_energy_wh,
            self.oxygen_l,
            self.co2_l,
            self.thermal_load_wh,
            self.fatigue_cost,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v > 0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedEvaCandidate {
    pub id: String,
    pub score: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct EvaResourceOptimizer {
    weights: EvaResourceWeights,
    normalization: EvaNormalization,
}

impl EvaResourceOptimizer {
    pub fn new(weights: EvaResourceWeights, normalization: EvaNormalization) -> Option<Self> {
        (weights.is_valid() && normalization.is_valid()).then_some(Self {
            weights,
            normalization,
        })
    }

    pub fn rank(&self, candidates: &[EvaResourceCandidate]) -> Vec<RankedEvaCandidate> {
        let mut ranked: Vec<_> = candidates
            .iter()
            .filter(|candidate| candidate.eligible())
            .map(|candidate| RankedEvaCandidate {
                id: candidate.id.clone(),
                score: self.score(candidate),
            })
            .collect();
        ranked.sort_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)));
        ranked
    }

    pub fn best<'a>(
        &self,
        candidates: &'a [EvaResourceCandidate],
    ) -> Option<&'a EvaResourceCandidate> {
        candidates
            .iter()
            .filter(|candidate| candidate.eligible())
            .min_by(|a, b| self.score(a).total_cmp(&self.score(b)).then_with(|| a.id.cmp(&b.id)))
    }

    fn score(&self, c: &EvaResourceCandidate) -> f64 {
        self.weights.electrical_energy
            * c.electrical_energy_wh
            / self.normalization.electrical_energy_wh
            + self.weights.oxygen * c.oxygen_l / self.normalization.oxygen_l
            + self.weights.co2 * c.co2_l / self.normalization.co2_l
            + self.weights.thermal_load * c.thermal_load_wh / self.normalization.thermal_load_wh
            + self.weights.fatigue * c.fatigue_cost / self.normalization.fatigue_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optimizer() -> EvaResourceOptimizer {
        EvaResourceOptimizer::new(
            EvaResourceWeights::default(),
            EvaNormalization {
                electrical_energy_wh: 100.0,
                oxygen_l: 100.0,
                co2_l: 100.0,
                thermal_load_wh: 100.0,
                fatigue_cost: 1.0,
            },
        )
        .unwrap()
    }

    fn candidate(id: &str, energy: f64, fatigue: f64) -> EvaResourceCandidate {
        EvaResourceCandidate {
            id: id.into(),
            electrical_energy_wh: energy,
            oxygen_l: 30.0,
            co2_l: 25.0,
            thermal_load_wh: 40.0,
            fatigue_cost: fatigue,
            safety_permitted: true,
            life_support_available: true,
            survival_power_satisfied: true,
        }
    }

    #[test]
    fn optimizer_can_trade_battery_for_lower_fatigue() {
        let low_power_high_fatigue = candidate("low-power", 20.0, 0.8);
        let more_power_low_fatigue = candidate("assist", 50.0, 0.2);
        let best = optimizer()
            .best(&[low_power_high_fatigue, more_power_low_fatigue])
            .unwrap();
        assert_eq!(best.id, "assist");
    }

    #[test]
    fn unsafe_candidate_is_never_ranked() {
        let safe = candidate("safe", 80.0, 0.8);
        let mut unsafe_but_cheap = candidate("unsafe", 0.0, 0.0);
        unsafe_but_cheap.safety_permitted = false;
        let ranked = optimizer().rank(&[unsafe_but_cheap, safe]);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].id, "safe");
    }

    #[test]
    fn survival_power_is_a_hard_gate_not_a_weight() {
        let safe = candidate("safe", 100.0, 1.0);
        let mut starved = candidate("starved", 0.0, 0.0);
        starved.survival_power_satisfied = false;
        assert_eq!(optimizer().best(&[starved, safe]).unwrap().id, "safe");
    }
}
