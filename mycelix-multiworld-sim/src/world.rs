// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! World model: resources, culture, infrastructure, and population management.

use crate::agent::{CivAgent, LifeStage};
use crate::economy::WorldEconomy;
use crate::harmony::HarmonyTracker;
use crate::knowledge::WorldKnowledge;
use crate::population::SirEpidemic;
use crate::stochastic::StochasticEngine;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single resource stock within a world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStock {
    /// Current amount available.
    pub current: f64,
    /// Maximum storage capacity.
    pub capacity: f64,
    /// Production rate per tick.
    pub production_rate: f64,
    /// Consumption rate per tick (scales with population).
    pub consumption_rate: f64,
    /// Below this fraction of capacity, the resource is critical.
    pub critical_threshold: f64,
}

impl ResourceStock {
    /// Whether this resource is below its critical threshold.
    pub fn is_critical(&self) -> bool {
        self.capacity > 0.0 && (self.current / self.capacity) < self.critical_threshold
    }
}

/// Resource inventory for a world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldResources {
    stocks: HashMap<String, ResourceStock>,
}

impl WorldResources {
    /// Create empty resources.
    pub fn new() -> Self {
        Self {
            stocks: HashMap::new(),
        }
    }

    /// Get a resource stock by name.
    pub fn get(&self, name: &str) -> Option<&ResourceStock> {
        self.stocks.get(name)
    }

    /// Get mutable reference.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ResourceStock> {
        self.stocks.get_mut(name)
    }

    /// Insert or update a resource.
    pub fn set(&mut self, name: impl Into<String>, stock: ResourceStock) {
        self.stocks.insert(name.into(), stock);
    }

    /// Whether any resource is at critical level.
    pub fn any_critical(&self) -> bool {
        self.stocks.values().any(|s| s.is_critical())
    }

    /// Overall self-sufficiency ratio: average of (production / consumption) across
    /// all resources, clamped to [0, 1].
    pub fn self_sufficiency(&self) -> f64 {
        if self.stocks.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .stocks
            .values()
            .map(|s| {
                if s.consumption_rate > 0.0 {
                    (s.production_rate / s.consumption_rate).min(1.0)
                } else {
                    1.0
                }
            })
            .sum();
        sum / self.stocks.len() as f64
    }

    /// Resource names.
    pub fn resource_names(&self) -> Vec<&str> {
        self.stocks.keys().map(|s| s.as_str()).collect()
    }

    /// Default resource set for a lunar colony.
    pub fn lunar_default() -> Self {
        let mut r = Self::new();
        r.set(
            "food",
            ResourceStock {
                current: 1000.0,
                capacity: 2000.0,
                production_rate: 45.0,   // Hydroponics + imported supplements
                consumption_rate: 40.0,
                critical_threshold: 0.1,
            },
        );
        r.set(
            "water",
            ResourceStock {
                current: 800.0,
                capacity: 3000.0,
                production_rate: 50.0,
                consumption_rate: 45.0,
                critical_threshold: 0.15,
            },
        );
        r.set(
            "energy",
            ResourceStock {
                current: 1000.0,
                capacity: 5000.0,
                production_rate: 100.0,
                consumption_rate: 80.0,
                critical_threshold: 0.1,
            },
        );
        r.set(
            "materials",
            ResourceStock {
                current: 300.0,
                capacity: 1000.0,
                production_rate: 10.0,
                consumption_rate: 15.0,
                critical_threshold: 0.2,
            },
        );
        r.set(
            "oxygen",
            ResourceStock {
                current: 900.0,
                capacity: 2000.0,
                production_rate: 60.0,
                consumption_rate: 50.0,
                critical_threshold: 0.2,
            },
        );
        r
    }

    /// Default resource set for Earth (abundant).
    ///
    /// Note: consumption_rate is per-100-people in tick_economy, so these rates
    /// must be scaled to support the Earth population (~10,000 initial).
    /// Production must exceed consumption at initial population for sustainability.
    pub fn earth_default() -> Self {
        let mut r = Self::new();
        for name in &["food", "water", "energy", "materials", "oxygen"] {
            r.set(
                *name,
                ResourceStock {
                    current: 1_000_000.0,
                    capacity: 1_000_000.0,
                    // Production and consumption rates tuned for ~10,000 population.
                    // In tick_economy: actual_production = rate * infrastructure (0.9)
                    //                  actual_consumption = rate * (pop/100)
                    // For pop=10,000: consumption = 100 * rate
                    // Production 15,000 * 0.9 = 13,500 > 100 * 100 = 10,000
                    production_rate: 15_000.0,
                    consumption_rate: 100.0,
                    critical_threshold: 0.05,
                },
            );
        }
        r
    }
}

impl Default for WorldResources {
    fn default() -> Self {
        Self::new()
    }
}

/// Cultural profile influencing governance, reproduction, and social dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalProfile {
    /// Emphasis weights for each of the Eight Harmonies. Should sum to ~1.0.
    pub harmony_weights: [f64; 8],
    /// Individualism vs collectivism [0, 1].
    pub individualism: f64,
    /// Willingness to take risks [0, 1].
    pub risk_tolerance: f64,
    /// Openness to outsiders / immigrants [0, 1].
    pub xenophilia: f64,
    /// Attachment to tradition vs innovation [0, 1].
    pub traditionalism: f64,
}

impl CulturalProfile {
    /// Brownian cultural drift: magnitude proportional to 1/sqrt(pop) * (1 - contact_frequency).
    /// Small populations drift faster; high inter-group contact slows drift.
    pub fn drift(
        &mut self,
        rng: &mut StochasticEngine,
        population: usize,
        contact_frequency: f64,
    ) {
        let magnitude = 0.01 / (population as f64).sqrt().max(1.0) * (1.0 - contact_frequency);

        for w in &mut self.harmony_weights {
            *w += rng.next_gaussian(0.0, magnitude);
            *w = w.max(0.01);
        }
        // Renormalize
        let sum: f64 = self.harmony_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.harmony_weights {
                *w /= sum;
            }
        }

        self.individualism =
            (self.individualism + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
        self.risk_tolerance =
            (self.risk_tolerance + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
        self.xenophilia =
            (self.xenophilia + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
        self.traditionalism =
            (self.traditionalism + rng.next_gaussian(0.0, magnitude)).clamp(0.0, 1.0);
    }

    /// Cosine distance on harmony_weights between two cultures.
    pub fn cultural_distance(&self, other: &CulturalProfile) -> f64 {
        let dot: f64 = self
            .harmony_weights
            .iter()
            .zip(other.harmony_weights.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f64 = self.harmony_weights.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = other
            .harmony_weights
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let denom = mag_a * mag_b;
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }

    /// Earth default: balanced harmonies, moderate traits.
    pub fn earth_default() -> Self {
        Self {
            harmony_weights: [0.125; 8],
            individualism: 0.5,
            risk_tolerance: 0.4,
            xenophilia: 0.5,
            traditionalism: 0.5,
        }
    }

    /// Pioneer default: higher risk tolerance, lower traditionalism.
    pub fn pioneer_default() -> Self {
        Self {
            harmony_weights: [0.125; 8],
            individualism: 0.4,
            risk_tolerance: 0.7,
            xenophilia: 0.6,
            traditionalism: 0.3,
        }
    }
}

/// A world (colony, settlement, planet) in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct World {
    pub id: u32,
    pub name: String,
    /// Location label: "Earth", "Moon", "Mars", "Europa", etc.
    pub location: String,
    pub founded_tick: u32,
    pub parent_world_id: Option<u32>,
    pub agents: Vec<CivAgent>,
    pub next_agent_id: u64,
    pub resources: WorldResources,
    pub culture: CulturalProfile,
    /// Infrastructure development level [0, 1].
    pub infrastructure_level: f64,
    /// Carrying capacity (max sustainable population).
    pub max_population: usize,
    /// Habitable area in square meters.
    pub habitable_area_m2: f64,
    /// Founding emphasis on each of the Eight Harmonies.
    pub founding_harmony_emphasis: [f64; 8],
    /// Active epidemics.
    pub epidemics: Vec<SirEpidemic>,
    /// Knowledge and technology tracking.
    pub knowledge: WorldKnowledge,
    /// 8-sector Cobb-Douglas economy.
    pub economy: WorldEconomy,
    /// Eight Harmonies tracker.
    pub harmony: HarmonyTracker,
}

impl World {
    /// Count of living agents.
    pub fn population(&self) -> usize {
        self.agents.iter().filter(|a| a.is_alive()).count()
    }

    /// Count of living adults.
    pub fn adults(&self, tick: u32) -> usize {
        self.agents
            .iter()
            .filter(|a| {
                a.is_alive()
                    && matches!(
                        a.life_stage(tick),
                        LifeStage::Adult | LifeStage::Elder
                    )
            })
            .count()
    }

    /// Count of living children.
    pub fn children(&self, tick: u32) -> usize {
        self.agents
            .iter()
            .filter(|a| a.is_alive() && a.life_stage(tick) == LifeStage::Child)
            .count()
    }

    /// Mean health of living agents.
    pub fn mean_health(&self) -> f64 {
        let living: Vec<f64> = self
            .agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| a.health)
            .collect();
        if living.is_empty() {
            return 0.0;
        }
        living.iter().sum::<f64>() / living.len() as f64
    }

    /// Mean consciousness phi of living agents.
    pub fn mean_phi(&self) -> f64 {
        let phis: Vec<f64> = self
            .agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| a.consciousness.phi())
            .collect();
        if phis.is_empty() {
            return 0.0;
        }
        phis.iter().sum::<f64>() / phis.len() as f64
    }

    /// Fraction of living agents at each consciousness tier (0-4).
    pub fn tier_distribution(&self) -> [f64; 5] {
        let mut counts = [0usize; 5];
        let mut total = 0usize;
        for a in self.agents.iter().filter(|a| a.is_alive()) {
            counts[a.consciousness.tier() as usize] += 1;
            total += 1;
        }
        if total == 0 {
            return [0.0; 5];
        }
        let mut dist = [0.0; 5];
        for i in 0..5 {
            dist[i] = counts[i] as f64 / total as f64;
        }
        dist
    }

    /// Add an agent to this world.
    pub fn add_agent(&mut self, agent: CivAgent) {
        self.agents.push(agent);
    }

    /// Create a founding population snapshot string.
    pub fn founding_population_snapshot(&self) -> String {
        let living = self.population();
        let males = self
            .agents
            .iter()
            .filter(|a| a.is_alive() && a.sex == crate::agent::BiologicalSex::Male)
            .count();
        let females = living - males;
        format!(
            "{}: {} founders ({} M, {} F), location: {}",
            self.name, living, males, females, self.location
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, ConsciousnessState, SkillVector};
    use crate::stochastic::StochasticEngine;

    fn make_world_with_agents(n: usize) -> World {
        let mut world = World {
            id: 0,
            name: "TestWorld".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 1_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: WorldKnowledge::new(),
            economy: WorldEconomy::new(),
            harmony: HarmonyTracker::new(),
        };

        for i in 0..n {
            let agent = CivAgent {
                id: i as u64,
                birth_tick: 0,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id: 0,
                health: 0.9,
                skills: SkillVector::new(),
                education_level: 0.0,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: crate::needs::PsychologicalNeeds::new(),
            tend_balance: 0.0,
            };
            world.agents.push(agent);
        }
        world.next_agent_id = n as u64;
        world
    }

    #[test]
    fn test_world_population_counts_alive_only() {
        let mut world = make_world_with_agents(10);
        assert_eq!(world.population(), 10);
        world.agents[0].death_tick = Some(5);
        assert_eq!(world.population(), 9);
    }

    #[test]
    fn test_resource_self_sufficiency() {
        let r = WorldResources::earth_default();
        let ss = r.self_sufficiency();
        assert!(ss > 0.9, "Earth should be near self-sufficient, was {ss}");

        let lunar = WorldResources::lunar_default();
        let ss_lunar = lunar.self_sufficiency();
        assert!(
            ss_lunar < ss,
            "Lunar should be less self-sufficient than Earth"
        );
    }

    #[test]
    fn test_cultural_distance_is_symmetric() {
        let a = CulturalProfile::earth_default();
        let b = CulturalProfile::pioneer_default();
        let d_ab = a.cultural_distance(&b);
        let d_ba = b.cultural_distance(&a);
        assert!(
            (d_ab - d_ba).abs() < 1e-10,
            "Distance should be symmetric: {d_ab} vs {d_ba}"
        );
    }

    #[test]
    fn test_cultural_drift_rate_scales_with_population() {
        let mut small = CulturalProfile::earth_default();
        let mut large = CulturalProfile::earth_default();
        let reference = CulturalProfile::earth_default();

        let mut rng1 = StochasticEngine::new(42);

        // Small population drifts more
        for _ in 0..100 {
            small.drift(&mut rng1, 10, 0.0);
        }
        // Reset RNG for fair comparison
        let mut rng2 = StochasticEngine::new(42);
        for _ in 0..100 {
            large.drift(&mut rng2, 10_000, 0.0);
        }

        let d_small = reference.cultural_distance(&small);
        let d_large = reference.cultural_distance(&large);
        assert!(
            d_small > d_large,
            "Small pop should drift more: {d_small} vs {d_large}"
        );
    }

    #[test]
    fn test_resource_critical_check() {
        let mut r = WorldResources::new();
        r.set(
            "food",
            ResourceStock {
                current: 5.0,
                capacity: 100.0,
                production_rate: 1.0,
                consumption_rate: 2.0,
                critical_threshold: 0.1,
            },
        );
        assert!(r.get("food").unwrap().is_critical());

        r.set(
            "water",
            ResourceStock {
                current: 80.0,
                capacity: 100.0,
                production_rate: 5.0,
                consumption_rate: 4.0,
                critical_threshold: 0.1,
            },
        );
        assert!(!r.get("water").unwrap().is_critical());
    }
}
