// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cohort Manager: population as statistical groups, not individuals.
//!
//! Instead of simulating 70,000 individual agents with identical update logic,
//! group them into cohorts by (world, age_band, sector, health_band).
//! Each cohort updates as a single statistical entity per tick.
//!
//! Only "notable" agents (leaders, specialists, named characters, anyone
//! involved in a narrative event) are tracked individually.
//!
//! This is how actuarial models, demographic projections, and military
//! simulations actually work at scale. It's not an approximation — it's
//! explicitly representing what we actually know about the population.
//!
//! # Architecture
//!
//! ```text
//! CohortManager
//! ├── cohorts: HashMap<CohortKey, Cohort>   (50-200 cohorts)
//! ├── notables: Vec<NotableAgent>            (50-200 individuals)
//! └── social_graph: DiGraph<NotableIdx, Relationship>  (P6)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Key for a population cohort.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CohortKey {
    pub world_id: u32,
    pub age_band: AgeBand,
    pub primary_sector: u8, // 0-7
    pub health_band: HealthBand,
    pub sex: CohortSex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeBand {
    Child,   // 0-14
    Youth,   // 15-24
    Adult,   // 25-44
    MidAge,  // 45-64
    Elder,   // 65-84
    Ancient, // 85+
}

impl AgeBand {
    pub fn from_age_years(age: f64) -> Self {
        match age as u32 {
            0..=14 => Self::Child,
            15..=24 => Self::Youth,
            25..=44 => Self::Adult,
            45..=64 => Self::MidAge,
            65..=84 => Self::Elder,
            _ => Self::Ancient,
        }
    }

    pub fn can_work(&self) -> bool {
        !matches!(self, Self::Child)
    }

    pub fn can_reproduce(&self) -> bool {
        matches!(self, Self::Youth | Self::Adult | Self::MidAge)
    }

    pub fn midpoint_years(&self) -> f64 {
        match self {
            Self::Child => 7.0,
            Self::Youth => 20.0,
            Self::Adult => 35.0,
            Self::MidAge => 55.0,
            Self::Elder => 75.0,
            Self::Ancient => 90.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthBand {
    Excellent, // 0.8-1.0
    Good,      // 0.5-0.8
    Poor,      // 0.2-0.5
    Critical,  // 0.0-0.2
}

impl HealthBand {
    pub fn from_health(h: f64) -> Self {
        if h >= 0.8 {
            Self::Excellent
        } else if h >= 0.5 {
            Self::Good
        } else if h >= 0.2 {
            Self::Poor
        } else {
            Self::Critical
        }
    }

    pub fn midpoint(&self) -> f64 {
        match self {
            Self::Excellent => 0.9,
            Self::Good => 0.65,
            Self::Poor => 0.35,
            Self::Critical => 0.1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CohortSex {
    Male,
    Female,
}

/// A population cohort: statistical aggregate of similar agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cohort {
    pub count: u32,
    /// Mean skill level in primary sector [0, 1].
    pub mean_skill: f64,
    /// Mean allostatic load [0, 1].
    pub mean_load: f64,
    /// Mean consciousness tier [0, 1].
    pub mean_consciousness: f64,
    /// Fraction paired (have partner).
    pub paired_fraction: f64,
    /// Mean education level [0, 1].
    pub mean_education: f64,
}

/// A notable agent tracked individually (for narrative and social dynamics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotableAgent {
    pub id: u64,
    pub name: String,
    pub world_id: u32,
    pub age_years: f64,
    pub health: f64,
    pub primary_sector: u8,
    pub role: NotableRole,
    /// Allostatic load [0, 1].
    pub load: f64,
    /// Spinozist affect snapshot.
    pub conatus: f64,
    /// Connections to other notables (indices into notables vec).
    pub connections: Vec<usize>,
    /// Narrative events this agent is attached to.
    pub event_ticks: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotableRole {
    Leader,
    Medic,
    Engineer,
    Scientist,
    Teacher,
    Artist,
    Founder,
}

/// Manages the cohort population model + notable agents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CohortManager {
    pub cohorts: HashMap<CohortKey, Cohort>,
    pub notables: Vec<NotableAgent>,
    pub total_population: u32,
}

impl CohortManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build cohorts from a V1 agent list (migration path).
    pub fn from_v1_agents(
        agents: &[crate::agent::CivAgent],
        world_id: u32,
        current_tick: u32,
    ) -> Self {
        let mut manager = Self::new();
        let mut total = 0u32;

        for agent in agents.iter().filter(|a| a.is_alive()) {
            let age = agent.age_years(current_tick);
            let key = CohortKey {
                world_id,
                age_band: AgeBand::from_age_years(age),
                primary_sector: agent.skills.strongest_index() as u8,
                health_band: HealthBand::from_health(agent.health),
                sex: match agent.sex {
                    crate::agent::BiologicalSex::Male => CohortSex::Male,
                    crate::agent::BiologicalSex::Female => CohortSex::Female,
                },
            };

            let cohort = manager.cohorts.entry(key).or_insert(Cohort {
                count: 0,
                mean_skill: 0.0,
                mean_load: 0.0,
                mean_consciousness: 0.0,
                paired_fraction: 0.0,
                mean_education: 0.0,
            });

            // Running mean update
            let n = cohort.count as f64;
            let skill = agent.skills.as_slice()[agent.skills.strongest_index()];
            cohort.mean_skill = (cohort.mean_skill * n + skill) / (n + 1.0);
            cohort.mean_load = (cohort.mean_load * n + agent.needs.allostatic_load) / (n + 1.0);
            cohort.mean_consciousness =
                (cohort.mean_consciousness * n + agent.consciousness.phi()) / (n + 1.0);
            cohort.mean_education = (cohort.mean_education * n + agent.education_level) / (n + 1.0);
            if agent.partner_id.is_some() {
                cohort.paired_fraction = (cohort.paired_fraction * n + 1.0) / (n + 1.0);
            } else {
                cohort.paired_fraction = cohort.paired_fraction * n / (n + 1.0);
            }
            cohort.count += 1;
            total += 1;
        }

        manager.total_population = total;
        manager
    }

    /// Total population across all cohorts.
    pub fn population(&self) -> u32 {
        self.total_population
    }

    /// Number of distinct cohorts.
    pub fn cohort_count(&self) -> usize {
        self.cohorts.len()
    }

    /// Workers (all non-child cohorts).
    pub fn worker_count(&self) -> u32 {
        self.cohorts
            .iter()
            .filter(|(k, _)| k.age_band.can_work())
            .map(|(_, c)| c.count)
            .sum()
    }

    /// Skilled workers in a specific sector.
    pub fn sector_count(&self, sector: u8) -> u32 {
        self.cohorts
            .iter()
            .filter(|(k, _)| k.primary_sector == sector && k.age_band.can_work())
            .map(|(_, c)| c.count)
            .sum()
    }

    /// Mean allostatic load across all cohorts (weighted by count).
    pub fn mean_load(&self) -> f64 {
        let (sum, count) = self.cohorts.values().fold((0.0, 0u32), |(s, n), c| {
            (s + c.mean_load * c.count as f64, n + c.count)
        });
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

// ============================================================================
// P6: Social Graph for Notable Agents
// ============================================================================

/// Relationship type between notable agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    Partner,
    ParentChild,
    Mentor,
    Friend,
    Rival,
    Colleague,
}

/// Edge weight in the social graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub kind: RelationType,
    /// Strength of the relationship [-1, 1]. Negative = hostile.
    pub strength: f64,
    /// How long this relationship has existed (ticks).
    pub duration: u32,
}

/// Social graph connecting notable agents.
/// Uses petgraph for graph operations (already a dependency).
#[derive(Debug, Clone)]
pub struct SocialGraph {
    pub graph: petgraph::graph::UnGraph<usize, Relationship>,
    /// Map from notable index → graph node index.
    pub node_map: HashMap<usize, petgraph::graph::NodeIndex>,
}

impl Default for SocialGraph {
    fn default() -> Self {
        Self {
            graph: petgraph::graph::UnGraph::new_undirected(),
            node_map: HashMap::new(),
        }
    }
}

impl SocialGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a notable agent to the graph.
    pub fn add_notable(&mut self, notable_idx: usize) {
        let node = self.graph.add_node(notable_idx);
        self.node_map.insert(notable_idx, node);
    }

    /// Connect two notables.
    pub fn connect(&mut self, a: usize, b: usize, rel: Relationship) {
        if let (Some(&na), Some(&nb)) = (self.node_map.get(&a), self.node_map.get(&b)) {
            self.graph.add_edge(na, nb, rel);
        }
    }

    /// Get all connections for a notable.
    pub fn connections(&self, notable_idx: usize) -> Vec<(usize, &Relationship)> {
        use petgraph::visit::EdgeRef;
        if let Some(&node) = self.node_map.get(&notable_idx) {
            self.graph
                .edges(node)
                .map(|e| {
                    let other = if e.source() == node {
                        self.graph[e.target()]
                    } else {
                        self.graph[e.source()]
                    };
                    (other, e.weight())
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Propagate grief when a notable dies.
    /// Returns list of (notable_idx, grief_intensity) for affected notables.
    pub fn propagate_grief(&self, dead_idx: usize) -> Vec<(usize, f64)> {
        self.connections(dead_idx)
            .iter()
            .map(|(idx, rel)| {
                let intensity = match rel.kind {
                    RelationType::Partner => 0.9,
                    RelationType::ParentChild => 0.8,
                    RelationType::Mentor => 0.5,
                    RelationType::Friend => 0.4,
                    RelationType::Colleague => 0.2,
                    RelationType::Rival => 0.1, // Even rivals grieve
                };
                (*idx, intensity * rel.strength.abs())
            })
            .collect()
    }

    /// Number of connections.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_age_band_classification() {
        assert_eq!(AgeBand::from_age_years(5.0), AgeBand::Child);
        assert_eq!(AgeBand::from_age_years(20.0), AgeBand::Youth);
        assert_eq!(AgeBand::from_age_years(35.0), AgeBand::Adult);
        assert_eq!(AgeBand::from_age_years(55.0), AgeBand::MidAge);
        assert_eq!(AgeBand::from_age_years(75.0), AgeBand::Elder);
        assert_eq!(AgeBand::from_age_years(90.0), AgeBand::Ancient);
    }

    #[test]
    fn test_health_band() {
        assert_eq!(HealthBand::from_health(0.95), HealthBand::Excellent);
        assert_eq!(HealthBand::from_health(0.6), HealthBand::Good);
        assert_eq!(HealthBand::from_health(0.3), HealthBand::Poor);
        assert_eq!(HealthBand::from_health(0.1), HealthBand::Critical);
    }

    #[test]
    fn test_social_graph_connections() {
        let mut graph = SocialGraph::new();
        graph.add_notable(0);
        graph.add_notable(1);
        graph.add_notable(2);

        graph.connect(
            0,
            1,
            Relationship {
                kind: RelationType::Partner,
                strength: 0.9,
                duration: 120,
            },
        );
        graph.connect(
            0,
            2,
            Relationship {
                kind: RelationType::Friend,
                strength: 0.6,
                duration: 60,
            },
        );

        assert_eq!(graph.edge_count(), 2);
        let conns = graph.connections(0);
        assert_eq!(conns.len(), 2);
    }

    #[test]
    fn test_grief_propagation() {
        let mut graph = SocialGraph::new();
        graph.add_notable(0); // The one who dies
        graph.add_notable(1); // Partner
        graph.add_notable(2); // Friend

        graph.connect(
            0,
            1,
            Relationship {
                kind: RelationType::Partner,
                strength: 0.95,
                duration: 240,
            },
        );
        graph.connect(
            0,
            2,
            Relationship {
                kind: RelationType::Friend,
                strength: 0.7,
                duration: 60,
            },
        );

        let grief = graph.propagate_grief(0);
        assert_eq!(grief.len(), 2);
        // Partner should grieve more than friend
        let partner_grief = grief.iter().find(|(idx, _)| *idx == 1).unwrap().1;
        let friend_grief = grief.iter().find(|(idx, _)| *idx == 2).unwrap().1;
        assert!(partner_grief > friend_grief);
    }
}
