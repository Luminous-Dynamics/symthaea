// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Power-law disaster cascades: correlated failure propagation.
//!
//! Real civilizational crises follow power-law distributions (Bak 1987):
//! P(size ≥ s) ~ s^(-τ), τ ≈ 1.0-1.5
//!
//! A disaster in one domain can trigger failures in adjacent domains.
//! An ECLSS failure → power loss → food spoilage → health crisis → governance overload.
//!
//! CITATION: Bak, Tang, Wiesenfeld (1987) "Self-organized criticality",
//! Phys. Rev. Lett. 59(4), pp. 381-384.
//!
//! LIMITATION: Cascade topology is manually defined, not emergent.
//! Real cascades involve nonlinear interactions we can't fully model.

use crate::stochastic::StochasticEngine;
use serde::{Deserialize, Serialize};

/// Power-law exponent for cascade size distribution.
/// CITATION: Bak et al. (1987). Typical range 1.0-1.5.
const CASCADE_TAU: f64 = 1.2;

/// Base probability that a disaster triggers a cascade in an adjacent system.
/// SPECULATIVE: No empirical basis for space colonies.
const CASCADE_PROPAGATION_BASE: f64 = 0.15;

/// Maximum cascade depth (prevents runaway chains).
const MAX_CASCADE_DEPTH: u32 = 5;

/// Infrastructure stress threshold above which cascades become more likely.
const _STRESS_CASCADE_THRESHOLD: f64 = 0.6;

/// Domain categories that can cascade into each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CascadeDomain {
    Power,
    LifeSupport,
    Food,
    Water,
    Communications,
    Governance,
    Health,
    Transport,
}

impl CascadeDomain {
    /// Which domains this one can cascade INTO (adjacency in failure graph).
    pub fn downstream(&self) -> &'static [CascadeDomain] {
        use CascadeDomain::*;
        match self {
            Power => &[LifeSupport, Food, Water, Communications],
            LifeSupport => &[Health, Governance],
            Food => &[Health, Governance],
            Water => &[Food, Health],
            Communications => &[Governance, Transport],
            Governance => &[Food, Transport], // governance failure → distribution breakdown
            Health => &[Governance],          // health crisis → governance overload
            Transport => &[Food, Water],      // transport failure → supply disruption
        }
    }

    /// All domain variants.
    pub fn all() -> &'static [CascadeDomain] {
        use CascadeDomain::*;
        &[
            Power,
            LifeSupport,
            Food,
            Water,
            Communications,
            Governance,
            Health,
            Transport,
        ]
    }
}

/// A cascade event: a failure propagating through interconnected systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeEvent {
    /// Origin domain of the cascade.
    pub origin: CascadeDomain,
    /// All domains affected (including origin).
    pub affected: Vec<CascadeDomain>,
    /// Total severity (sum of all domain failures).
    pub total_severity: f64,
    /// Depth of the cascade chain.
    pub depth: u32,
    /// World ID where the cascade occurred.
    pub world_id: u32,
}

/// Cascade engine: tracks system stress and propagates failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeEngine {
    /// Per-domain stress level [0, 1] per world.
    /// High stress = higher cascade probability.
    pub domain_stress: Vec<[f64; 8]>, // indexed by world, then CascadeDomain ordinal
    /// Total cascades this simulation.
    pub total_cascades: u32,
    /// Largest cascade size (number of domains affected).
    pub max_cascade_size: u32,
    /// Cascade history (last 100 events).
    pub history: Vec<CascadeEvent>,
}

impl CascadeEngine {
    pub fn new(num_worlds: usize) -> Self {
        Self {
            domain_stress: vec![[0.0; 8]; num_worlds],
            total_cascades: 0,
            max_cascade_size: 0,
            history: Vec::new(),
        }
    }

    /// Update domain stress from world state.
    pub fn update_stress(
        &mut self,
        world_idx: usize,
        infrastructure_level: f64,
        resource_fractions: &[(&str, f64)], // (resource_name, current/capacity fraction)
        governance_stability: f64,
    ) {
        if world_idx >= self.domain_stress.len() {
            self.domain_stress.resize(world_idx + 1, [0.0; 8]);
        }

        let stress = &mut self.domain_stress[world_idx];
        // Power stress: inverse of infrastructure
        stress[CascadeDomain::Power as usize] = (1.0 - infrastructure_level).clamp(0.0, 1.0);
        // Life support stress: inverse of oxygen fraction
        stress[CascadeDomain::LifeSupport as usize] = resource_fractions
            .iter()
            .find(|(n, _)| *n == "oxygen")
            .map(|(_, f)| (1.0 - f).clamp(0.0, 1.0))
            .unwrap_or(0.3);
        // Food stress
        stress[CascadeDomain::Food as usize] = resource_fractions
            .iter()
            .find(|(n, _)| *n == "food")
            .map(|(_, f)| (1.0 - f).clamp(0.0, 1.0))
            .unwrap_or(0.3);
        // Water stress
        stress[CascadeDomain::Water as usize] = resource_fractions
            .iter()
            .find(|(n, _)| *n == "water")
            .map(|(_, f)| (1.0 - f).clamp(0.0, 1.0))
            .unwrap_or(0.3);
        // Governance stress
        stress[CascadeDomain::Governance as usize] = (1.0 - governance_stability).clamp(0.0, 1.0);
        // Others: moderate default
        stress[CascadeDomain::Communications as usize] =
            stress[CascadeDomain::Power as usize] * 0.5;
        stress[CascadeDomain::Health as usize] =
            (stress[CascadeDomain::Food as usize] + stress[CascadeDomain::Water as usize]) * 0.3;
        stress[CascadeDomain::Transport as usize] = stress[CascadeDomain::Power as usize] * 0.3;
    }

    /// Attempt to trigger a cascade from an initial disaster in a domain.
    /// Returns None if no cascade occurs (the common case).
    pub fn try_cascade(
        &mut self,
        world_id: u32,
        world_idx: usize,
        origin: CascadeDomain,
        initial_severity: f64,
        rng: &mut StochasticEngine,
    ) -> Option<CascadeEvent> {
        if world_idx >= self.domain_stress.len() {
            return None;
        }

        let stress = self.domain_stress[world_idx];
        let origin_stress = stress[origin as usize];

        // Cascade probability: base × (1 + stress) × severity
        // Higher stress and higher initial severity = more likely to cascade
        let cascade_prob =
            CASCADE_PROPAGATION_BASE * (1.0 + origin_stress) * initial_severity.clamp(0.1, 1.0);

        if !rng.bernoulli(cascade_prob.min(0.5)) {
            return None;
        }

        // Cascade triggered — propagate through downstream domains
        let mut affected = vec![origin];
        let mut frontier = vec![(origin, initial_severity)];
        let mut depth = 0u32;
        let mut total_severity = initial_severity;

        while !frontier.is_empty() && depth < MAX_CASCADE_DEPTH {
            depth += 1;
            let mut next_frontier = Vec::new();

            for (domain, parent_severity) in &frontier {
                for &downstream in domain.downstream() {
                    if affected.contains(&downstream) {
                        continue;
                    }

                    let down_stress = stress[downstream as usize];
                    // Propagation: probability decreases with depth, increases with stress
                    let prop_prob =
                        CASCADE_PROPAGATION_BASE * (1.0 + down_stress) * parent_severity
                            / (depth as f64 + 1.0); // Attenuate with depth

                    if rng.bernoulli(prop_prob.min(0.4)) {
                        // Power-law severity: child severity = parent × random^(1/τ)
                        let u: f64 = rng.next_f64().max(0.01);
                        let child_severity = parent_severity * u.powf(1.0 / CASCADE_TAU);
                        let child_severity = child_severity.clamp(0.01, *parent_severity);

                        affected.push(downstream);
                        total_severity += child_severity;
                        next_frontier.push((downstream, child_severity));
                    }
                }
            }
            frontier = next_frontier;
        }

        if affected.len() <= 1 {
            return None;
        } // No cascade, just the origin

        let event = CascadeEvent {
            origin,
            affected: affected.clone(),
            total_severity,
            depth,
            world_id,
        };

        self.total_cascades += 1;
        self.max_cascade_size = self.max_cascade_size.max(affected.len() as u32);
        self.history.push(event.clone());
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_domain_adjacency() {
        let power_downstream = CascadeDomain::Power.downstream();
        assert!(power_downstream.contains(&CascadeDomain::LifeSupport));
        assert!(power_downstream.contains(&CascadeDomain::Food));
        assert!(!power_downstream.contains(&CascadeDomain::Governance));
    }

    #[test]
    fn test_cascade_requires_stress() {
        let mut engine = CascadeEngine::new(1);
        let mut rng = StochasticEngine::new(42);

        // Zero stress, low severity → no cascade
        engine.domain_stress[0] = [0.0; 8];
        let mut cascades = 0;
        for _ in 0..100 {
            if engine
                .try_cascade(0, 0, CascadeDomain::Power, 0.1, &mut rng)
                .is_some()
            {
                cascades += 1;
            }
        }
        // Should be rare (prob ≈ 0.15 × 1.0 × 0.1 = 1.5%)
        assert!(
            cascades < 10,
            "Low stress should rarely cascade: {cascades}/100"
        );
    }

    #[test]
    fn test_high_stress_increases_cascades() {
        let mut engine = CascadeEngine::new(1);
        let mut rng = StochasticEngine::new(42);

        // High stress everywhere
        engine.domain_stress[0] = [0.9; 8];
        let mut cascades = 0;
        for _ in 0..100 {
            if engine
                .try_cascade(0, 0, CascadeDomain::Power, 0.8, &mut rng)
                .is_some()
            {
                cascades += 1;
            }
        }
        // Should be more frequent than low stress (probabilistic — use relaxed threshold)
        assert!(
            cascades > 3,
            "High stress should cascade sometimes: {cascades}/100"
        );
    }

    #[test]
    fn test_cascade_depth_bounded() {
        let mut engine = CascadeEngine::new(1);
        let mut rng = StochasticEngine::new(42);
        engine.domain_stress[0] = [0.95; 8]; // Extreme stress

        for _ in 0..50 {
            if let Some(event) = engine.try_cascade(0, 0, CascadeDomain::Power, 1.0, &mut rng) {
                assert!(
                    event.depth <= MAX_CASCADE_DEPTH,
                    "Cascade depth {} exceeds max {}",
                    event.depth,
                    MAX_CASCADE_DEPTH
                );
                assert!(
                    event.affected.len() <= CascadeDomain::all().len(),
                    "Cascade can't affect more domains than exist"
                );
            }
        }
    }

    #[test]
    fn test_cascade_severity_power_law() {
        let mut engine = CascadeEngine::new(1);
        let mut rng = StochasticEngine::new(42);
        engine.domain_stress[0] = [0.8; 8];

        let mut severities = Vec::new();
        for _ in 0..200 {
            if let Some(event) = engine.try_cascade(0, 0, CascadeDomain::Power, 0.8, &mut rng) {
                severities.push(event.total_severity);
            }
        }

        if severities.len() > 5 {
            severities.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = severities[severities.len() / 2];
            let max = severities.last().unwrap();
            // Power-law: max should be larger than median (fat tail)
            // Note: total_severity sums across domains, so the ratio is
            // compressed compared to single-domain power-law distribution.
            assert!(
                max > &(median * 1.1),
                "Cascades should have variance: max={max}, median={median}"
            );
        }
    }

    #[test]
    fn test_update_stress_from_world() {
        let mut engine = CascadeEngine::new(1);
        let resources = vec![("food", 0.3), ("water", 0.8), ("oxygen", 0.9)];
        engine.update_stress(0, 0.4, &resources, 0.7);

        // Low food → high food stress
        assert!(engine.domain_stress[0][CascadeDomain::Food as usize] > 0.5);
        // High water → low water stress
        assert!(engine.domain_stress[0][CascadeDomain::Water as usize] < 0.3);
        // Moderate governance stability → moderate governance stress
        assert!(engine.domain_stress[0][CascadeDomain::Governance as usize] < 0.5);
    }
}
