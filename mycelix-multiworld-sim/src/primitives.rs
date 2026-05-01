// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Six irreducible primitives for civilizational simulation.
//!
//! These are physical/mathematical constraints that cannot be derived from
//! existing modules. They must be modeled explicitly.
//!
//! 1. Network topology — social structure determines diffusion rates
//! 2. Resource depletion — Hubbert curves, nonlinear exhaustion
//! 3. Institutional path dependence — switching costs, lock-in
//! 4. Ecosystem services — biodiversity stock, tipping points
//! 5. Trust asymmetry — destruction faster than creation
//! 6. Knowledge network effects — combinatorial adjacent possible

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE 1: Network Topology
// ═══════════════════════════════════════════════════════════════════════════

/// Social network metrics for a population (cohort-compatible).
/// Determines diffusion rates for disease, innovation, ideology, and trust.
///
/// CITATION: Watts & Strogatz (1998) "Collective dynamics of small-world networks"
/// Barabási & Albert (1999) "Emergence of scaling in random networks"
/// Dunbar (1992) "Neocortex size as a constraint on group size"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    /// Mean connections per agent (Dunbar's number ≈ 150 for humans).
    pub mean_degree: f64,
    /// Clustering coefficient [0, 1]. High = tight communities.
    /// CITATION: Watts & Strogatz (1998). Real social networks: ~0.3-0.6.
    pub clustering: f64,
    /// Average path length (degrees of separation). Small-world: ~6.
    /// CITATION: Milgram (1967). Scales as ln(N)/ln(k).
    pub avg_path_length: f64,
    /// Fraction of "hub" nodes with >5× mean connections (scale-free property).
    pub hub_fraction: f64,
    /// Network fragmentation [0, 1]. 0 = fully connected, 1 = isolated groups.
    pub fragmentation: f64,
}

impl NetworkTopology {
    /// Compute network metrics from population size and urbanization.
    /// Urban populations have denser, more connected networks.
    pub fn from_population(pop_millions: f64, urbanization: f64) -> Self {
        let pop = (pop_millions * 1_000_000.0).max(1.0);
        // Mean degree scales with urbanization: rural ~50, urban ~150 (Dunbar)
        let mean_degree = 50.0 + 100.0 * urbanization;
        // Clustering decreases with population (more diverse connections)
        let clustering = (0.6 - 0.2 * (pop.ln() / 20.0).min(1.0)).max(0.1);
        // Average path length: small-world ln(N)/ln(k)
        let avg_path = if mean_degree > 1.0 {
            pop.ln() / mean_degree.ln()
        } else {
            pop
        };
        // Hub fraction: scale-free networks have ~1-5% hubs
        let hub_fraction = 0.02 + 0.03 * urbanization;
        // Fragmentation: higher in rural, lower in urban
        let fragmentation = (0.4 - 0.3 * urbanization).max(0.05);

        Self {
            mean_degree,
            clustering,
            avg_path_length: avg_path,
            hub_fraction,
            fragmentation,
        }
    }

    /// Innovation diffusion rate: how fast new ideas spread.
    /// CITATION: Rogers (2003) "Diffusion of Innovations"
    pub fn innovation_diffusion_rate(&self) -> f64 {
        // Higher degree + lower path length + lower fragmentation = faster spread
        let connectivity = self.mean_degree / 150.0; // normalized to Dunbar
        let reach = 6.0 / self.avg_path_length.max(1.0); // normalized to small-world
        (connectivity * reach * (1.0 - self.fragmentation)).clamp(0.0, 2.0)
    }

    /// Disease transmission multiplier.
    pub fn disease_transmission_mult(&self) -> f64 {
        // Dense networks with high clustering spread disease faster
        (self.mean_degree / 100.0 * (1.0 + self.clustering)).clamp(0.5, 3.0)
    }

    /// Ideology spread rate (faction recruitment, religious conversion).
    pub fn ideology_spread_rate(&self) -> f64 {
        // Cluster-dependent: tight communities resist outside ideology
        let openness = 1.0 - self.clustering * 0.5;
        (self.mean_degree / 150.0 * openness).clamp(0.1, 1.5)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE 2: Resource Depletion Curves
// ═══════════════════════════════════════════════════════════════════════════

/// Non-renewable resource with Hubbert-style depletion curve.
///
/// CITATION: Hubbert (1956) "Nuclear energy and the fossil fuels"
/// Production follows a logistic derivative: P(t) = P_max × sech²((t-t_peak)/τ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepletableResource {
    pub name: String,
    /// Total recoverable reserve (original amount).
    pub total_reserve: f64,
    /// Amount extracted so far.
    pub cumulative_extracted: f64,
    /// Peak extraction rate (units per tick at peak).
    pub peak_rate: f64,
    /// Time constant (ticks from 25% to 75% depletion).
    pub depletion_tau: f64,
    /// Current extraction rate (computed from depletion curve).
    pub current_rate: f64,
    /// EROI degradation: as resource depletes, EROI decreases.
    /// CITATION: Hall (2014). Late-stage oil: EROI drops from 30:1 to 5:1.
    pub eroi_at_full: f64,
    pub eroi_at_half: f64,
    pub eroi_at_depleted: f64,
}

impl DepletableResource {
    /// Compute current extraction rate and EROI from depletion fraction.
    pub fn tick(&mut self) {
        let fraction_remaining = 1.0 - (self.cumulative_extracted / self.total_reserve).min(1.0);

        // Hubbert curve: peaks at 50% depletion, declines symmetrically
        // P(f) = 4 × P_max × f × (1-f) where f = fraction_extracted
        let f = 1.0 - fraction_remaining;
        self.current_rate = 4.0 * self.peak_rate * f.max(0.01) * fraction_remaining.max(0.01);

        // Extract
        self.cumulative_extracted += self.current_rate;
        self.cumulative_extracted = self.cumulative_extracted.min(self.total_reserve);
    }

    /// Current EROI based on depletion (degrades as easy reserves are exhausted).
    pub fn current_eroi(&self) -> f64 {
        let fraction_remaining = 1.0 - (self.cumulative_extracted / self.total_reserve).min(1.0);
        if fraction_remaining > 0.5 {
            // Pre-peak: EROI between full and half
            let t = (fraction_remaining - 0.5) / 0.5;
            self.eroi_at_half + t * (self.eroi_at_full - self.eroi_at_half)
        } else {
            // Post-peak: EROI between half and depleted
            let t = fraction_remaining / 0.5;
            self.eroi_at_depleted + t * (self.eroi_at_half - self.eroi_at_depleted)
        }
    }

    pub fn fraction_remaining(&self) -> f64 {
        1.0 - (self.cumulative_extracted / self.total_reserve).min(1.0)
    }
}

/// Standard depletable resources for Earth.
pub fn earth_resources() -> Vec<DepletableResource> {
    vec![
        DepletableResource {
            name: "Oil".into(),
            total_reserve: 4000.0, // billion barrels (conventional + unconventional + undiscovered)
            cumulative_extracted: 400.0, // ~400Gb extracted by 1970 (10% of total)
            peak_rate: 3.0,        // ~36Gb/year at peak ÷ 12 months
            depletion_tau: 900.0,  // broad peak
            current_rate: 0.0,
            eroi_at_full: 30.0,
            eroi_at_half: 12.0,
            eroi_at_depleted: 4.0,
        },
        DepletableResource {
            name: "Natural Gas".into(),
            total_reserve: 7000.0, // trillion cubic feet equivalent
            cumulative_extracted: 1000.0,
            peak_rate: 8.0,
            depletion_tau: 720.0, // 60 years
            current_rate: 0.0,
            eroi_at_full: 25.0,
            eroi_at_half: 12.0,
            eroi_at_depleted: 4.0,
        },
        DepletableResource {
            name: "Phosphorus".into(), // critical for agriculture
            total_reserve: 70000.0,    // million tonnes (USGS)
            cumulative_extracted: 5000.0,
            peak_rate: 20.0,
            depletion_tau: 1200.0, // 100 years
            current_rate: 0.0,
            eroi_at_full: 50.0, // not energy, but extraction effort
            eroi_at_half: 20.0,
            eroi_at_depleted: 5.0,
        },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE 3: Institutional Path Dependence
// ═══════════════════════════════════════════════════════════════════════════

/// Institutional inertia: the cost of changing established systems.
///
/// CITATION: North (1990) "Institutions, Institutional Change and Economic Performance"
/// Arthur (1994) "Increasing Returns and Path Dependence in the Economy"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalInertia {
    /// Current institutional configuration (governance type, economic model, etc.)
    pub current_regime: String,
    /// How many ticks this regime has been in place.
    pub regime_duration: u32,
    /// Lock-in strength: increases with duration. Switching cost ∝ √(duration).
    /// CITATION: Arthur (1994) — QWERTY effect, increasing returns.
    pub lock_in_strength: f64,
    /// Number of regime changes in history (more changes = more fragile).
    pub change_count: u32,
    /// Institutional quality [0, 1]. Degrades without maintenance.
    pub quality: f64,
}

impl InstitutionalInertia {
    pub fn new(regime: &str) -> Self {
        Self {
            current_regime: regime.into(),
            regime_duration: 0,
            lock_in_strength: 0.0,
            change_count: 0,
            quality: 0.5,
        }
    }

    /// Tick: increase lock-in, degrade quality without investment.
    pub fn tick(&mut self, investment_fraction: f64) {
        self.regime_duration += 1;
        // Lock-in grows with sqrt of duration (diminishing returns on entrenchment)
        self.lock_in_strength = (self.regime_duration as f64).sqrt() * 0.01;
        // Quality degrades without investment, improves with investment
        let degradation = 0.0005; // natural institutional decay
        let improvement = investment_fraction * 0.002;
        self.quality = (self.quality - degradation + improvement).clamp(0.0, 1.0);
    }

    /// Cost of switching to a new regime [0, 1].
    /// High lock-in + high quality = very expensive to change.
    /// High lock-in + low quality = cheaper (people want change).
    pub fn switching_cost(&self) -> f64 {
        (self.lock_in_strength * self.quality).clamp(0.0, 1.0)
    }

    /// Attempt regime change. Returns true if successful.
    pub fn attempt_change(&mut self, pressure: f64, new_regime: &str) -> bool {
        if pressure > self.switching_cost() {
            self.current_regime = new_regime.into();
            self.regime_duration = 0;
            self.lock_in_strength = 0.0;
            self.change_count += 1;
            // Quality drops during transition
            self.quality *= 0.7;
            true
        } else {
            false
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE 4: Ecosystem Services
// ═══════════════════════════════════════════════════════════════════════════

/// Ecosystem health: biodiversity stock that provides services.
///
/// CITATION: Rockström et al. (2009) "Planetary Boundaries"
/// Costanza et al. (1997) "The value of the world's ecosystem services"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemHealth {
    /// Biodiversity index [0, 1]. 1.0 = pristine, 0 = collapsed.
    /// CITATION: Living Planet Index (WWF): declined 69% from 1970 to 2018.
    pub biodiversity: f64,
    /// Forest cover fraction [0, 1].
    /// Tipping point: Amazon at ~60% triggers savannification.
    /// CITATION: Lovejoy & Nobre (2018)
    pub forest_cover: f64,
    /// Soil health [0, 1]. Degrades with intensive agriculture.
    /// CITATION: FAO: 33% of soils are degraded globally.
    pub soil_health: f64,
    /// Ocean health [0, 1]. Acidification + overfishing + warming.
    pub ocean_health: f64,
    /// Freshwater availability [0, 1].
    pub freshwater: f64,
}

impl EcosystemHealth {
    pub fn earth_1970() -> Self {
        Self {
            biodiversity: 0.95, // ~5% loss by 1970
            forest_cover: 0.85, // pre-major deforestation
            soil_health: 0.80,
            ocean_health: 0.90,
            freshwater: 0.90,
        }
    }

    /// Tick ecosystem based on economic activity and population.
    pub fn tick(&mut self, population_billions: f64, gdp_per_capita: f64, policy_protection: f64) {
        // Pressure scales with population × consumption
        let pressure = population_billions * (gdp_per_capita / 10000.0).sqrt() * 0.0001;

        // Protection reduces pressure (environmental policy, consciousness)
        let net_pressure = pressure * (1.0 - policy_protection.clamp(0.0, 0.8));

        self.biodiversity = (self.biodiversity - net_pressure * 0.3).max(0.0);
        self.forest_cover = (self.forest_cover - net_pressure * 0.2).max(0.0);
        self.soil_health = (self.soil_health - net_pressure * 0.15).max(0.0);
        self.ocean_health = (self.ocean_health - net_pressure * 0.1).max(0.0);
        self.freshwater = (self.freshwater - net_pressure * 0.1).max(0.0);

        // Natural regeneration (slow)
        self.biodiversity = (self.biodiversity + 0.00005).min(1.0);
        self.forest_cover = (self.forest_cover + 0.0001).min(1.0);
    }

    /// Aggregate ecosystem service value [0, 1].
    /// Below 0.5 = degraded, below 0.3 = critical (tipping points).
    pub fn service_index(&self) -> f64 {
        self.biodiversity * 0.3
            + self.forest_cover * 0.2
            + self.soil_health * 0.2
            + self.ocean_health * 0.15
            + self.freshwater * 0.15
    }

    /// Agriculture productivity modifier from ecosystem services.
    /// Below tipping points, food production collapses nonlinearly.
    pub fn agriculture_modifier(&self) -> f64 {
        let soil_factor = if self.soil_health > 0.5 {
            1.0
        } else {
            self.soil_health * 2.0
        };
        let water_factor = if self.freshwater > 0.3 {
            1.0
        } else {
            self.freshwater * 3.3
        };
        let pollination = if self.biodiversity > 0.4 {
            1.0
        } else {
            self.biodiversity * 2.5
        };
        (soil_factor * water_factor * pollination).clamp(0.1, 1.0)
    }

    /// Check for tipping points. Returns list of critical thresholds crossed.
    pub fn tipping_points(&self) -> Vec<String> {
        let mut tips = Vec::new();
        if self.forest_cover < 0.60 {
            tips.push("Amazon savannification risk (forest <60%)".into());
        }
        if self.biodiversity < 0.30 {
            tips.push("Mass extinction threshold (biodiversity <30%)".into());
        }
        if self.ocean_health < 0.40 {
            tips.push("Ocean ecosystem collapse (ocean <40%)".into());
        }
        if self.soil_health < 0.30 {
            tips.push("Agricultural collapse risk (soil <30%)".into());
        }
        tips
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE 5: Trust Asymmetry
// ═══════════════════════════════════════════════════════════════════════════

/// Asymmetric trust dynamics: destruction is faster than creation.
///
/// CITATION: Slovic (1993) "Perceived risk, trust, and democracy"
/// — "Trust is fragile. It is typically created rather slowly, but it can
///    be destroyed in an instant."
/// Empirical ratio: ~5:1 (5 good interactions to recover from 1 betrayal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricTrust {
    /// Current trust level [0, 1].
    pub level: f64,
    /// Trust building rate per positive interaction.
    pub build_rate: f64,
    /// Trust destruction rate per negative interaction (higher = more fragile).
    /// CITATION: Slovic (1993). Typical ratio: destruction 3-5× faster than building.
    pub destroy_rate: f64,
    /// Recovery momentum: how many consecutive positive interactions since last betrayal.
    pub recovery_streak: u32,
    /// Total betrayals in history (creates lasting wariness).
    pub betrayal_count: u32,
}

impl AsymmetricTrust {
    pub fn new() -> Self {
        Self {
            level: 0.5,
            build_rate: 0.005,   // slow build
            destroy_rate: 0.025, // 5× faster destruction
            recovery_streak: 0,
            betrayal_count: 0,
        }
    }

    /// Record a positive interaction (cooperation, aid, honesty).
    pub fn positive_interaction(&mut self) {
        self.recovery_streak += 1;
        // Trust builds slowly, and more slowly after betrayals (wariness)
        let wariness = 1.0 / (1.0 + self.betrayal_count as f64 * 0.2);
        self.level = (self.level + self.build_rate * wariness).min(1.0);
    }

    /// Record a negative interaction (betrayal, defection, dishonesty).
    pub fn negative_interaction(&mut self) {
        self.recovery_streak = 0;
        self.betrayal_count += 1;
        // Trust destroyed rapidly — asymmetric
        self.level = (self.level - self.destroy_rate).max(0.0);
    }

    /// Trust recovery after sustained positive behavior.
    /// Takes ~5× longer than destruction (Slovic 1993).
    pub fn ticks_to_recover(&self) -> u32 {
        let deficit = 1.0 - self.level;
        let wariness = 1.0 / (1.0 + self.betrayal_count as f64 * 0.2);
        (deficit / (self.build_rate * wariness)).ceil() as u32
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE 6: Knowledge Network Effects
// ═══════════════════════════════════════════════════════════════════════════

/// Combinatorial knowledge: innovations unlock adjacent possibilities.
///
/// CITATION: Kauffman (1996) "At Home in the Universe" — adjacent possible
/// Arthur (2009) "The Nature of Technology" — combinatorial evolution
/// Romer (1990) "Endogenous Technological Change" — ideas are non-rival
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNetwork {
    /// Known technologies/concepts (each unlocks new possibilities).
    pub known_count: u32,
    /// Adjacent possible: concepts that COULD be discovered given current knowledge.
    /// Grows combinatorially: adjacent ∝ known × (known-1) / 2.
    pub adjacent_possible: u32,
    /// Innovation rate: probability of discovering a new concept per tick.
    /// Scales with researchers × adjacent_possible / known (diminishing within a paradigm).
    pub innovation_rate: f64,
    /// Paradigm depth: how deep we are into current paradigm.
    /// Resets on paradigm shift (new paradigm = new adjacent possible explosion).
    pub paradigm_depth: u32,
    /// Total paradigm shifts (scientific revolutions).
    pub paradigm_shifts: u32,
}

impl KnowledgeNetwork {
    pub fn new(initial_known: u32) -> Self {
        let adjacent = initial_known * (initial_known.saturating_sub(1)) / 2;
        Self {
            known_count: initial_known,
            adjacent_possible: adjacent.max(1),
            innovation_rate: 0.01,
            paradigm_depth: 0,
            paradigm_shifts: 0,
        }
    }

    /// Tick: attempt innovation, update adjacent possible.
    pub fn tick(&mut self, researchers_fraction: f64, network_diffusion: f64) {
        // Innovation probability: researchers × adjacent / (known × paradigm_inertia)
        let paradigm_inertia = 1.0 + self.paradigm_depth as f64 * 0.1;
        self.innovation_rate = researchers_fraction
            * (self.adjacent_possible as f64 / self.known_count.max(1) as f64)
            * network_diffusion
            / paradigm_inertia;

        // Recalculate adjacent possible (combinatorial growth)
        let n = self.known_count;
        self.adjacent_possible = (n * (n.saturating_sub(1)) / 2).max(1);
    }

    /// Record a discovery.
    pub fn discover(&mut self) {
        self.known_count += 1;
        self.paradigm_depth += 1;

        // Check for paradigm shift (every ~20 discoveries within a paradigm)
        // Paradigm shifts reset the exploration frontier
        if self.paradigm_depth >= 20 {
            self.paradigm_shifts += 1;
            self.paradigm_depth = 0;
            // New paradigm: explosion of adjacent possible
            self.adjacent_possible *= 3;
        }

        // Update adjacent possible
        let n = self.known_count;
        self.adjacent_possible = (n * (n.saturating_sub(1)) / 2).max(1);
    }

    /// Combinatorial growth rate: how fast the possibility space is expanding.
    pub fn growth_rate(&self) -> f64 {
        self.adjacent_possible as f64 / self.known_count.max(1) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMBINED PRIMITIVES ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/// All six primitives bundled per world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivilizationalPrimitives {
    pub network: NetworkTopology,
    pub resources: Vec<DepletableResource>,
    pub institutions: InstitutionalInertia,
    pub ecosystem: EcosystemHealth,
    pub trust: AsymmetricTrust,
    pub knowledge: KnowledgeNetwork,
}

impl CivilizationalPrimitives {
    pub fn earth_defaults() -> Self {
        Self {
            network: NetworkTopology::from_population(3700.0, 0.36),
            resources: earth_resources(),
            institutions: InstitutionalInertia::new("mixed_economy"),
            ecosystem: EcosystemHealth::earth_1970(),
            trust: AsymmetricTrust::new(),
            knowledge: KnowledgeNetwork::new(100), // ~100 foundational technologies by 1970
        }
    }

    /// Create Earth defaults with ecosystem initialized from B(t) biosphere history.
    /// The B(t) terminal state represents 3.8 billion years of biosphere evolution,
    /// degraded to 2026 conditions via HANPP thermodynamic bridge.
    pub fn earth_from_biosphere(bt_ecosystem: EcosystemHealth) -> Self {
        Self {
            network: NetworkTopology::from_population(3700.0, 0.36),
            resources: earth_resources(),
            institutions: InstitutionalInertia::new("mixed_economy"),
            ecosystem: bt_ecosystem,
            trust: AsymmetricTrust::new(),
            knowledge: KnowledgeNetwork::new(100),
        }
    }

    /// Tick all primitives for one month.
    pub fn tick(
        &mut self,
        pop_millions: f64,
        urbanization: f64,
        gdp_per_capita: f64,
        research_fraction: f64,
        policy_protection: f64,
        investment_fraction: f64,
    ) {
        // 1. Update network topology from population + urbanization
        self.network = NetworkTopology::from_population(pop_millions, urbanization);

        // 2. Tick resource depletion
        for resource in &mut self.resources {
            resource.tick();
        }

        // 3. Tick institutional inertia
        self.institutions.tick(investment_fraction);

        // 4. Tick ecosystem
        let pop_b = pop_millions / 1000.0;
        self.ecosystem
            .tick(pop_b, gdp_per_capita, policy_protection);

        // 5. Knowledge network with network diffusion
        let diffusion = self.network.innovation_diffusion_rate();
        self.knowledge.tick(research_fraction, diffusion);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_scales_with_urbanization() {
        let rural = NetworkTopology::from_population(100.0, 0.2);
        let urban = NetworkTopology::from_population(100.0, 0.8);
        assert!(
            urban.mean_degree > rural.mean_degree,
            "Urban networks denser: {} vs {}",
            urban.mean_degree,
            rural.mean_degree
        );
        assert!(
            urban.fragmentation < rural.fragmentation,
            "Urban less fragmented: {} vs {}",
            urban.fragmentation,
            rural.fragmentation
        );
    }

    #[test]
    fn test_innovation_diffusion_increases_with_connectivity() {
        let sparse = NetworkTopology::from_population(10.0, 0.1);
        let dense = NetworkTopology::from_population(1000.0, 0.8);
        assert!(dense.innovation_diffusion_rate() > sparse.innovation_diffusion_rate());
    }

    #[test]
    fn test_hubbert_curve_peaks_and_declines() {
        let mut resource = DepletableResource {
            name: "test".into(),
            total_reserve: 100.0,
            cumulative_extracted: 0.0,
            peak_rate: 1.0,
            depletion_tau: 100.0,
            current_rate: 0.0,
            eroi_at_full: 30.0,
            eroi_at_half: 15.0,
            eroi_at_depleted: 5.0,
        };
        // Run enough ticks to fully deplete
        for _ in 0..500 {
            resource.tick();
        }
        // Resource should be substantially depleted
        assert!(
            resource.fraction_remaining() < 0.1,
            "Should be mostly depleted: {:.1}% remaining",
            resource.fraction_remaining() * 100.0
        );
        // Current rate should be near zero at end (post-peak decline)
        assert!(
            resource.current_rate < resource.peak_rate * 0.5,
            "Post-depletion rate should be low: {}",
            resource.current_rate
        );
    }

    #[test]
    fn test_eroi_degrades_with_depletion() {
        let mut resource = earth_resources().into_iter().next().unwrap(); // Oil
        let eroi_early = resource.current_eroi();
        resource.cumulative_extracted = resource.total_reserve * 0.8;
        let eroi_late = resource.current_eroi();
        assert!(
            eroi_late < eroi_early,
            "EROI should degrade: {} vs {}",
            eroi_late,
            eroi_early
        );
    }

    #[test]
    fn test_institutional_switching_cost_increases() {
        let mut inst = InstitutionalInertia::new("democracy");
        let cost_early = inst.switching_cost();
        for _ in 0..120 {
            inst.tick(0.1);
        }
        let cost_late = inst.switching_cost();
        assert!(
            cost_late > cost_early,
            "Lock-in should increase: {} vs {}",
            cost_late,
            cost_early
        );
    }

    #[test]
    fn test_ecosystem_tipping_points() {
        let mut eco = EcosystemHealth::earth_1970();
        assert!(eco.tipping_points().is_empty(), "1970: no tipping points");
        eco.forest_cover = 0.55;
        eco.biodiversity = 0.25;
        let tips = eco.tipping_points();
        assert_eq!(tips.len(), 2, "Should have 2 tipping points: {:?}", tips);
    }

    #[test]
    fn test_trust_asymmetry() {
        let mut trust = AsymmetricTrust::new();
        let initial = trust.level;
        trust.positive_interaction();
        let after_good = trust.level;
        trust.negative_interaction();
        let after_bad = trust.level;
        // One bad should undo more than one good
        let good_delta = after_good - initial;
        let bad_delta = after_good - after_bad;
        assert!(
            bad_delta > good_delta * 3.0,
            "Destruction should be >3× faster: bad={bad_delta}, good={good_delta}"
        );
    }

    #[test]
    fn test_knowledge_combinatorial_growth() {
        let mut k = KnowledgeNetwork::new(10);
        let adj_10 = k.adjacent_possible;
        k.discover();
        k.discover();
        k.discover();
        let adj_13 = k.adjacent_possible;
        // Adjacent possible should grow faster than linear
        assert!(
            adj_13 > adj_10 + 10,
            "Combinatorial growth: {} vs {}",
            adj_13,
            adj_10
        );
    }

    #[test]
    fn test_knowledge_paradigm_shift() {
        let mut k = KnowledgeNetwork::new(10);
        assert_eq!(k.paradigm_shifts, 0);
        for _ in 0..20 {
            k.discover();
        }
        assert!(
            k.paradigm_shifts >= 1,
            "Should have paradigm shift after 20 discoveries"
        );
    }

    #[test]
    fn test_combined_primitives() {
        let mut p = CivilizationalPrimitives::earth_defaults();
        p.tick(3700.0, 0.36, 3000.0, 0.05, 0.3, 0.1);
        assert!(p.network.mean_degree > 50.0);
        assert!(p.ecosystem.service_index() > 0.5);
        assert!(p.knowledge.adjacent_possible > 0);
    }
}
