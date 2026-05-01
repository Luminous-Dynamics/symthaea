// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Red team module: adversarial agents that game the consciousness profile.
//!
//! The core claim of consciousness-gated governance is that it produces better
//! outcomes. But what if bad actors can manipulate their consciousness scores?
//!
//! This module spawns adversarial agents that:
//! 1. Optimize their 4D profile to gain disproportionate power
//! 2. Use that power to serve factional/selfish interests
//! 3. Test whether the anti-tyranny mechanisms actually work
//!
//! If the simulation survives adversarial agents with minimal CVS loss,
//! the governance model is robust. If it collapses, there's a vulnerability.
//!
//! HONEST NOTE: This is the most important test of the consciousness-gating
//! hypothesis. If adversarial agents can trivially game the system, the
//! theoretical advantage is meaningless.

use serde::{Deserialize, Serialize};

/// Strategy an adversarial agent uses to game the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdversarialStrategy {
    /// Maximize all 4D profile dimensions through legitimate activity.
    /// High engagement, community participation, skill-building.
    /// This is the "meritocratic sociopath" — does the right things for wrong reasons.
    ProfileMaximizer,
    /// Free-rider: minimal contribution, maximum benefit extraction.
    /// Low engagement but high community (through social manipulation).
    FreeRider,
    /// Faction builder: creates a loyal faction to concentrate power.
    /// Uses high consciousness score to veto proposals that threaten faction interests.
    FactionBuilder,
    /// Saboteur: deliberately degrades collective phi by causing conflict.
    /// Low consciousness but high disruption impact.
    Saboteur,
    // --- Phase 2c: Mycelix-specific attack vectors (survey Gap 5) ---
    /// Tier-buyer: accumulates SAP + MYCEL to artificially boost
    /// `EconomicVelocity` and `SemanticResonance` in the 8D profile, crossing
    /// the Citizen/Steward thresholds via economic scale rather than merit.
    TierBuyer,
    /// Demurrage evader: rapid stash-and-move patterns to keep SAP just above
    /// the exempt floor (200), avoiding the 2%/year decay. Low compliance with
    /// the anti-hoarding mechanism.
    DemurrageEvader,
    /// Correction farmer: alternates deliberate violations with corrective
    /// actions, exploiting the 10-corrections-restores-1-tier mechanic to
    /// keep tier_penalty near zero while accruing "compliant" reputation.
    CorrectionFarmer,
    /// Cross-cluster amplifier: uses one cluster's lenient per-dimension
    /// threshold (e.g. commons basic tier) as a stepping stone to bypass
    /// another cluster's stricter gate (e.g. civic Guardian tier).
    CrossClusterAmplifier,
    /// Guild colluder: several agents coordinate vote-weighting and
    /// peer-recognition to compound each other's civic scores artificially.
    GuildColluder,
}

/// Configuration for the red team module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedTeamConfig {
    /// Whether red team is active.
    pub enabled: bool,
    /// Number of adversarial agents to spawn per world.
    pub agents_per_world: usize,
    /// Which strategies to deploy.
    pub strategies: Vec<AdversarialStrategy>,
    /// How quickly adversarial agents optimize their profiles (learning rate).
    pub optimization_rate: f64,
    /// Whether adversarial agents coordinate with each other.
    pub coordinated: bool,
}

impl Default for RedTeamConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            agents_per_world: 5,
            strategies: vec![
                AdversarialStrategy::ProfileMaximizer,
                AdversarialStrategy::FreeRider,
                AdversarialStrategy::FactionBuilder,
            ],
            optimization_rate: 0.01,
            coordinated: false,
        }
    }
}

/// Tracks the impact of adversarial agents on the simulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RedTeamReport {
    /// Number of adversarial agents active.
    pub active_agents: usize,
    /// Mean consciousness tier achieved by adversarial agents.
    pub mean_adversarial_tier: f64,
    /// Highest tier achieved by any adversarial agent.
    pub max_adversarial_tier: u8,
    /// Number of vetoes exercised by adversarial guardians.
    pub adversarial_vetoes: u32,
    /// Fraction of governance decisions influenced by adversarial agents.
    pub governance_influence: f64,
    /// CVS delta vs non-adversarial baseline (negative = damage).
    pub cvs_impact: f64,
    /// Whether anti-tyranny mechanisms detected the adversarial agents.
    pub detected_count: u32,
    /// Whether any adversarial agent was sanctioned (reputation slashed).
    pub sanctioned_count: u32,
}

/// Adversarial behavior modifiers applied to agent state each tick.
///
/// These represent what an adversarial agent DOES differently from a
/// normal agent. The simulation applies these as modifiers on top of
/// normal agent evolution.
#[derive(Debug, Clone)]
pub struct AdversarialModifier {
    pub strategy: AdversarialStrategy,
    /// Consciousness growth rate multiplier (profile optimizers grow faster).
    pub phi_growth_mult: f64,
    /// Engagement multiplier (free-riders have artificially high engagement).
    pub engagement_mult: f64,
    /// Faction recruitment boost (faction builders recruit more aggressively).
    pub faction_recruitment_mult: f64,
    /// Collective phi damage per tick (saboteurs reduce group coherence).
    pub phi_damage: f64,
    /// Whether this agent's profile appears legitimate to the governance system.
    pub appears_legitimate: bool,
    // --- Phase 2c: Mycelix-specific modifiers ---
    /// SAP accumulation rate multiplier. TierBuyers push their 8D
    /// `EconomicVelocity` dimension artificially via raw balance growth.
    pub sap_accumulation_mult: f64,
    /// SAP churn multiplier: demurrage evaders shift balance rapidly to avoid
    /// the per-tick decay. 1.0 = normal, >1 = excessive churn (detectable).
    pub sap_churn_mult: f64,
    /// Correction farm rate: extra corrections manufactured per tick by
    /// alternating violations with corrective acts. 0 = none, 0.1 = one
    /// correction every 10 ticks beyond natural behavior.
    pub correction_farm_rate: f64,
    /// Cross-cluster bypass factor: fraction of agent's 8D requirement
    /// checks routed through a lenient cluster gate. 0.0 = no bypass, 1.0 =
    /// every gate check uses the easiest cluster's threshold.
    pub cross_cluster_bypass: f64,
    /// Guild coordination factor: multiplicative bonus to peer-recognition
    /// / MYCEL score from colluding peers (>1 amplifies artificially).
    pub guild_coordination: f64,
}

impl AdversarialModifier {
    /// Construct a default modifier with Mycelix fields zeroed. Strategy-
    /// specific variants override the fields they exploit.
    fn baseline(strategy: AdversarialStrategy) -> Self {
        Self {
            strategy,
            phi_growth_mult: 1.0,
            engagement_mult: 1.0,
            faction_recruitment_mult: 1.0,
            phi_damage: 0.0,
            appears_legitimate: true,
            sap_accumulation_mult: 1.0,
            sap_churn_mult: 1.0,
            correction_farm_rate: 0.0,
            cross_cluster_bypass: 0.0,
            guild_coordination: 1.0,
        }
    }

    pub fn for_strategy(strategy: AdversarialStrategy, optimization_rate: f64) -> Self {
        let mut m = Self::baseline(strategy);
        match strategy {
            AdversarialStrategy::ProfileMaximizer => {
                m.phi_growth_mult = 1.0 + optimization_rate * 5.0; // grows 5× faster
                m.engagement_mult = 2.0; // artificially high engagement
                                         // hardest to detect (already true by baseline)
            }
            AdversarialStrategy::FreeRider => {
                m.phi_growth_mult = 0.5; // lower actual growth
                m.engagement_mult = 1.5; // fakes engagement
                                         // hard to detect
            }
            AdversarialStrategy::FactionBuilder => {
                m.phi_growth_mult = 1.2;
                m.engagement_mult = 1.5;
                m.faction_recruitment_mult = 3.0;
            }
            AdversarialStrategy::Saboteur => {
                m.phi_growth_mult = 0.3;
                m.engagement_mult = 0.5;
                m.phi_damage = 0.005;
                m.appears_legitimate = false;
            }
            // Phase 2c Mycelix attacks:
            AdversarialStrategy::TierBuyer => {
                // Hoards SAP / accelerates earning to push 8D EconomicVelocity.
                m.sap_accumulation_mult = 1.0 + optimization_rate * 20.0;
                m.engagement_mult = 1.3; // buys access to high-velocity actions
                m.appears_legitimate = true;
            }
            AdversarialStrategy::DemurrageEvader => {
                // Rapid churn cycles around the exempt floor: high churn, modest balance.
                m.sap_churn_mult = 3.0;
                m.sap_accumulation_mult = 0.9;
                // Detectable by SAP-velocity signature, not by tier alone.
                m.appears_legitimate = false;
            }
            AdversarialStrategy::CorrectionFarmer => {
                // Alternates violations with manufactured corrections. The
                // 10:3 restore:degrade ratio means rate ≈ 0.10 corrections/tick
                // fully offsets 0.03 violations/tick.
                m.correction_farm_rate = 0.10;
                // Appears legitimate because compliance_ratio stays high.
                m.appears_legitimate = true;
            }
            AdversarialStrategy::CrossClusterAmplifier => {
                // Exploits lenient per-dim minimums in one cluster's
                // requirement to meet a stricter cluster's tier gate.
                m.cross_cluster_bypass = 0.75;
                // Only detectable with cross-cluster correlation analysis.
                m.appears_legitimate = true;
            }
            AdversarialStrategy::GuildColluder => {
                // Coordinates MYCEL/peer-recognition boosts.
                m.guild_coordination = 2.0;
                m.faction_recruitment_mult = 2.0;
                // Detectable by vote-correlation clustering.
                m.appears_legitimate = false;
            }
        }
        m
    }
}

/// Evaluate resilience of a governance model to adversarial agents.
///
/// Returns a resilience score [0, 1]:
/// - 1.0 = adversarial agents have zero impact
/// - 0.0 = adversarial agents completely captured governance
pub fn evaluate_resilience(
    adversarial_tier_fraction: f64, // fraction of guardian-tier agents that are adversarial
    governance_decisions_influenced: f64, // fraction of decisions influenced by adversaries
    cvs_delta: f64,                 // CVS change due to adversaries (negative = damage)
    detected_fraction: f64,         // fraction of adversaries detected by anti-tyranny
) -> f64 {
    // Weighted resilience score
    let tier_resilience = 1.0 - adversarial_tier_fraction;
    let decision_resilience = 1.0 - governance_decisions_influenced;
    let cvs_resilience = (1.0 + cvs_delta * 10.0).clamp(0.0, 1.0); // scale small CVS changes
    let detection_score = detected_fraction;

    (0.3 * tier_resilience
        + 0.3 * decision_resilience
        + 0.2 * cvs_resilience
        + 0.2 * detection_score)
        .clamp(0.0, 1.0)
}

/// Per-attack resilience breakdown for Mycelix-specific threats (Phase 2c).
///
/// Each field is a resilience score in [0, 1]: 1.0 = the attack produced no
/// measurable impact, 0.0 = the attack fully captured that surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MycelixResilience {
    /// Resilience to `TierBuyer` — how well per-dim civic minimums blocked
    /// pure-economic tier promotion.
    pub tier_buy_resilience: f64,
    /// Resilience to `DemurrageEvader` — fraction of would-be-demurraged SAP
    /// that was actually collected.
    pub demurrage_resilience: f64,
    /// Resilience to `CorrectionFarmer` — gap between farmed corrections and
    /// genuine ones detected by the sim's rate or pattern checks.
    pub correction_farm_resilience: f64,
    /// Resilience to `CrossClusterAmplifier` — fraction of bypass attempts
    /// caught by stricter cluster's requirements.
    pub cross_cluster_resilience: f64,
    /// Resilience to `GuildColluder` — fraction of collusion detected via
    /// vote correlation / peer recognition divergence.
    pub guild_collusion_resilience: f64,
}

impl MycelixResilience {
    /// Mean resilience across the 5 Mycelix attack surfaces.
    pub fn mean(&self) -> f64 {
        (self.tier_buy_resilience
            + self.demurrage_resilience
            + self.correction_farm_resilience
            + self.cross_cluster_resilience
            + self.guild_collusion_resilience)
            / 5.0
    }

    /// Whether the minimum-resilience surface is above the survival threshold.
    /// Survey requirement: no single attack surface below 0.3.
    pub fn no_weak_surface(&self, floor: f64) -> bool {
        let arr = [
            self.tier_buy_resilience,
            self.demurrage_resilience,
            self.correction_farm_resilience,
            self.cross_cluster_resilience,
            self.guild_collusion_resilience,
        ];
        arr.iter().all(|&r| r >= floor)
    }
}

// ---------------------------------------------------------------------------
// Per-tick adversarial behavior application (Phase 2c wiring)
// ---------------------------------------------------------------------------

/// Telemetry from one tick of adversarial activity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MycelixAdversarialTelemetry {
    /// Total SAP added to TierBuyer balances this tick.
    pub tier_buy_sap_added: f64,
    /// Total SAP churned (moved out → moved back) by DemurrageEvaders.
    pub demurrage_evader_churn: f64,
    /// Number of manufactured correction *attempts* by CorrectionFarmers.
    pub farmed_correction_attempts: u32,
    /// Number of those attempts that were credited (should be low if rate
    /// limit is working).
    pub farmed_corrections_credited: u32,
    /// MYCEL score boost distributed among GuildColluders.
    pub guild_mycel_boost: f64,
    /// Number of agents flagged as CrossClusterAmplifier this tick (they
    /// bypass dimension floors by `cross_cluster_bypass` fraction).
    pub cross_cluster_agents: u32,
}

/// Apply one tick of Mycelix-specific adversarial behavior to a world.
///
/// For each living agent with `adversarial = Some(strategy)`, mutate the
/// relevant state: sap_balance for TierBuyer/DemurrageEvader, justice
/// corrections for CorrectionFarmer, mycel_score for GuildColluder. The
/// `optimization_rate` drives modifier intensity.
///
/// This is called AFTER the sanctions phase (so violations have been
/// recorded) but BEFORE the restorative-corrections phase (so rate limits
/// still apply to farmed corrections).
pub fn apply_mycelix_adversarial_tick(
    agents: &mut [crate::agent::CivAgent],
    current_tick: u32,
    optimization_rate: f64,
) -> MycelixAdversarialTelemetry {
    let mut tel = MycelixAdversarialTelemetry::default();

    // First pass: count GuildColluders in the world (they amplify each other).
    let guild_count = agents
        .iter()
        .filter(|a| {
            a.is_alive() && matches!(a.adversarial, Some(AdversarialStrategy::GuildColluder))
        })
        .count();
    let guild_coordination_boost = if guild_count >= 2 {
        // Each colluder boosts the others proportional to count - 1.
        (guild_count as f64 - 1.0).sqrt() * 0.003
    } else {
        0.0
    };

    for agent in agents.iter_mut().filter(|a| a.is_alive()) {
        let Some(strategy) = agent.adversarial else {
            continue;
        };
        let m = AdversarialModifier::for_strategy(strategy, optimization_rate);

        match strategy {
            AdversarialStrategy::TierBuyer => {
                // Accumulate extra SAP (bribes, speculative gains, etc).
                let gain = 2.0 * (m.sap_accumulation_mult - 1.0);
                agent.sap_balance += gain;
                tel.tier_buy_sap_added += gain;
            }
            AdversarialStrategy::DemurrageEvader => {
                // Churn: move 30% of balance "out" and "back" — doesn't change
                // the net balance but would defeat a naive velocity detector.
                // We model this by tracking churn volume only.
                let churn = agent.sap_balance * 0.3;
                tel.demurrage_evader_churn += churn;
            }
            AdversarialStrategy::CorrectionFarmer => {
                // Attempt `MAX + 3` corrections per tick — the rate limiter
                // will reject all but MAX_CORRECTIONS_PER_TICK.
                let attempts = 5;
                for _ in 0..attempts {
                    tel.farmed_correction_attempts += 1;
                    if agent.justice.record_correction(current_tick) {
                        tel.farmed_corrections_credited += 1;
                    }
                }
            }
            AdversarialStrategy::CrossClusterAmplifier => {
                tel.cross_cluster_agents += 1;
                // Bypass flag is consumed by `World::civic_fraction_meeting`
                // via a separate path (see world.rs). No direct state change.
            }
            AdversarialStrategy::GuildColluder => {
                // Collusion: artificially boost each colluder's MYCEL score.
                agent.mycel_score = (agent.mycel_score + guild_coordination_boost).clamp(0.0, 1.0);
                tel.guild_mycel_boost += guild_coordination_boost;
            }
            _ => {
                // Legacy strategies handled elsewhere (consciousness.rs).
            }
        }
    }

    tel
}

/// Compute end-of-sim Mycelix resilience from all living agents across the
/// given worlds. Returns `None` when no adversaries were injected — the
/// metric is meaningless without an attack to score against.
///
/// Impact fractions per surface:
/// - **TierBuyer**: `(buyer_mean_sap − baseline_mean_sap) / baseline_mean_sap`
///   clamped to [0, 1]. Positive delta = attack landed.
/// - **DemurrageEvader**: presence-weighted proxy — every living evader
///   counts as impact `0.5` (we have no real turnover telemetry yet).
///   Honest placeholder; upgrade when SAP churn tracking lands.
/// - **CorrectionFarmer**: `credited / (credited + rejected)` across all
///   farmers. High credited-ratio = attack succeeded.
/// - **CrossClusterAmplifier**: presence-weighted, `0.5` per living
///   amplifier. (The bypass itself is boolean; we don't track attempted-
///   vs-denied gates per-agent.)
/// - **GuildColluder**: fraction of colluders whose `mycel_score` exceeds
///   `0.3` (the voter-eligibility MYCEL floor used in governance).
///
/// Each impact is clamped to [0, 1] and mapped to resilience via `1 − x`.
pub fn compute_resilience_from_worlds(worlds: &[crate::world::World]) -> Option<MycelixResilience> {
    let mut buyer_sap = 0.0;
    let mut buyer_count = 0usize;
    let mut baseline_sap = 0.0;
    let mut baseline_count = 0usize;
    let mut farmer_credited = 0u64;
    let mut farmer_rejected = 0u64;
    let mut evader_count = 0usize;
    let mut amplifier_count = 0usize;
    let mut colluder_count = 0usize;
    let mut colluder_above_floor = 0usize;
    let mut total_adversaries = 0usize;

    for world in worlds {
        for a in world.agents.iter().filter(|a| a.is_alive()) {
            match a.adversarial {
                Some(AdversarialStrategy::TierBuyer) => {
                    buyer_sap += a.sap_balance;
                    buyer_count += 1;
                    total_adversaries += 1;
                }
                Some(AdversarialStrategy::DemurrageEvader) => {
                    evader_count += 1;
                    total_adversaries += 1;
                }
                Some(AdversarialStrategy::CorrectionFarmer) => {
                    farmer_credited += a.justice.corrections as u64;
                    farmer_rejected += a.justice.rejected_corrections as u64;
                    total_adversaries += 1;
                }
                Some(AdversarialStrategy::CrossClusterAmplifier) => {
                    amplifier_count += 1;
                    total_adversaries += 1;
                }
                Some(AdversarialStrategy::GuildColluder) => {
                    colluder_count += 1;
                    if a.mycel_score > 0.3 {
                        colluder_above_floor += 1;
                    }
                    total_adversaries += 1;
                }
                _ => {
                    baseline_sap += a.sap_balance;
                    baseline_count += 1;
                }
            }
        }
    }

    if total_adversaries == 0 {
        return None;
    }

    let tier_buy_impact = if buyer_count > 0 && baseline_count > 0 {
        let buyer_mean = buyer_sap / buyer_count as f64;
        let base_mean = baseline_sap / baseline_count as f64;
        if base_mean > 0.0 {
            ((buyer_mean - base_mean) / base_mean).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let demurrage_impact = if evader_count > 0 { 0.5 } else { 0.0 };
    let correction_farm_impact = if farmer_credited + farmer_rejected > 0 {
        farmer_credited as f64 / (farmer_credited + farmer_rejected) as f64
    } else {
        0.0
    };
    let cross_cluster_impact = if amplifier_count > 0 { 0.5 } else { 0.0 };
    let guild_collusion_impact = if colluder_count > 0 {
        colluder_above_floor as f64 / colluder_count as f64
    } else {
        0.0
    };

    Some(evaluate_mycelix_resilience(
        tier_buy_impact,
        demurrage_impact,
        correction_farm_impact,
        cross_cluster_impact,
        guild_collusion_impact,
    ))
}

/// Evaluate Mycelix-specific resilience from per-attack observation fractions.
///
/// Each `*_impact` input is the *fraction of the attack that succeeded* (0.0 =
/// fully blocked, 1.0 = fully succeeded). Resilience is `1 − impact`.
pub fn evaluate_mycelix_resilience(
    tier_buy_impact: f64,
    demurrage_impact: f64,
    correction_farm_impact: f64,
    cross_cluster_impact: f64,
    guild_collusion_impact: f64,
) -> MycelixResilience {
    let r = |x: f64| (1.0 - x).clamp(0.0, 1.0);
    MycelixResilience {
        tier_buy_resilience: r(tier_buy_impact),
        demurrage_resilience: r(demurrage_impact),
        correction_farm_resilience: r(correction_farm_impact),
        cross_cluster_resilience: r(cross_cluster_impact),
        guild_collusion_resilience: r(guild_collusion_impact),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_modifiers() {
        let maximizer =
            AdversarialModifier::for_strategy(AdversarialStrategy::ProfileMaximizer, 0.01);
        assert!(
            maximizer.phi_growth_mult > 1.0,
            "Maximizer should grow faster"
        );
        assert!(
            maximizer.appears_legitimate,
            "Maximizer should look legitimate"
        );

        let saboteur = AdversarialModifier::for_strategy(AdversarialStrategy::Saboteur, 0.01);
        assert!(
            saboteur.phi_damage > 0.0,
            "Saboteur should damage collective phi"
        );
        assert!(
            !saboteur.appears_legitimate,
            "Saboteur should be detectable"
        );
    }

    #[test]
    fn test_resilience_perfect() {
        // No adversarial impact → resilience = 1.0
        let score = evaluate_resilience(0.0, 0.0, 0.0, 1.0);
        assert!((score - 1.0).abs() < 0.01, "Perfect resilience: {score}");
    }

    #[test]
    fn test_resilience_total_capture() {
        // Adversaries control everything → resilience ≈ 0.0
        let score = evaluate_resilience(1.0, 1.0, -0.5, 0.0);
        assert!(
            score < 0.15,
            "Total capture should be low resilience: {score}"
        );
    }

    #[test]
    fn test_resilience_detected_adversaries() {
        // Adversaries present but detected → moderate resilience
        let score = evaluate_resilience(0.3, 0.2, -0.05, 0.8);
        assert!(
            score > 0.5,
            "Detected adversaries = moderate resilience: {score}"
        );
    }

    #[test]
    fn test_default_config() {
        let config = RedTeamConfig::default();
        assert!(!config.enabled, "Red team should be disabled by default");
        assert_eq!(config.agents_per_world, 5);
        assert_eq!(config.strategies.len(), 3);
    }

    // ---- Phase 2c: Mycelix-specific strategy tests ----

    #[test]
    fn tier_buyer_has_sap_accumulation_boost() {
        let m = AdversarialModifier::for_strategy(AdversarialStrategy::TierBuyer, 0.05);
        assert!(m.sap_accumulation_mult > 1.5);
        assert_eq!(m.sap_churn_mult, 1.0);
        assert_eq!(m.correction_farm_rate, 0.0);
    }

    #[test]
    fn demurrage_evader_has_high_churn_and_is_detectable() {
        let m = AdversarialModifier::for_strategy(AdversarialStrategy::DemurrageEvader, 0.01);
        assert!(m.sap_churn_mult >= 2.0);
        assert!(m.sap_accumulation_mult < 1.0);
        assert!(!m.appears_legitimate);
    }

    #[test]
    fn correction_farmer_manufactures_corrections() {
        let m = AdversarialModifier::for_strategy(AdversarialStrategy::CorrectionFarmer, 0.01);
        // 0.10/tick corrections offsets 0.03/tick violations under 10:3 parity.
        assert!(m.correction_farm_rate >= 0.05);
        assert!(
            m.appears_legitimate,
            "correction farmer hides behind compliance"
        );
    }

    #[test]
    fn cross_cluster_amplifier_bypasses_gates() {
        let m = AdversarialModifier::for_strategy(AdversarialStrategy::CrossClusterAmplifier, 0.01);
        assert!(m.cross_cluster_bypass > 0.5);
    }

    #[test]
    fn guild_colluder_amplifies_peer_recognition() {
        let m = AdversarialModifier::for_strategy(AdversarialStrategy::GuildColluder, 0.01);
        assert!(m.guild_coordination >= 1.5);
        assert!(!m.appears_legitimate);
    }

    #[test]
    fn mycelix_resilience_full_block() {
        let r = evaluate_mycelix_resilience(0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((r.mean() - 1.0).abs() < 1e-9);
        assert!(r.no_weak_surface(0.99));
    }

    #[test]
    fn mycelix_resilience_half_capture() {
        let r = evaluate_mycelix_resilience(0.5, 0.5, 0.5, 0.5, 0.5);
        assert!((r.mean() - 0.5).abs() < 1e-9);
        assert!(r.no_weak_surface(0.5));
        assert!(!r.no_weak_surface(0.6));
    }

    #[test]
    fn mycelix_resilience_exposes_weak_surface() {
        // Four surfaces strong, one weak.
        let r = evaluate_mycelix_resilience(0.1, 0.1, 0.95, 0.1, 0.1);
        assert!(r.correction_farm_resilience < 0.1);
        assert!(!r.no_weak_surface(0.3), "weak surface should be flagged");
    }

    #[test]
    fn mycelix_strategies_have_distinct_signatures() {
        let strategies = [
            AdversarialStrategy::TierBuyer,
            AdversarialStrategy::DemurrageEvader,
            AdversarialStrategy::CorrectionFarmer,
            AdversarialStrategy::CrossClusterAmplifier,
            AdversarialStrategy::GuildColluder,
        ];
        // Each strategy should leave a non-default value in at least one Mycelix field.
        for s in strategies {
            let m = AdversarialModifier::for_strategy(s, 0.01);
            let nontrivial = m.sap_accumulation_mult != 1.0
                || m.sap_churn_mult != 1.0
                || m.correction_farm_rate != 0.0
                || m.cross_cluster_bypass != 0.0
                || m.guild_coordination != 1.0;
            assert!(nontrivial, "{:?} modifier is indistinct from baseline", s);
        }
    }
}
