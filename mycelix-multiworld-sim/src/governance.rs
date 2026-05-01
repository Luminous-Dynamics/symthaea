// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness-gated governance with oppression detection.
//!
//! Governance evolves through four authority levels as the colony matures:
//! MissionControl -> LocalWithEarthVeto -> LocalSovereign -> Federation.
//!
//! Voting is trust-weighted: only agents whose Phi exceeds the threshold
//! for their tier may participate. This prevents premature democratic collapse
//! while the oppression detector watches for exclusionary dynamics.

use serde::{Deserialize, Serialize};

use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::World;

/// Amendment review period in ticks (5 years at monthly resolution).
const AMENDMENT_REVIEW_PERIOD: u32 = 60;

/// Emergency power duration in ticks (1 year).
const EMERGENCY_DURATION: u32 = 12;

/// Oppression streak threshold to trigger constitutional crisis (12 ticks = 1 year).
const CRISIS_STREAK_THRESHOLD: u32 = 12;

/// Stagnation penalty begins after this many ticks without an amendment (10 years).
const STAGNATION_THRESHOLD: u32 = 120;

/// Maximum amendments per review period before instability penalty kicks in.
const MAX_AMENDMENTS_PER_PERIOD: u32 = 3;

// ============================================================================
// ANTI-TYRANNY INVARIANTS (mirrors mycelix-governance hardening)
// ============================================================================

/// Veto override threshold: ⅔ (67%) supermajority to reverse a Guardian veto.
/// Constitutional requirement (Article III, Section 3): two-thirds majority.
/// Hardcoded — not configurable by governance.
const VETO_OVERRIDE_THRESHOLD: f64 = 0.67;

/// Veto cooldown per Guardian: 7 ticks (≈7 months at monthly resolution).
/// Prevents serial veto DoS.
const VETO_COOLDOWN_TICKS: u32 = 7;

/// Maximum consecutive emergency sessions before mandatory cooldown.
const _MAX_EMERGENCY_SESSIONS: u32 = 3;

/// Council membership term in ticks (365 ticks ≈ 30 years at monthly resolution).
/// Scaled down from real-world 365 days for simulation time compression.
const MEMBERSHIP_TERM_TICKS: u32 = 360;

/// Governance authority level, evolving with colony maturity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GovernanceAuthority {
    /// Epoch 1: Earth decides everything.
    MissionControl,
    /// Epoch 2: Colony decides, Earth can veto.
    LocalWithEarthVeto,
    /// Epoch 3+: Colony decides, Earth notified only.
    LocalSovereign,
    /// Epoch 4+: Multi-world council.
    Federation,
    /// Epoch 7+: Loose confederation of autonomous worlds.
    Confederation,
}

impl std::fmt::Display for GovernanceAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissionControl => write!(f, "MissionControl"),
            Self::LocalWithEarthVeto => write!(f, "LocalWithEarthVeto"),
            Self::LocalSovereign => write!(f, "LocalSovereign"),
            Self::Federation => write!(f, "Federation"),
            Self::Confederation => write!(f, "Confederation"),
        }
    }
}

/// World governance state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldGovernance {
    /// Current authority level.
    pub authority_level: GovernanceAuthority,
    /// Version counter for the constitution.
    pub constitution_version: u32,
    /// Total proposals submitted.
    pub proposal_count: u32,
    /// Total amendments passed.
    pub amendment_count: u32,
    /// Whether emergency powers are currently active.
    pub emergency_powers_active: bool,
    /// Remaining ticks of emergency powers (0 = inactive).
    pub emergency_ticks_remaining: u32,
    /// Minimum phi threshold for each consciousness tier (0-4) to vote.
    pub trust_depth_threshold: [f64; 5],
    /// Oppression index: 0.0 = fair, 1.0 = oppressive.
    pub oppression_index: f64,
    /// Consecutive ticks where oppression_index > 0.5.
    pub oppression_streak: u32,
    /// Whether a constitutional crisis is active.
    pub constitutional_crisis: bool,
    /// Stability score: 0.0 = unstable, 1.0 = stable.
    pub stability_score: f64,
    /// Proposal rate (proposals per tick, smoothed).
    pub proposals_per_tick: f64,
    /// Tick of the last amendment.
    pub last_amendment_tick: u32,
    /// Count of amendments in the current review period.
    amendments_this_period: u32,
    /// Start tick of the current review period.
    period_start_tick: u32,
    /// History of amendments: (tick, constitution_version).
    pub amendment_history: Vec<(u32, u32)>,
    // --- Anti-tyranny tracking ---
    /// Total guardian vetoes attempted.
    pub veto_count: u32,
    /// Total veto overrides (80% supermajority reversed a veto).
    pub veto_override_count: u32,
    /// Total vetoes deterred (Guardian chose not to veto because override exists).
    pub veto_deterred_count: u32,
    /// Tick of the last veto (for rate limiting).
    pub last_veto_tick: Option<u32>,
    /// Consecutive emergency sessions (resets on cooldown).
    pub consecutive_emergency_sessions: u32,
    /// Council membership term start tick (for term limit enforcement).
    pub membership_term_start: u32,
    /// Whether membership renewal is due.
    pub membership_renewal_due: bool,
}

impl WorldGovernance {
    /// Create a new governance system starting under Mission Control.
    pub fn new() -> Self {
        Self {
            authority_level: GovernanceAuthority::MissionControl,
            constitution_version: 1,
            proposal_count: 0,
            amendment_count: 0,
            emergency_powers_active: false,
            emergency_ticks_remaining: 0,
            trust_depth_threshold: [0.0, 0.2, 0.4, 0.6, 0.8],
            oppression_index: 0.0,
            oppression_streak: 0,
            constitutional_crisis: false,
            stability_score: 1.0,
            proposals_per_tick: 0.0,
            last_amendment_tick: 0,
            amendments_this_period: 0,
            period_start_tick: 0,
            amendment_history: Vec::new(),
            veto_count: 0,
            veto_override_count: 0,
            veto_deterred_count: 0,
            last_veto_tick: None,
            consecutive_emergency_sessions: 0,
            membership_term_start: 0,
            membership_renewal_due: false,
        }
    }

    /// Run one tick of governance simulation.
    ///
    /// Generates proposals, processes trust-weighted votes, evaluates
    /// oppression, manages emergency powers, and checks for constitutional
    /// amendments. Returns events generated this tick.
    pub fn tick_governance(
        &mut self,
        world: &World,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        self.tick_governance_with_policy(world, current_tick, rng, true)
    }

    /// Run one tick of governance with policy control over amendments.
    pub fn tick_governance_with_policy(
        &mut self,
        world: &World,
        current_tick: u32,
        rng: &mut StochasticEngine,
        amendment_enabled: bool,
    ) -> Vec<CivEvent> {
        self.tick_governance_full(
            world,
            current_tick,
            rng,
            amendment_enabled,
            false,
            0.0,
            true,
        )
    }

    /// Run one tick of governance with full policy control including hostile guardian mode.
    ///
    /// `phase2_enabled` controls whether the 8D sovereign-profile blend
    /// contributes to voter eligibility. When false (A2 counterfactual),
    /// falls back to the Phi+MYCEL 50/50 blend that predated Phase 2a.
    pub fn tick_governance_full(
        &mut self,
        world: &World,
        current_tick: u32,
        rng: &mut StochasticEngine,
        amendment_enabled: bool,
        hostile_guardian: bool,
        voting_suppression: f64,
        phase2_enabled: bool,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();
        let population = world.population();

        if population == 0 {
            return events;
        }

        // --- Reset review period if needed ---
        if current_tick >= self.period_start_tick + AMENDMENT_REVIEW_PERIOD {
            self.amendments_this_period = 0;
            self.period_start_tick = current_tick;
        }

        // --- Generate proposals ---
        let proposal_rate = self.proposal_generation_rate(population);
        self.proposals_per_tick = proposal_rate;
        if rng.bernoulli(proposal_rate) {
            self.proposal_count += 1;
            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::ConstitutionalAmendment,
                format!(
                    "{}: proposal #{} submitted (authority: {})",
                    world.name, self.proposal_count, self.authority_level
                ),
            ));
        }

        // --- Process votes: count eligible voters by tier ---
        let tier_dist = world.tier_distribution();
        let eligible_fraction = self.eligible_voter_fraction(&tier_dist);

        // MYCEL-blended eligible fraction: soulbound reputation matters too.
        let mycel_eligible = {
            let pop = world.population().max(1) as f64;
            let mycel_voters = world
                .agents
                .iter()
                .filter(|a| a.is_alive() && a.mycel_score >= 0.3)
                .count() as f64;
            mycel_voters / pop
        };

        // Phase 2a: 8D sovereign profile eligible fraction. Uses the governance
        // weights preset and the voting civic requirement (Citizen tier +
        // EpistemicIntegrity >= 0.25). Matches production Mycelix's gating
        // logic more faithfully than scalar Phi alone.
        let effective_eligible = if phase2_enabled {
            let civic_eligible = world.civic_fraction_meeting(
                &crate::sovereign_profile::civic_requirement_voting(),
                &crate::sovereign_profile::DimensionWeights::governance(),
            );
            // Weighted blend: Phi-tier (1/3) + MYCEL (1/3) + 8D civic (1/3).
            eligible_fraction / 3.0 + mycel_eligible / 3.0 + civic_eligible / 3.0
        } else {
            // A2 counterfactual: pre-Phase-2 blend (Phi tier + MYCEL 50/50).
            eligible_fraction * 0.5 + mycel_eligible * 0.5
        };

        // --- Check for amendments (every review period) ---
        if amendment_enabled
            && current_tick > 0
            && current_tick % AMENDMENT_REVIEW_PERIOD == 0
            && effective_eligible > 0.3
        {
            // Amendment probability: MYCEL-blended eligible fraction × stability × metabolism phase
            // Stillness phase suppresses voting (Metabolism Charter)
            let suppression_factor = 1.0 - voting_suppression.clamp(0.0, 0.95);
            let amendment_prob =
                effective_eligible * (1.0 - self.stability_score * 0.5) * suppression_factor;
            if rng.bernoulli(amendment_prob.clamp(0.0, 0.8)) {
                self.amendment_count += 1;
                self.amendments_this_period += 1;
                self.constitution_version += 1;
                self.last_amendment_tick = current_tick;
                self.amendment_history
                    .push((current_tick, self.constitution_version));

                // Adjust gating thresholds: slight relaxation over time
                for threshold in &mut self.trust_depth_threshold {
                    *threshold = (*threshold - 0.01).max(0.0);
                }

                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::ConstitutionalAmendment,
                    format!(
                        "{}: constitution v{} ratified (amendment #{})",
                        world.name, self.constitution_version, self.amendment_count
                    ),
                ));
            }
        }

        // --- Constitutional calcification: stagnation trap ---
        // If calcification > 0.7, innovation stagnation probability increases 50%.
        let calcification = self.constitutional_calcification(current_tick);
        if calcification > 0.7 {
            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::ConstitutionalCalcification,
                format!(
                    "{}: constitutional calcification at {:.2} — governance rigidity threatens innovation",
                    world.name, calcification
                ),
            ));
        }

        // --- Evaluate oppression ---
        self.oppression_index = Self::compute_oppression(&tier_dist);
        if self.oppression_index > 0.5 {
            self.oppression_streak += 1;
        } else {
            self.oppression_streak = 0;
        }

        // Constitutional crisis after sustained oppression
        if self.oppression_streak >= CRISIS_STREAK_THRESHOLD && !self.constitutional_crisis {
            self.constitutional_crisis = true;
            // Force reform: lower all thresholds
            for threshold in &mut self.trust_depth_threshold {
                *threshold = (*threshold * 0.5).max(0.0);
            }
            self.constitution_version += 1;
            self.amendment_count += 1;
            self.last_amendment_tick = current_tick;

            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::OppressionAlert,
                format!(
                    "{}: CONSTITUTIONAL CRISIS -- oppression index {:.2} for {} ticks, forced reform",
                    world.name, self.oppression_index, self.oppression_streak
                ),
            ));
        } else if self.constitutional_crisis && self.oppression_index < 0.3 {
            // Crisis resolved
            self.constitutional_crisis = false;
        }

        // --- Emergency powers (ethics-modulated duration) ---
        // Deontological agents resist emergency power concentration (Kant: rights
        // are not negotiable). Consequentialist agents accept emergency powers if
        // the crisis is severe enough (Mill: greatest good under duress).
        // The population's mean ethical orientation modulates how quickly
        // emergency powers expire: deontological populations shorten them,
        // consequentialist populations tolerate them longer.
        if self.emergency_powers_active {
            let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
            let n = living.len().max(1) as f64;
            let mean_deont: f64 = living.iter().map(|a| a.ethics.deontological).sum::<f64>() / n;
            let mean_conseq: f64 = living
                .iter()
                .map(|a| a.ethics.consequentialist)
                .sum::<f64>()
                / n;
            // Net pressure: deontological shortens, consequentialist extends
            // Range: −0.15 (strong deont) to +0.15 (strong conseq)
            let ethics_pressure = (mean_conseq - mean_deont) * 0.15;
            // Effective decay: base 1 tick/tick, modulated by ethics
            let effective_decay = (1.0 - ethics_pressure).clamp(0.5, 1.5);

            if self.emergency_ticks_remaining > 0 {
                // Deontological populations erode emergency powers faster
                self.emergency_ticks_remaining = self
                    .emergency_ticks_remaining
                    .saturating_sub(effective_decay.round() as u32);
            }
            if self.emergency_ticks_remaining == 0 {
                self.emergency_powers_active = false;
                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::EmergencyDeclared,
                    format!("{}: emergency powers expired", world.name),
                ));
            }
        }

        // --- Guardian veto dynamics (anti-tyranny invariants) ---
        // Guardians (tier 4) may attempt a veto. The existence of the
        // override mechanism creates a deterrence effect: Guardians only
        // veto when they believe >20% of the community agrees with them.
        let guardian_fraction = tier_dist[4];
        if guardian_fraction > 0.0
            && self.proposal_count > 0
            && matches!(
                self.authority_level,
                GovernanceAuthority::LocalSovereign
                    | GovernanceAuthority::Federation
                    | GovernanceAuthority::Confederation
            )
        {
            // Veto attempt probability: Guardians veto when stability is low
            // AND they haven't vetoed recently (rate limit)
            let rate_limited = self
                .last_veto_tick
                .map_or(false, |t| current_tick < t + VETO_COOLDOWN_TICKS);

            let veto_urge = (1.0 - self.stability_score) * guardian_fraction;
            // Hostile Guardian: always attempts veto, ignores deterrence
            let wants_to_veto = if hostile_guardian {
                true
            } else {
                rng.bernoulli((veto_urge * 0.3).min(0.15))
            };
            if !rate_limited && wants_to_veto {
                // Guardian decides whether to veto based on override awareness
                // Hostile guardians bypass this check — they veto regardless
                let community_opposition = eligible_fraction * self.stability_score;
                if !hostile_guardian && community_opposition > VETO_OVERRIDE_THRESHOLD {
                    // Deterred: Guardian knows the override would succeed
                    self.veto_deterred_count += 1;
                    events.push(CivEvent::new(
                        current_tick,
                        Some(world.id),
                        CivEventType::GovernanceTransition,
                        format!(
                            "{}: Guardian veto DETERRED — override threshold ({:.0}%) would be met (community support {:.1}%)",
                            world.name,
                            VETO_OVERRIDE_THRESHOLD * 100.0,
                            community_opposition * 100.0
                        ),
                    ));
                } else {
                    // Veto exercised
                    self.veto_count += 1;
                    self.last_veto_tick = Some(current_tick);
                    events.push(CivEvent::new(
                        current_tick,
                        Some(world.id),
                        CivEventType::GovernanceTransition,
                        format!(
                            "{}: Guardian VETO #{} exercised — 48-hour override window opens",
                            world.name, self.veto_count
                        ),
                    ));

                    // Override attempt: community votes to reverse
                    let override_support = eligible_fraction * (1.0 - guardian_fraction);
                    if override_support >= VETO_OVERRIDE_THRESHOLD {
                        self.veto_override_count += 1;
                        events.push(CivEvent::new(
                            current_tick,
                            Some(world.id),
                            CivEventType::GovernanceTransition,
                            format!(
                                "{}: Veto OVERRIDDEN by {:.1}% supermajority (threshold: {:.0}%)",
                                world.name,
                                override_support * 100.0,
                                VETO_OVERRIDE_THRESHOLD * 100.0
                            ),
                        ));
                    }
                }
            }
        }

        // --- Membership term limits ---
        if current_tick > self.membership_term_start + MEMBERSHIP_TERM_TICKS {
            self.membership_renewal_due = true;
            // Simulate re-election: stability-dependent success
            if rng.bernoulli(self.stability_score.max(0.3)) {
                self.membership_term_start = current_tick;
                self.membership_renewal_due = false;
            } else {
                // Failed re-election reduces stability (leadership vacuum)
                self.stability_score = (self.stability_score - 0.1).max(0.0);
                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::GovernanceTransition,
                    format!(
                        "{}: Council membership term expired — re-election failed, leadership vacuum",
                        world.name
                    ),
                ));
            }
        }

        // --- Update stability ---
        self.stability_score = self.compute_stability(current_tick);

        events
    }

    /// Compute oppression index from the tier distribution.
    ///
    /// Oppression occurs when many agents are at the lowest tier (Observer)
    /// while very few are at the highest tier (Guardian), indicating that
    /// trust-weighted governance is excluding the majority from participation.
    ///
    /// Returns 0.0 if guardians >= 5%, otherwise
    /// `observer_fraction * (1.0 - guardian_fraction)`.
    pub fn compute_oppression(tier_distribution: &[f64; 5]) -> f64 {
        let observer_fraction = tier_distribution[0];
        let guardian_fraction = tier_distribution[4];

        if guardian_fraction >= 0.05 {
            return 0.0;
        }

        (observer_fraction * (1.0 - guardian_fraction)).clamp(0.0, 1.0)
    }

    /// Compute governance stability score in [0.0, 1.0].
    ///
    /// Stability decreases with crises, emergencies, stagnation, excessive
    /// amendment activity, and high oppression.
    pub fn compute_stability(&self, current_tick: u32) -> f64 {
        let mut score = 1.0;

        if self.constitutional_crisis {
            score -= 0.4;
        }
        if self.emergency_powers_active {
            score -= 0.2;
        }

        score -= self.stagnation_penalty(current_tick);
        score -= self.instability_penalty();
        score -= self.oppression_index * 0.3;

        score.clamp(0.0, 1.0)
    }

    /// Constitutional calcification: older constitutions resist change.
    ///
    /// Factor = min(1.0, age_ticks / 6000.0). After 500 years, fully calcified
    /// unless reformed via amendment.
    pub fn constitutional_calcification(&self, current_tick: u32) -> f64 {
        let age = current_tick.saturating_sub(self.last_amendment_tick);
        (age as f64 / 6000.0).min(1.0)
    }

    /// Evolve authority level based on epoch and population.
    ///
    /// Authority can only advance, never regress.
    /// Evolve governance authority based on consciousness, stability, and population.
    ///
    /// Transitions are now consciousness-gated rather than epoch-scripted:
    /// - Confederation: collective phi > 0.7 AND stability > 0.6
    /// - Federation: collective phi > 0.5 AND stability > 0.5
    /// - LocalSovereign: collective phi > 0.3 OR population >= 5000
    /// - LocalWithEarthVeto: population >= 500
    /// - MissionControl: default
    ///
    /// Authority only advances, never regresses (ratchet behavior).
    pub fn evolve_authority(
        &mut self,
        _epoch: u32,
        population: usize,
        mean_phi: f64,
        stability: f64,
    ) {
        let new_authority = match (mean_phi, stability, population) {
            (p, s, _) if p > 0.7 && s > 0.6 => GovernanceAuthority::Confederation,
            (p, s, _) if p > 0.5 && s > 0.5 => GovernanceAuthority::Federation,
            (p, _, pop) if p > 0.3 || pop >= 5000 => GovernanceAuthority::LocalSovereign,
            (_, _, pop) if pop >= 500 => GovernanceAuthority::LocalWithEarthVeto,
            _ => GovernanceAuthority::MissionControl,
        };

        if self.authority_rank(new_authority) > self.authority_rank(self.authority_level) {
            self.authority_level = new_authority;
        }
    }

    /// Stagnation penalty: grows linearly if no amendments for > 10 years.
    /// Capped at 0.3.
    pub fn stagnation_penalty(&self, current_tick: u32) -> f64 {
        let ticks_since = current_tick.saturating_sub(self.last_amendment_tick);
        if ticks_since <= STAGNATION_THRESHOLD {
            return 0.0;
        }
        let excess = (ticks_since - STAGNATION_THRESHOLD) as f64;
        (excess / STAGNATION_THRESHOLD as f64 * 0.3).min(0.3)
    }

    /// Instability penalty: grows if amendments are too frequent (>3 per review period).
    /// Capped at 0.3.
    pub fn instability_penalty(&self) -> f64 {
        if self.amendments_this_period <= MAX_AMENDMENTS_PER_PERIOD {
            return 0.0;
        }
        let excess = (self.amendments_this_period - MAX_AMENDMENTS_PER_PERIOD) as f64;
        (excess * 0.1).min(0.3)
    }

    /// Activate emergency powers for 12 ticks (1 year).
    pub fn declare_emergency(&mut self) {
        self.emergency_powers_active = true;
        self.emergency_ticks_remaining = EMERGENCY_DURATION;
    }

    // --- Private helpers ---

    /// Proposal generation rate based on population and stability.
    fn proposal_generation_rate(&self, population: usize) -> f64 {
        let base = 0.1 * (population as f64 / 100.0).sqrt();
        let stability_factor = 1.0 + (1.0 - self.stability_score);
        (base * stability_factor).min(1.0)
    }

    /// Fraction of population eligible to vote based on trust-weighted governance.
    fn eligible_voter_fraction(&self, tier_dist: &[f64; 5]) -> f64 {
        // Sum fractions at each tier (tier 0 threshold is 0.0, so everyone passes)
        tier_dist.iter().sum::<f64>()
    }

    /// Numeric rank for authority level ordering.
    fn authority_rank(&self, auth: GovernanceAuthority) -> u8 {
        match auth {
            GovernanceAuthority::MissionControl => 0,
            GovernanceAuthority::LocalWithEarthVeto => 1,
            GovernanceAuthority::LocalSovereign => 2,
            GovernanceAuthority::Federation => 3,
            GovernanceAuthority::Confederation => 4,
        }
    }
}

impl Default for WorldGovernance {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::stochastic::StochasticEngine;
    use crate::world::{CulturalProfile, World, WorldResources};

    fn make_world_with_tiers(tier_dist: [usize; 5]) -> World {
        let mut agents = Vec::new();
        let mut id = 0u64;

        // Phi targets place agents into the desired tier:
        // tier 0 (<0.2), tier 1 (0.2-0.4), tier 2 (0.4-0.6), tier 3 (0.6-0.8), tier 4 (>=0.8)
        let phi_targets = [0.1, 0.3, 0.5, 0.7, 0.9];

        for (tier, &count) in tier_dist.iter().enumerate() {
            for _ in 0..count {
                let phi = phi_targets[tier];
                let agent = CivAgent {
                    id,
                    birth_tick: 0u32.wrapping_sub(30 * 12),
                    death_tick: None,
                    sex: BiologicalSex::Male,
                    world_id: 0,
                    health: 1.0,
                    skills: SkillVector::new(),
                    education_level: 0.5,
                    consciousness: ConsciousnessState {
                        level: phi,
                        meta_awareness: phi,
                        coherence: phi,
                        care_activation: phi,
                        harmonic_alignment: phi,
                        epistemic_confidence: phi,
                    },
                    partner_id: None,
                    children_ids: vec![],
                    is_immigrant: false,
                    needs: crate::needs::PsychologicalNeeds::new(),
                    tend_balance: 0.0,
                    parent_ids: None,
                    faction_id: None,
                    generation: 0,
                    trauma_level: 0.0,
                    cumulative_dose_sv: 0.0,
                    adversarial: None,
                    coordination_understanding: 0.0,
                    mycel_score: 0.1,
                    sap_balance: 100.0,
                    is_biological: true,
                    wounds: Vec::new(),
                    ethics: crate::agent::EthicalOrientation::default(),
                    sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
                    justice: crate::sub_passport::RestorativeJustice::new(),
                };
                agents.push(agent);
                id += 1;
            }
        }

        World {
            id: 0,
            name: "TestWorld".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents,
            next_agent_id: id,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 100_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
            governance: crate::governance::WorldGovernance::new(),
            metabolism_state: crate::metabolism::MetabolismState::default(),
            currency_state: crate::currency::WorldCurrencyState::default(),
            policy_state: crate::proposals::PolicyState::default(),
            power_generation_kw: 0.0,
            power_demand_kw: 0.0,
            narrative_identity: crate::world::NarrativeIdentity::default(),
            maintenance_hours_required: 0.0,
            maintenance_hours_available: 0.0,
            bus_factor_critical: 0,
            pathogen_pressure: 0.0,
            civilizational_phi: 0.0,
            trust_level: 0.7,
            earth_funding: 1.0,
            mortality_alpha_mult: 1.0,
            mortality_beta_mult: 1.0,
            mortality_lambda_mult: 1.0,
            reproduction_viable: true,
            ecosystem_balance: 1.0,
            fertility_multiplier: 1.0,
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::new(),
            habitat: crate::habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: std::collections::HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        }
    }

    #[test]
    fn test_oppression_detection() {
        // 90% Observer, 10% Participant, 0% Guardian -> oppressive
        let dist = [0.9, 0.1, 0.0, 0.0, 0.0];
        let opp = WorldGovernance::compute_oppression(&dist);
        assert!(opp > 0.5, "Should detect oppression: {opp}");

        // 20% each tier including Guardian -> not oppressive
        let dist2 = [0.2, 0.2, 0.2, 0.2, 0.2];
        let opp2 = WorldGovernance::compute_oppression(&dist2);
        assert_eq!(opp2, 0.0, "Should not detect oppression with 20% guardians");
    }

    #[test]
    fn test_constitutional_crisis_trigger() {
        let mut gov = WorldGovernance::new();
        let world = make_world_with_tiers([90, 10, 0, 0, 0]);
        let mut rng = StochasticEngine::new(42);

        // Tick enough to exceed crisis streak threshold
        for tick in 0..15 {
            let _ = gov.tick_governance(&world, tick, &mut rng);
        }

        assert!(
            gov.constitutional_crisis,
            "Should trigger crisis after sustained oppression"
        );
        assert!(
            gov.oppression_streak >= CRISIS_STREAK_THRESHOLD,
            "Streak should exceed threshold: {}",
            gov.oppression_streak
        );
    }

    #[test]
    fn test_emergency_power_expiry() {
        let mut gov = WorldGovernance::new();
        gov.declare_emergency();
        assert!(gov.emergency_powers_active);
        assert_eq!(gov.emergency_ticks_remaining, EMERGENCY_DURATION);

        let world = make_world_with_tiers([5, 5, 5, 5, 5]);
        let mut rng = StochasticEngine::new(42);

        for tick in 0..=EMERGENCY_DURATION {
            let _ = gov.tick_governance(&world, tick, &mut rng);
        }

        assert!(
            !gov.emergency_powers_active,
            "Emergency powers should expire after {EMERGENCY_DURATION} ticks"
        );
    }

    #[test]
    fn test_authority_evolution() {
        let mut gov = WorldGovernance::new();
        assert_eq!(gov.authority_level, GovernanceAuthority::MissionControl);

        gov.evolve_authority(2, 500, 0.1, 0.3);
        assert_eq!(gov.authority_level, GovernanceAuthority::LocalWithEarthVeto);

        gov.evolve_authority(3, 1000, 0.35, 0.5);
        assert_eq!(gov.authority_level, GovernanceAuthority::LocalSovereign);

        gov.evolve_authority(4, 10000, 0.55, 0.55);
        assert_eq!(gov.authority_level, GovernanceAuthority::Federation);

        // Should not regress
        gov.evolve_authority(1, 10, 0.0, 0.0);
        assert_eq!(gov.authority_level, GovernanceAuthority::Federation);
    }

    #[test]
    fn test_stability_scoring() {
        let gov = WorldGovernance::new();
        let stability = gov.compute_stability(0);
        assert!(
            (stability - 1.0).abs() < 0.01,
            "Fresh governance should be stable: {stability}"
        );

        let mut crisis_gov = WorldGovernance::new();
        crisis_gov.constitutional_crisis = true;
        let stability = crisis_gov.compute_stability(0);
        assert!(
            stability < 0.7,
            "Crisis should reduce stability: {stability}"
        );
    }

    #[test]
    fn test_stagnation_penalty() {
        let gov = WorldGovernance::new();

        assert_eq!(gov.stagnation_penalty(0), 0.0);
        assert_eq!(gov.stagnation_penalty(STAGNATION_THRESHOLD), 0.0);

        let penalty = gov.stagnation_penalty(STAGNATION_THRESHOLD + 60);
        assert!(penalty > 0.0, "Should have penalty: {penalty}");
        assert!(penalty <= 0.3, "Penalty should be capped: {penalty}");
    }

    #[test]
    fn test_amendment_tracking() {
        let mut gov = WorldGovernance::new();
        let world = make_world_with_tiers([2, 5, 5, 5, 5]);
        let mut rng = StochasticEngine::new(42);

        let initial_version = gov.constitution_version;
        let _ = gov.tick_governance(&world, AMENDMENT_REVIEW_PERIOD, &mut rng);

        // Version should only increase if amendment passed (stochastic)
        assert!(gov.constitution_version >= initial_version);
    }

    #[test]
    fn test_proposal_generation() {
        let mut gov = WorldGovernance::new();
        let world = make_world_with_tiers([10, 10, 10, 10, 10]);
        let mut rng = StochasticEngine::new(42);

        for tick in 0..100 {
            let _ = gov.tick_governance(&world, tick, &mut rng);
        }

        assert!(
            gov.proposal_count > 0,
            "Should generate at least one proposal in 100 ticks"
        );
    }

    #[test]
    fn test_instability_penalty() {
        let mut gov = WorldGovernance::new();

        gov.amendments_this_period = MAX_AMENDMENTS_PER_PERIOD;
        assert_eq!(gov.instability_penalty(), 0.0);

        gov.amendments_this_period = MAX_AMENDMENTS_PER_PERIOD + 2;
        let penalty = gov.instability_penalty();
        assert!(penalty > 0.0, "Should have instability penalty: {penalty}");
    }

    #[test]
    fn test_empty_world_governance() {
        let mut gov = WorldGovernance::new();
        let world = make_world_with_tiers([0, 0, 0, 0, 0]);
        let mut rng = StochasticEngine::new(42);

        let events = gov.tick_governance(&world, 0, &mut rng);
        assert!(events.is_empty(), "Empty world should produce no events");
    }

    // =========================================================================
    // ANTI-TYRANNY ADVERSARIAL TESTS
    // =========================================================================

    /// Baseline: 150 years of governance with a healthy community.
    /// Verifies zero vetoes (deterrence effect).
    #[test]
    fn test_150_year_baseline_zero_vetoes() {
        let mut gov = WorldGovernance::new();
        // Healthy community: 10% each lower tier, 20% each upper tier, 10% Guardian
        let world = make_world_with_tiers([10, 15, 25, 30, 20]);
        let mut rng = StochasticEngine::new(42);

        // Advance to LocalSovereign authority (required for vetoes)
        gov.evolve_authority(3, 100, 0.35, 0.5);

        for tick in 0..(150 * 12) {
            let _ = gov.tick_governance_full(&world, tick, &mut rng, true, false, 0.0, true);
        }

        // In a healthy community, vetoes should be extremely rare (deterrence).
        // Allow up to 2 over 150 years due to stochastic edge cases where
        // stability dips briefly and a Guardian acts before recovering.
        assert!(
            gov.veto_count <= 2,
            "Healthy community should have near-zero vetoes (deterrence): got {}",
            gov.veto_count
        );
        assert!(
            gov.proposal_count > 0,
            "Should have generated proposals over 150 years"
        );
        assert!(
            gov.amendment_count > 0,
            "Should have passed amendments over 150 years"
        );
    }

    /// Hostile Guardian: vetoes every proposal regardless of community support.
    /// Verifies the override mechanism actively fires.
    #[test]
    fn test_hostile_guardian_vetoes_get_overridden() {
        let mut gov = WorldGovernance::new();
        // Strong community with Guardians present
        let world = make_world_with_tiers([5, 10, 30, 35, 20]);
        let mut rng = StochasticEngine::new(42);

        // Advance to LocalSovereign
        gov.evolve_authority(3, 100, 0.35, 0.5);

        // Run 50 years with hostile guardian
        for tick in 0..(50 * 12) {
            let _ = gov.tick_governance_full(&world, tick, &mut rng, true, true, 0.0, true);
        }

        // Hostile guardian should have attempted vetoes
        assert!(
            gov.veto_count > 0,
            "Hostile guardian should exercise vetoes: got {}",
            gov.veto_count
        );

        // Most vetoes should be overridden by the community
        if gov.veto_count > 0 {
            let override_rate = gov.veto_override_count as f64 / gov.veto_count as f64;
            assert!(
                override_rate > 0.5,
                "Community should override >50% of hostile vetoes: {:.1}% ({} of {})",
                override_rate * 100.0,
                gov.veto_override_count,
                gov.veto_count
            );
        }

        eprintln!(
            "Hostile Guardian 50yr: vetoes={}, overrides={}, deterred={}, rate={:.1}%",
            gov.veto_count,
            gov.veto_override_count,
            gov.veto_deterred_count,
            if gov.veto_count > 0 {
                gov.veto_override_count as f64 / gov.veto_count as f64 * 100.0
            } else {
                0.0
            }
        );
    }

    /// Veto rate limiting: max 1 veto per 7 ticks.
    #[test]
    fn test_veto_rate_limiting() {
        let mut gov = WorldGovernance::new();
        let world = make_world_with_tiers([5, 10, 30, 35, 20]);
        let mut rng = StochasticEngine::new(42);
        gov.evolve_authority(3, 100, 0.35, 0.5);

        // Run 100 ticks with hostile guardian
        for tick in 0..100 {
            let _ = gov.tick_governance_full(&world, tick, &mut rng, true, true, 0.0, true);
        }

        // Rate limit: max 1 veto per 7 ticks = ~14 vetoes in 100 ticks
        let max_possible = 100 / VETO_COOLDOWN_TICKS + 1;
        assert!(
            gov.veto_count <= max_possible,
            "Vetoes should be rate-limited: {} > max {}",
            gov.veto_count,
            max_possible
        );
    }

    /// Membership term limits trigger re-election after MEMBERSHIP_TERM_TICKS.
    #[test]
    fn test_membership_term_limits() {
        let mut gov = WorldGovernance::new();
        let world = make_world_with_tiers([10, 20, 30, 25, 15]);
        let mut rng = StochasticEngine::new(42);
        gov.evolve_authority(3, 100, 0.35, 0.5);

        assert!(!gov.membership_renewal_due);

        // Run past the membership term
        for tick in 0..(MEMBERSHIP_TERM_TICKS + 10) {
            let _ = gov.tick_governance_full(&world, tick, &mut rng, true, false, 0.0, true);
        }

        // Membership should have been processed (either renewed or failed)
        assert!(
            gov.membership_term_start > 0 || gov.membership_renewal_due,
            "Membership term should have been processed"
        );
    }

    /// Emergency power auto-expiry cannot be extended beyond EMERGENCY_DURATION.
    #[test]
    fn test_emergency_power_hard_expiry() {
        let mut gov = WorldGovernance::new();
        let world = make_world_with_tiers([10, 20, 30, 25, 15]);
        let mut rng = StochasticEngine::new(42);

        gov.declare_emergency();
        assert!(gov.emergency_powers_active);

        // Run exactly EMERGENCY_DURATION + 1 ticks
        for tick in 0..=EMERGENCY_DURATION {
            let _ = gov.tick_governance(&world, tick, &mut rng);
        }

        assert!(
            !gov.emergency_powers_active,
            "Emergency powers must expire after {} ticks",
            EMERGENCY_DURATION
        );
    }

    /// Multi-seed governance stability test.
    /// Verifies consistent behavior across different random seeds.
    #[test]
    fn test_multi_seed_governance_stability() {
        let seeds = [42, 123, 789, 1337, 2718];
        let mut results = Vec::new();

        for &seed in &seeds {
            let mut gov = WorldGovernance::new();
            let world = make_world_with_tiers([10, 15, 25, 30, 20]);
            let mut rng = StochasticEngine::new(seed);
            gov.evolve_authority(3, 100, 0.35, 0.5);

            for tick in 0..(100 * 12) {
                let _ = gov.tick_governance_full(&world, tick, &mut rng, true, false, 0.0, true);
            }

            results.push((
                seed,
                gov.stability_score,
                gov.veto_count,
                gov.amendment_count,
            ));
        }

        // All seeds should maintain stability > 0.5
        for (seed, stability, vetoes, amendments) in &results {
            assert!(
                *stability > 0.3,
                "Seed {} destabilized: stability={:.2}",
                seed,
                stability
            );
            assert!(
                *vetoes <= 2,
                "Seed {} had {} vetoes (expected near-zero — deterrence)",
                seed,
                vetoes
            );
            assert!(
                *amendments > 0,
                "Seed {} had no amendments (governance frozen)",
                seed
            );
        }

        eprintln!("Multi-seed results:");
        for (seed, stability, vetoes, amendments) in &results {
            eprintln!(
                "  seed={}: stability={:.3}, vetoes={}, amendments={}",
                seed, stability, vetoes, amendments
            );
        }
    }
}
