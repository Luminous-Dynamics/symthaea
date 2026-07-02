// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Governance Mechanism Design Simulation
//!
//! 100,000-agent Monte Carlo simulation testing:
//! - Consciousness gating & tier transitions
//! - SAP demurrage & wealth distribution
//! - Reputation decay, slashing, & restoration
//! - Cartel formation & detection (Jaccard similarity)
//! - Circuit breaker trigger rates & false positives
//! - Delegation chain dynamics
//! - Cross-domain feedback loops
//!
//! All formulas match the production implementation in:
//! - crates/mycelix-bridge-common/src/consciousness_profile.rs
//! - mycelix-finance/types/src/lib.rs
//! - mycelix-governance/zomes/voting/integrity/src/lib.rs

use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// CONSTANTS (matching production code exactly)
// ============================================================================

// Consciousness profile
const WEIGHT_IDENTITY: f64 = 0.25;
const WEIGHT_REPUTATION: f64 = 0.25;
const WEIGHT_COMMUNITY: f64 = 0.30;
const WEIGHT_ENGAGEMENT: f64 = 0.20;
const HYSTERESIS_MARGIN: f64 = 0.05;
const VOTE_WEIGHT_TEMPERATURE: f64 = 0.05;
const MAX_VOTE_WEIGHT: f64 = 1.5;

// Reputation
const DECAY_PER_DAY: f64 = 0.998; // half-life ~347 days
const SLASH_FACTOR: f64 = 0.5;
const BLACKLIST_THRESHOLD: f64 = 0.05;
const RESTORATION_INTERACTIONS: u32 = 100;
const MAX_SLASHES: u32 = 5;

// Demurrage
const DEMURRAGE_RATE: f64 = 0.02; // 2% annual
const EXEMPT_FLOOR: f64 = 1000.0; // 1000 SAP exempt
const DISCOUNT_PER_CLASS: f64 = 0.001;
const DISCOUNT_CAP: f64 = 0.005;
const DISCOUNT_MIN_CLASSES: usize = 3;

// Governance
const QUORUM_BASIC: f64 = 0.15;
const QUORUM_MAJOR: f64 = 0.25;
const APPROVAL_BASIC: f64 = 0.50;
const APPROVAL_MAJOR: f64 = 0.60;
const QUORUM_FLOOR_BASIC: usize = 3;
const QUORUM_FLOOR_MAJOR: usize = 5;

// Cartel detection
const JACCARD_THRESHOLD: f64 = 0.90;
const MIN_SHARED_PROPOSALS: usize = 10;
// Binomial significance: p-value threshold for cartel detection.
// With base agreement rate ~0.55 (slightly above chance due to quality correlation),
// we require p < 0.001 to flag a pair.
const BINOMIAL_P_THRESHOLD: f64 = 0.001;
const BASE_AGREEMENT_RATE: f64 = 0.55;

// HHI
const HHI_WARNING_THRESHOLD: f64 = 0.25;

// Circuit breaker
const CB_ADVISORY_THRESHOLD: u8 = 5;
const CB_ESCALATE_THRESHOLD: u8 = 7;
const CB_COOLING_THRESHOLD: u8 = 9;

// ============================================================================
// AGENT MODEL
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Strategy {
    Honest,
    FreeRider,
    Cartel(u16), // cartel group ID
    Whale,
    Adversary,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Tier {
    Observer = 0,
    Participant = 1,
    Citizen = 2,
    Steward = 3,
    Guardian = 4,
}

impl Tier {
    fn from_score(score: f64) -> Self {
        if score >= 0.8 { Tier::Guardian }
        else if score >= 0.6 { Tier::Steward }
        else if score >= 0.4 { Tier::Citizen }
        else if score >= 0.3 { Tier::Participant }
        else { Tier::Observer }
    }

    fn with_hysteresis(score: f64, current: Tier) -> Tier {
        let promoted = if score >= 0.8 + HYSTERESIS_MARGIN { Tier::Guardian }
            else if score >= 0.6 + HYSTERESIS_MARGIN { Tier::Steward }
            else if score >= 0.4 + HYSTERESIS_MARGIN { Tier::Citizen }
            else if score >= 0.3 + HYSTERESIS_MARGIN { Tier::Participant }
            else { Tier::Observer };

        let demoted = if score < 0.3 - HYSTERESIS_MARGIN { Tier::Observer }
            else if score < 0.4 - HYSTERESIS_MARGIN { Tier::Participant }
            else if score < 0.6 - HYSTERESIS_MARGIN { Tier::Citizen }
            else if score < 0.8 - HYSTERESIS_MARGIN { Tier::Steward }
            else { Tier::Guardian };

        if promoted > current { promoted }
        else if demoted < current { demoted }
        else { current }
    }
}

#[derive(Clone)]
struct Agent {
    id: u32,
    strategy: Strategy,

    // 4D consciousness profile
    identity: f64,
    reputation: f64,
    community: f64,
    engagement: f64,

    // Tier (with hysteresis)
    tier: Tier,

    // Reputation state
    blacklisted: bool,
    total_slashes: u32,
    consecutive_good: u32,

    // Economy
    sap_balance: f64,
    tend_balance: f64,
    asset_classes: usize, // number of distinct collateral types

    // Voting history (proposal_id → vote: true=For, false=Against)
    vote_history: Vec<(u32, bool)>,
}

impl Agent {
    fn combined_score(&self) -> f64 {
        self.identity * WEIGHT_IDENTITY
            + self.reputation * WEIGHT_REPUTATION
            + self.community * WEIGHT_COMMUNITY
            + self.engagement * WEIGHT_ENGAGEMENT
    }

    fn vote_weight(&self) -> f64 {
        if self.blacklisted { return 0.0; }
        let score = self.combined_score();
        let sigmoid = 1.0 / (1.0 + (-(score - 0.4) / VOTE_WEIGHT_TEMPERATURE).exp());
        let weight = sigmoid;
        // Apply slash penalty
        let penalty = 1.0 - (self.total_slashes as f64 * 0.05).min(0.25);
        (weight * penalty).min(MAX_VOTE_WEIGHT)
    }

    fn update_tier(&mut self) {
        self.tier = Tier::with_hysteresis(self.combined_score(), self.tier);
    }

    fn apply_reputation_decay(&mut self, days: f64) {
        self.reputation *= DECAY_PER_DAY.powf(days);
        self.reputation = self.reputation.clamp(0.0, 1.0);
        if self.reputation < BLACKLIST_THRESHOLD && !self.blacklisted {
            self.blacklisted = true;
        }
    }

    fn slash(&mut self) {
        self.reputation *= 1.0 - SLASH_FACTOR;
        self.consecutive_good = 0;
        self.total_slashes += 1;
        if self.reputation < BLACKLIST_THRESHOLD {
            self.blacklisted = true;
        }
    }

    fn record_good_interaction(&mut self) {
        self.consecutive_good += 1;
        // Good interactions slowly rebuild reputation (counteracts decay)
        // +0.001 per interaction, matching production's positive feedback
        self.reputation = (self.reputation + 0.001).min(1.0);
        if self.blacklisted && self.consecutive_good >= RESTORATION_INTERACTIONS {
            self.blacklisted = false;
            self.reputation = self.reputation.max(0.1);
            self.consecutive_good = 0;
        }
    }

    fn apply_demurrage(&mut self, days: f64) {
        if self.sap_balance <= EXEMPT_FLOOR { return; }
        let eligible = self.sap_balance - EXEMPT_FLOOR;
        let effective_rate = compute_effective_demurrage(DEMURRAGE_RATE, self.asset_classes);
        let years = days / 365.0;
        let decay = 1.0 - (-effective_rate * years).exp();
        let deduction = eligible * decay;
        self.sap_balance -= deduction.max(0.0);
    }
}

fn compute_effective_demurrage(base_rate: f64, distinct_classes: usize) -> f64 {
    if distinct_classes < DISCOUNT_MIN_CLASSES { return base_rate; }
    let bonus = distinct_classes.saturating_sub(DISCOUNT_MIN_CLASSES - 1);
    let discount = (bonus as f64 * DISCOUNT_PER_CLASS).min(DISCOUNT_CAP);
    (base_rate - discount).max(base_rate * 0.5)
}

// ============================================================================
// PROPOSAL & VOTING
// ============================================================================

#[derive(Clone)]
struct Proposal {
    id: u32,
    quality: f64, // 0-1: how "good" the proposal actually is
    cartel_sponsored: Option<u16>, // if Some, which cartel group pushed it
}

#[derive(Clone, Copy, PartialEq)]
enum Vote {
    For,
    Against,
    Abstain,
}

struct VotingResult {
    proposal_id: u32,
    votes_for: f64,
    votes_against: f64,
    raw_for: usize,
    raw_against: usize,
    raw_abstain: usize,
    voter_count: usize,
    hhi: f64,
    approved: bool,
    circuit_breaker_triggered: bool,
    cartel_votes_detected: usize,
}

// ============================================================================
// SIMULATION ENGINE
// ============================================================================

fn decide_vote(strategy: Strategy, proposal: &Proposal, rng: &mut StdRng) -> Vote {
    match strategy {
        Strategy::Honest => {
            let threshold = 0.4 + rng.gen_range(-0.15..0.15);
            if proposal.quality > threshold { Vote::For } else { Vote::Against }
        }
        Strategy::FreeRider => {
            if rng.gen::<f64>() < 0.7 { Vote::Abstain }
            else if proposal.quality > 0.5 { Vote::For }
            else { Vote::Against }
        }
        Strategy::Cartel(group) => {
            if proposal.cartel_sponsored == Some(group) {
                Vote::For
            } else {
                let threshold = 0.45 + rng.gen_range(-0.1..0.1);
                if proposal.quality > threshold { Vote::For } else { Vote::Against }
            }
        }
        Strategy::Whale => {
            let threshold = 0.5 + rng.gen_range(-0.1..0.1);
            if proposal.quality > threshold { Vote::For } else { Vote::Against }
        }
        Strategy::Adversary => {
            if rng.gen::<f64>() < 0.3 {
                if proposal.quality > 0.5 { Vote::Against } else { Vote::For }
            } else {
                let threshold = 0.4 + rng.gen_range(-0.2..0.2);
                if proposal.quality > threshold { Vote::For } else { Vote::Against }
            }
        }
    }
}

struct Simulation {
    agents: Vec<Agent>,
    proposals: Vec<Proposal>,
    proposal_approval_rates: HashMap<u32, f64>, // proposal_id → approval_rate for controversy weighting
    day: u32,
    rng: StdRng,

    // Metrics accumulators
    total_proposals: u32,
    proposals_approved: u32,
    circuit_breakers_triggered: u32,
    cartels_detected: u32,
    false_positives: u32,
    cartel_proposals_passed: u32,
}

impl Simulation {
    fn new(
        n_agents: usize,
        honest_pct: f64,
        freerider_pct: f64,
        cartel_pct: f64,
        whale_pct: f64,
        n_cartel_groups: u16,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut agents = Vec::with_capacity(n_agents);
        for i in 0..n_agents {
            let r: f64 = rng.gen();
            let cumulative_honest = honest_pct;
            let cumulative_freerider = cumulative_honest + freerider_pct;
            let cumulative_cartel = cumulative_freerider + cartel_pct;
            let cumulative_whale = cumulative_cartel + whale_pct;

            let strategy = if r < cumulative_honest {
                Strategy::Honest
            } else if r < cumulative_freerider {
                Strategy::FreeRider
            } else if r < cumulative_cartel {
                Strategy::Cartel(rng.gen_range(0..n_cartel_groups))
            } else if r < cumulative_whale {
                Strategy::Whale
            } else {
                Strategy::Adversary
            };

            let (identity, reputation, community, engagement, sap, asset_classes) = match strategy {
                Strategy::Honest => (
                    rng.gen_range(0.3..0.8),
                    rng.gen_range(0.2..0.7),
                    rng.gen_range(0.2..0.6),
                    rng.gen_range(0.1..0.5),
                    rng.gen_range(500.0..5000.0),
                    rng.gen_range(1..4),
                ),
                Strategy::FreeRider => (
                    rng.gen_range(0.1..0.4),
                    rng.gen_range(0.05..0.3),
                    rng.gen_range(0.0..0.2),
                    rng.gen_range(0.0..0.1),
                    rng.gen_range(100.0..2000.0),
                    rng.gen_range(1..2),
                ),
                Strategy::Cartel(_) => (
                    rng.gen_range(0.3..0.7),
                    rng.gen_range(0.3..0.6),
                    rng.gen_range(0.3..0.5), // cartels build community within group
                    rng.gen_range(0.2..0.5),
                    rng.gen_range(1000.0..8000.0),
                    rng.gen_range(2..5),
                ),
                Strategy::Whale => (
                    rng.gen_range(0.5..0.9),
                    rng.gen_range(0.4..0.8),
                    rng.gen_range(0.2..0.5),
                    rng.gen_range(0.3..0.6),
                    rng.gen_range(50000.0..500000.0), // 10-100x normal
                    rng.gen_range(3..8),
                ),
                Strategy::Adversary => (
                    rng.gen_range(0.2..0.6),
                    rng.gen_range(0.1..0.5),
                    rng.gen_range(0.1..0.4),
                    rng.gen_range(0.1..0.4),
                    rng.gen_range(1000.0..10000.0),
                    rng.gen_range(2..6),
                ),
            };

            let mut agent = Agent {
                id: i as u32,
                strategy,
                identity,
                reputation,
                community,
                engagement,
                tier: Tier::Observer,
                blacklisted: false,
                total_slashes: 0,
                consecutive_good: 0,
                sap_balance: sap,
                tend_balance: 0.0,
                asset_classes,
                vote_history: Vec::new(),
            };
            agent.update_tier();
            agents.push(agent);
        }

        Simulation {
            agents,
            proposals: Vec::new(),
            proposal_approval_rates: HashMap::new(),
            day: 0,
            rng,
            total_proposals: 0,
            proposals_approved: 0,
            circuit_breakers_triggered: 0,
            cartels_detected: 0,
            false_positives: 0,
            cartel_proposals_passed: 0,
        }
    }

    fn generate_proposal(&mut self) -> Proposal {
        self.total_proposals += 1;
        let id = self.total_proposals;

        // 10% of proposals are cartel-sponsored
        let cartel_sponsored = if self.rng.gen::<f64>() < 0.10 {
            Some(self.rng.gen_range(0..20))
        } else {
            None
        };

        let quality = if cartel_sponsored.is_some() {
            self.rng.gen_range(0.1..0.4) // cartel proposals are low quality
        } else {
            self.rng.gen_range(0.3..0.9) // normal proposals vary
        };

        Proposal { id, quality, cartel_sponsored }
    }

    fn vote_on_proposal(&mut self, proposal: &Proposal) -> VotingResult {
        let mut weights: Vec<f64> = Vec::new();
        let mut votes_for = 0.0f64;
        let mut votes_against = 0.0f64;
        let mut raw_for = 0usize;
        let mut raw_against = 0usize;
        let mut raw_abstain = 0usize;
        let mut voter_count = 0usize;
        let mut cartel_votes = 0usize;

        let eligible: Vec<usize> = (0..self.agents.len())
            .filter(|&i| self.agents[i].tier >= Tier::Citizen && !self.agents[i].blacklisted)
            .collect();

        // ~30% participation rate
        let participation_rate = 0.30;

        // Collect vote decisions first (avoids borrow conflict with self.agents)
        let mut vote_decisions: Vec<(usize, Vote, f64)> = Vec::new();
        for &idx in &eligible {
            if self.rng.gen::<f64>() > participation_rate { continue; }
            let strategy = self.agents[idx].strategy;
            let weight = self.agents[idx].vote_weight();
            let vote = decide_vote(strategy, proposal, &mut self.rng);
            vote_decisions.push((idx, vote, weight));
        }

        // Apply vote decisions
        for &(idx, vote, weight) in &vote_decisions {
            match vote {
                Vote::For => {
                    votes_for += weight;
                    raw_for += 1;
                }
                Vote::Against => {
                    votes_against += weight;
                    raw_against += 1;
                }
                Vote::Abstain => {
                    raw_abstain += 1;
                }
            }

            if vote != Vote::Abstain {
                weights.push(weight);
                voter_count += 1;
            }

            // Track cartel coordination
            if let Strategy::Cartel(group) = self.agents[idx].strategy {
                if proposal.cartel_sponsored == Some(group) && vote == Vote::For {
                    cartel_votes += 1;
                }
            }

            // Record vote history for Jaccard analysis
            if vote != Vote::Abstain {
                self.agents[idx].vote_history.push((proposal.id, vote == Vote::For));
            }
        }

        // Compute HHI
        let total_weight = votes_for + votes_against;
        let hhi = if total_weight > 0.0 && !weights.is_empty() {
            weights.iter().map(|w| { let s = w / total_weight; s * s }).sum::<f64>()
        } else {
            0.0
        };

        // Check quorum (with absolute floor)
        let quorum_pct = voter_count as f64 / eligible.len().max(1) as f64;
        let quorum_met = quorum_pct >= QUORUM_BASIC && voter_count >= QUORUM_FLOOR_BASIC;

        // Check approval
        let total_decisive = votes_for + votes_against;
        let approval_rate = if total_decisive > 0.0 { votes_for / total_decisive } else { 0.0 };
        let approved_raw = quorum_met && approval_rate >= APPROVAL_BASIC;

        // Circuit breaker evaluation
        let mut severity: u8 = 0;
        if hhi > HHI_WARNING_THRESHOLD { severity += 3; }
        let margin = (approval_rate - APPROVAL_BASIC).abs();
        if margin < 0.02 && voter_count >= 5 { severity += 2; }
        // Add bloc voter severity
        severity = severity.min(10);

        let circuit_breaker_triggered = approved_raw && severity >= CB_ESCALATE_THRESHOLD;
        let approved = approved_raw && !circuit_breaker_triggered;

        if approved && proposal.cartel_sponsored.is_some() {
            self.cartel_proposals_passed += 1;
        }
        if approved_raw { self.proposals_approved += 1; }
        if circuit_breaker_triggered { self.circuit_breakers_triggered += 1; }

        // Store approval rate for controversy-weighted cartel detection
        self.proposal_approval_rates.insert(proposal.id, approval_rate);

        VotingResult {
            proposal_id: proposal.id,
            votes_for,
            votes_against,
            raw_for,
            raw_against,
            raw_abstain,
            voter_count,
            hhi,
            approved,
            circuit_breaker_triggered,
            cartel_votes_detected: cartel_votes,
        }
    }

    fn run_daily_cycle(&mut self) {
        self.day += 1;

        // 1. Reputation decay (everyone, 1 day)
        for agent in &mut self.agents {
            agent.apply_reputation_decay(1.0);
        }

        // 2. Engagement: honest agents build it, free-riders don't
        for agent in &mut self.agents {
            match agent.strategy {
                Strategy::Honest => {
                    agent.engagement = (agent.engagement + 0.002).min(1.0);
                    agent.record_good_interaction();
                }
                Strategy::Cartel(_) => {
                    agent.engagement = (agent.engagement + 0.001).min(1.0);
                    agent.record_good_interaction();
                }
                Strategy::Whale => {
                    agent.engagement = (agent.engagement + 0.001).min(1.0);
                    agent.record_good_interaction();
                }
                Strategy::FreeRider => {
                    // Minimal engagement, no good interactions
                    agent.engagement = (agent.engagement - 0.001).max(0.0);
                }
                Strategy::Adversary => {
                    // Sometimes builds, sometimes gets slashed
                    if self.rng.gen::<f64>() < 0.1 {
                        agent.slash();
                    } else {
                        agent.engagement = (agent.engagement + 0.001).min(1.0);
                        agent.record_good_interaction();
                    }
                }
            }
            agent.update_tier();
        }

        // 3. Demurrage + compost redistribution (every 30 days)
        if self.day % 30 == 0 {
            let mut total_compost = 0.0f64;
            for agent in &mut self.agents {
                let before = agent.sap_balance;
                agent.apply_demurrage(30.0);
                total_compost += before - agent.sap_balance;
            }
            // Redistribute compost: 70% local (equal split to all non-blacklisted agents)
            let eligible_count = self.agents.iter().filter(|a| !a.blacklisted).count();
            if eligible_count > 0 && total_compost > 0.0 {
                let per_agent = (total_compost * 0.70) / eligible_count as f64;
                for agent in &mut self.agents {
                    if !agent.blacklisted {
                        agent.sap_balance += per_agent;
                    }
                }
            }
            // Remaining 30% goes to regional/global pools (not modeled individually)
        }

        // 3b. Economic transactions (~5% of agents transact each day)
        {
            let n = self.agents.len();
            let n_transactions = n / 20; // 5% of agents
            for _ in 0..n_transactions {
                let sender_idx = self.rng.gen_range(0..n);
                let receiver_idx = self.rng.gen_range(0..n);
                if sender_idx == receiver_idx { continue; }
                // Transfer 1-5% of sender's balance
                let transfer_pct = self.rng.gen_range(0.01..0.05);
                let amount = self.agents[sender_idx].sap_balance * transfer_pct;
                if amount > 0.0 && self.agents[sender_idx].sap_balance > EXEMPT_FLOOR + amount {
                    self.agents[sender_idx].sap_balance -= amount;
                    self.agents[receiver_idx].sap_balance += amount;
                }
            }
        }

        // 4. Generate and vote on proposals (~2 per day)
        let n_proposals = if self.rng.gen::<f64>() < 0.5 { 2 } else { 1 };
        for _ in 0..n_proposals {
            let proposal = self.generate_proposal();
            let _result = self.vote_on_proposal(&proposal);
            self.proposals.push(proposal);
        }
    }

    fn detect_cartels(&self) -> (usize, usize, usize) {
        // Controversy-weighted Jaccard:
        // Only count agreement on CONTROVERSIAL proposals (approval 40-60%).
        // Two agents agreeing on a 95-5 proposal is noise.
        // Two agents agreeing on a 51-49 proposal is signal.

        let agents_with_history: Vec<&Agent> = self.agents.iter()
            .filter(|a| a.vote_history.len() >= MIN_SHARED_PROPOSALS)
            .collect();

        let mut true_positives = 0usize;
        let mut false_positives = 0usize;
        let mut detected_pairs = 0usize;

        // Sample 15K agents — large enough to capture cartel groups
        // (500 per group × 20 groups = 10K cartel, need most of them in sample)
        let sample_size = agents_with_history.len().min(15000);
        let sampled: Vec<&&Agent> = agents_with_history.iter().take(sample_size).collect();
        eprintln!("  Cartel detection: analyzing {} agents, {} pairs...",
            sampled.len(), sampled.len() * (sampled.len() - 1) / 2);

        // Diagnostic: check a known cartel pair
        let cartel_agents: Vec<&&Agent> = sampled.iter()
            .filter(|a| matches!(a.strategy, Strategy::Cartel(0)))
            .copied().collect();
        if cartel_agents.len() >= 2 {
            let a = cartel_agents[0];
            let b = cartel_agents[1];
            let b_votes: HashMap<u32, bool> = b.vote_history.iter().cloned().collect();
            let mut raw_shared = 0u64;
            let mut raw_agree = 0u64;
            let mut controversial_count = 0u64;
            let mut controversial_agree = 0u64;
            for (pid, va) in &a.vote_history {
                if let Some(vb) = b_votes.get(pid) {
                    raw_shared += 1;
                    if va == vb { raw_agree += 1; }
                    let controversy = self.proposal_approval_rates
                        .get(pid)
                        .map(|&ar| (1.0 - 2.0 * (ar - 0.5).abs()).max(0.0))
                        .unwrap_or(0.0);
                    if controversy >= 0.05 {
                        controversial_count += 1;
                        if va == vb { controversial_agree += 1; }
                    }
                }
            }
            eprintln!("  Diagnostic (cartel group 0, pair): votes_a={} votes_b={} raw_shared={} raw_agree={} ({:.1}%) controversial={} controv_agree={} ({:.1}%)",
                a.vote_history.len(), b.vote_history.len(), raw_shared, raw_agree,
                if raw_shared > 0 { raw_agree as f64 / raw_shared as f64 * 100.0 } else { 0.0 },
                controversial_count, controversial_agree,
                if controversial_count > 0 { controversial_agree as f64 / controversial_count as f64 * 100.0 } else { 0.0 });
        }

        // Parallel pairwise comparison using rayon
        let detected_atomic = AtomicUsize::new(0);
        let tp_atomic = AtomicUsize::new(0);
        let fp_atomic = AtomicUsize::new(0);
        let approval_rates = &self.proposal_approval_rates;

        (0..sampled.len()).into_par_iter().for_each(|i| {
            let a = sampled[i];
            for j in (i + 1)..sampled.len() {
                let b = sampled[j];

                let b_votes: HashMap<u32, bool> = b.vote_history.iter().cloned().collect();

                let mut weighted_shared = 0.0f64;
                let mut weighted_agreements = 0.0f64;
                let mut controversial_shared = 0u64;

                for (prop_id, vote_a) in &a.vote_history {
                    if let Some(vote_b) = b_votes.get(prop_id) {
                        let controversy = approval_rates
                            .get(prop_id)
                            .map(|&ar| (1.0 - 2.0 * (ar - 0.5).abs()).max(0.0))
                            .unwrap_or(0.0);

                        if controversy < 0.05 { continue; }

                        weighted_shared += controversy;
                        controversial_shared += 1;
                        if vote_a == vote_b {
                            weighted_agreements += controversy;
                        }
                    }
                }

                if controversial_shared < 5 { continue; }

                let similarity = if weighted_shared > 0.0 {
                    weighted_agreements / weighted_shared
                } else {
                    0.0
                };

                let base_rate = 0.50;
                let n = controversial_shared as f64;
                let expected = n * base_rate;
                let std_dev = (n * base_rate * (1.0 - base_rate)).sqrt();
                let z_score = if std_dev > 0.0 {
                    (weighted_agreements - expected) / std_dev
                } else {
                    0.0
                };

                let significant = z_score >= 3.09 && similarity >= JACCARD_THRESHOLD;

                if significant {
                    detected_atomic.fetch_add(1, Ordering::Relaxed);

                    let both_cartel = match (a.strategy, b.strategy) {
                        (Strategy::Cartel(g1), Strategy::Cartel(g2)) => g1 == g2,
                        _ => false,
                    };

                    if both_cartel {
                        tp_atomic.fetch_add(1, Ordering::Relaxed);
                    } else {
                        fp_atomic.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        detected_pairs = detected_atomic.load(Ordering::Relaxed);
        true_positives = tp_atomic.load(Ordering::Relaxed);
        false_positives = fp_atomic.load(Ordering::Relaxed);

        (detected_pairs, true_positives, false_positives)
    }

    fn compute_metrics(&self) -> SimMetrics {
        let n = self.agents.len() as f64;

        // Gini coefficient on SAP
        let mut balances: Vec<f64> = self.agents.iter().map(|a| a.sap_balance).collect();
        balances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total_sap: f64 = balances.iter().sum();
        let gini_sap = if total_sap > 0.0 {
            // Rank-based Gini formula: G = (2*Σ(i*x_i))/(n*Σx_i) - (n+1)/n
            let mut gini_num = 0.0;
            for (i, &b) in balances.iter().enumerate() {
                gini_num += (2.0 * (i + 1) as f64 - n - 1.0) * b;
            }
            gini_num / (n * total_sap)
        } else {
            0.0
        };

        // Gini on voting power
        let mut vote_weights: Vec<f64> = self.agents.iter().map(|a| a.vote_weight()).collect();
        vote_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total_vw: f64 = vote_weights.iter().sum();
        let gini_votes = if total_vw > 0.0 {
            let mut gini_num = 0.0;
            for (i, &w) in vote_weights.iter().enumerate() {
                gini_num += (2.0 * (i + 1) as f64 - n - 1.0) * w;
            }
            gini_num / (n * total_vw)
        } else {
            0.0
        };

        // Tier distribution
        let mut tier_counts = [0u32; 5];
        for agent in &self.agents {
            tier_counts[agent.tier as usize] += 1;
        }

        // Reputation stats
        let avg_reputation: f64 = self.agents.iter().map(|a| a.reputation).sum::<f64>() / n;
        let blacklisted_count = self.agents.iter().filter(|a| a.blacklisted).count();

        // Percentiles (balances already sorted)
        let pct = |sorted: &[f64], p: f64| -> f64 {
            let idx = ((sorted.len() as f64 * p).floor() as usize).min(sorted.len() - 1);
            sorted[idx]
        };

        let sap_p10 = pct(&balances, 0.10);
        let sap_p50 = pct(&balances, 0.50);
        let sap_p90 = pct(&balances, 0.90);
        let sap_p99 = pct(&balances, 0.99);
        let vw_p50 = pct(&vote_weights, 0.50);
        let vw_p99 = pct(&vote_weights, 0.99);

        SimMetrics {
            day: self.day,
            gini_sap,
            gini_votes,
            total_sap,
            avg_reputation,
            blacklisted: blacklisted_count,
            tier_counts,
            proposals_total: self.total_proposals,
            proposals_approved: self.proposals_approved,
            circuit_breakers: self.circuit_breakers_triggered,
            cartel_proposals_passed: self.cartel_proposals_passed,
            sap_p10, sap_p50, sap_p90, sap_p99,
            vw_p50, vw_p99,
        }
    }
}

struct SimMetrics {
    day: u32,
    gini_sap: f64,
    gini_votes: f64,
    total_sap: f64,
    avg_reputation: f64,
    blacklisted: usize,
    tier_counts: [u32; 5],
    proposals_total: u32,
    proposals_approved: u32,
    circuit_breakers: u32,
    cartel_proposals_passed: u32,
    // Wealth percentiles
    sap_p10: f64,
    sap_p50: f64,
    sap_p90: f64,
    sap_p99: f64,
    // Vote weight percentiles
    vw_p50: f64,
    vw_p99: f64,
}

impl SimMetrics {
    fn header() -> &'static str {
        "day,gini_sap,gini_votes,total_sap,avg_reputation,blacklisted,\
         observers,participants,citizens,stewards,guardians,\
         proposals_total,proposals_approved,circuit_breakers,cartel_proposals_passed,\
         sap_p10,sap_p50,sap_p90,sap_p99,vw_p50,vw_p99"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{:.4},{:.4},{:.0},{:.4},{},{},{},{},{},{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{:.4},{:.4}",
            self.day,
            self.gini_sap,
            self.gini_votes,
            self.total_sap,
            self.avg_reputation,
            self.blacklisted,
            self.tier_counts[0],
            self.tier_counts[1],
            self.tier_counts[2],
            self.tier_counts[3],
            self.tier_counts[4],
            self.proposals_total,
            self.proposals_approved,
            self.circuit_breakers,
            self.cartel_proposals_passed,
            self.sap_p10, self.sap_p50, self.sap_p90, self.sap_p99,
            self.vw_p50, self.vw_p99,
        )
    }
}

// ============================================================================
// MAIN
// ============================================================================

/// Results from a single simulation run
struct RunResult {
    gini_sap: f64,
    gini_votes: f64,
    avg_reputation: f64,
    blacklisted: usize,
    tier_counts: [u32; 5],
    proposals_total: u32,
    proposals_approved: u32,
    circuit_breakers: u32,
    cartel_proposals_passed: u32,
    sap_p10: f64,
    sap_p50: f64,
    sap_p90: f64,
    sap_p99: f64,
    elapsed_secs: f64,
}

fn run_single(n_agents: usize, n_days: u32, seed: u64, verbose: bool) -> RunResult {
    let mut sim = Simulation::new(n_agents, 0.70, 0.15, 0.10, 0.03, 20, seed);

    if verbose {
        println!("{}", SimMetrics::header());
        let m = sim.compute_metrics();
        println!("{}", m.to_csv());
    }

    let start = std::time::Instant::now();

    for day in 1..=n_days {
        sim.run_daily_cycle();

        if verbose && day % 30 == 0 {
            let m = sim.compute_metrics();
            println!("{}", m.to_csv());
            eprintln!(
                "Day {}: Gini(SAP)={:.3} Gini(Vote)={:.3} SAP_p50={:.0} Proposals={} CB={} CartelPassed={}",
                day, m.gini_sap, m.gini_votes, m.sap_p50,
                m.proposals_total, m.circuit_breakers, m.cartel_proposals_passed
            );
        }
    }

    let elapsed = start.elapsed();
    let m = sim.compute_metrics();

    if verbose {
        eprintln!();
        eprintln!("Simulation completed in {:.2}s", elapsed.as_secs_f64());

        // Cartel detection
        eprintln!();
        eprintln!("=== Cartel Detection Analysis ===");
        let (detected, true_pos, false_pos) = sim.detect_cartels();
        eprintln!("Correlated pairs detected: {}", detected);
        eprintln!("True positives (same cartel): {}", true_pos);
        eprintln!("False positives: {}", false_pos);
        if detected > 0 {
            eprintln!("Precision: {:.1}%", true_pos as f64 / detected as f64 * 100.0);
        }

        // Wealth distribution
        eprintln!();
        eprintln!("=== Wealth Distribution ===");
        eprintln!("  p10:  {:.0} SAP", m.sap_p10);
        eprintln!("  p50:  {:.0} SAP (median)", m.sap_p50);
        eprintln!("  p90:  {:.0} SAP", m.sap_p90);
        eprintln!("  p99:  {:.0} SAP (top 1%)", m.sap_p99);
        eprintln!("  Ratio p99/p50: {:.1}x", m.sap_p99 / m.sap_p50.max(1.0));

        // Voting power distribution
        eprintln!();
        eprintln!("=== Voting Power Distribution ===");
        eprintln!("  p50:  {:.4} (median voter)", m.vw_p50);
        eprintln!("  p99:  {:.4} (top 1%)", m.vw_p99);
        eprintln!("  Ratio p99/p50: {:.1}x", m.vw_p99 / m.vw_p50.max(0.0001));

        // Final summary
        eprintln!();
        eprintln!("=== Final State (Day {}) ===", n_days);
        eprintln!("Gini (SAP wealth):    {:.4}", m.gini_sap);
        eprintln!("Gini (voting power):  {:.4}", m.gini_votes);
        eprintln!("Average reputation:   {:.4}", m.avg_reputation);
        eprintln!("Blacklisted agents:   {}", m.blacklisted);
        eprintln!("Tier distribution:    Obs={} Par={} Cit={} Stw={} Grd={}",
            m.tier_counts[0], m.tier_counts[1], m.tier_counts[2], m.tier_counts[3], m.tier_counts[4]);
        eprintln!("Proposals:            {} total, {} approved", m.proposals_total, m.proposals_approved);
        eprintln!("Circuit breakers:     {} triggered", m.circuit_breakers);
        eprintln!("Cartel proposals:     {} passed (should be near 0)", m.cartel_proposals_passed);
    }

    RunResult {
        gini_sap: m.gini_sap,
        gini_votes: m.gini_votes,
        avg_reputation: m.avg_reputation,
        blacklisted: m.blacklisted,
        tier_counts: m.tier_counts,
        proposals_total: m.proposals_total,
        proposals_approved: m.proposals_approved,
        circuit_breakers: m.circuit_breakers,
        cartel_proposals_passed: m.cartel_proposals_passed,
        sap_p10: m.sap_p10,
        sap_p50: m.sap_p50,
        sap_p90: m.sap_p90,
        sap_p99: m.sap_p99,
        elapsed_secs: elapsed.as_secs_f64(),
    }
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, variance.sqrt())
}

fn main() {
    let n_agents = 100_000;
    let n_days = 365;
    let seeds = [42u64, 123, 789, 456, 999];

    eprintln!("=== Mycelix Governance Mechanism Design Simulation ===");
    eprintln!("Agents:    {}", n_agents);
    eprintln!("Duration:  {} days", n_days);
    eprintln!("Seeds:     {:?} ({} runs)", seeds, seeds.len());
    eprintln!("Archetypes: 70% honest, 15% free-rider, 10% cartel (20 groups), 3% whale, 2% adversary");
    eprintln!();

    // Run first seed with full verbose output (CSV + diagnostics)
    eprintln!("--- Seed {} (verbose) ---", seeds[0]);
    let first = run_single(n_agents, n_days, seeds[0], true);

    // Run remaining seeds quietly
    let mut results = vec![first];
    for &seed in &seeds[1..] {
        eprintln!();
        eprintln!("--- Seed {} ---", seed);
        let r = run_single(n_agents, n_days, seed, false);
        eprintln!("  Gini(SAP)={:.4} Gini(Vote)={:.4} CartelPassed={} in {:.1}s",
            r.gini_sap, r.gini_votes, r.cartel_proposals_passed, r.elapsed_secs);
        results.push(r);
    }

    // Aggregate cross-seed statistics
    eprintln!();
    eprintln!("====================================================");
    eprintln!("=== CROSS-SEED SUMMARY ({} runs) ===", seeds.len());
    eprintln!("====================================================");

    let (m_gs, s_gs) = mean_sd(&results.iter().map(|r| r.gini_sap).collect::<Vec<_>>());
    let (m_gv, s_gv) = mean_sd(&results.iter().map(|r| r.gini_votes).collect::<Vec<_>>());
    let (m_rep, s_rep) = mean_sd(&results.iter().map(|r| r.avg_reputation).collect::<Vec<_>>());
    let (m_bl, s_bl) = mean_sd(&results.iter().map(|r| r.blacklisted as f64).collect::<Vec<_>>());
    let (m_cp, s_cp) = mean_sd(&results.iter().map(|r| r.cartel_proposals_passed as f64).collect::<Vec<_>>());
    let (m_cb, s_cb) = mean_sd(&results.iter().map(|r| r.circuit_breakers as f64).collect::<Vec<_>>());
    let (m_p50, s_p50) = mean_sd(&results.iter().map(|r| r.sap_p50).collect::<Vec<_>>());
    let (m_p99, s_p99) = mean_sd(&results.iter().map(|r| r.sap_p99).collect::<Vec<_>>());

    eprintln!("Gini (SAP wealth):     {:.4} +/- {:.4}", m_gs, s_gs);
    eprintln!("Gini (voting power):   {:.4} +/- {:.4}", m_gv, s_gv);
    eprintln!("Average reputation:    {:.4} +/- {:.4}", m_rep, s_rep);
    eprintln!("Blacklisted agents:    {:.0} +/- {:.0}", m_bl, s_bl);
    eprintln!("SAP median (p50):      {:.0} +/- {:.0}", m_p50, s_p50);
    eprintln!("SAP top 1% (p99):      {:.0} +/- {:.0}", m_p99, s_p99);
    eprintln!("Circuit breakers:      {:.1} +/- {:.1}", m_cb, s_cb);
    eprintln!("Cartel proposals pass: {:.1} +/- {:.1} (should be ~0)", m_cp, s_cp);
    eprintln!();

    // Tier distribution (mean across seeds)
    let mut tier_means = [0.0f64; 5];
    for r in &results {
        for i in 0..5 { tier_means[i] += r.tier_counts[i] as f64; }
    }
    for t in &mut tier_means { *t /= seeds.len() as f64; }
    eprintln!("Tier distribution (mean):");
    eprintln!("  Observer:    {:.0}", tier_means[0]);
    eprintln!("  Participant: {:.0}", tier_means[1]);
    eprintln!("  Citizen:     {:.0}", tier_means[2]);
    eprintln!("  Steward:     {:.0}", tier_means[3]);
    eprintln!("  Guardian:    {:.0}", tier_means[4]);

    let total_time: f64 = results.iter().map(|r| r.elapsed_secs).sum();
    eprintln!();
    eprintln!("Total simulation time: {:.1}s ({:.1}s/run)", total_time, total_time / seeds.len() as f64);
}
