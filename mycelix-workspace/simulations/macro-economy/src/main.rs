// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Multi-Currency Macro Economy Simulation
//!
//! Population-scale simulation of the triple-currency system (SAP/TEND/MYCEL)
//! with consciousness-gated governance, demurrage, counter-cyclical credit,
//! and reputation dynamics.
//!
//! ## Usage
//!
//! ```bash
//! # Default: 500 agents, 365 days
//! cargo run --release
//!
//! # Custom parameters
//! cargo run --release -- --agents 1000 --days 730 --seed 42
//!
//! # Pipe CSV to file for analysis
//! cargo run --release -- --agents 500 --days 365 > results.csv 2> diagnostics.txt
//! ```

use rand::prelude::*;
use std::collections::HashMap;

// ============================================================================
// CONSTANTS (matching production: mycelix-finance/types, bridge-common)
// ============================================================================

// Consciousness profile weights
const WEIGHT_IDENTITY: f64 = 0.25;
const WEIGHT_REPUTATION: f64 = 0.25;
const WEIGHT_COMMUNITY: f64 = 0.30;
const WEIGHT_ENGAGEMENT: f64 = 0.20;
const HYSTERESIS_MARGIN: f64 = 0.05;

// SAP demurrage
const DEMURRAGE_RATE: f64 = 0.02;
const EXEMPT_FLOOR: f64 = 1000.0;
const DISCOUNT_PER_CLASS: f64 = 0.001;
const DISCOUNT_CAP: f64 = 0.005;
const DISCOUNT_MIN_CLASSES: usize = 3;
const COMPOST_LOCAL_PCT: f64 = 0.70;
const COMPOST_REGIONAL_PCT: f64 = 0.20;
const COMPOST_GLOBAL_PCT: f64 = 0.10;

// TEND counter-cyclical limits
const TEND_LIMIT_NORMAL: i32 = 40;
const TEND_LIMIT_ELEVATED: i32 = 60;
const TEND_LIMIT_HIGH: i32 = 80;
const TEND_LIMIT_EMERGENCY: i32 = 120;

// Fee tiers (by MYCEL score)
const FEE_NEWCOMER: f64 = 0.0010; // MYCEL < 0.3
const FEE_MEMBER: f64 = 0.0003; // MYCEL 0.3-0.7
const FEE_STEWARD: f64 = 0.0001; // MYCEL > 0.7

// Recognition
const MAX_RECOGNITIONS_PER_MONTH: u32 = 10;
const MIN_MYCEL_TO_GIVE: f64 = 0.3;
const RECOGNITION_BASE_WEIGHT: f64 = 1.0;

// MYCEL
const MYCEL_ANNUAL_DECAY: f64 = 0.05;
const JUBILEE_FACTOR: f64 = 0.8;
const JUBILEE_CYCLE_DAYS: u32 = 1460; // ~4 years

// Reputation
const REPUTATION_DECAY_PER_DAY: f64 = 0.998;
const _REPUTATION_SLASH_FACTOR: f64 = 0.5;
const _BLACKLIST_THRESHOLD: f64 = 0.05;

// ============================================================================
// TYPES
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Strategy {
    Saver,          // Holds SAP, low velocity
    Spender,        // High SAP velocity, low balances
    CommunityFirst, // Maximizes TEND/recognition
    FreeRider,      // Minimal contribution, extracts value
    Gamer,          // Tries to game MYCEL reputation
    Diversifier,    // Multi-asset for demurrage discount
    Newcomer,       // Low initial, growing
}

impl Strategy {
    fn all() -> &'static [Strategy] {
        &[
            Strategy::Saver,
            Strategy::Spender,
            Strategy::CommunityFirst,
            Strategy::FreeRider,
            Strategy::Gamer,
            Strategy::Diversifier,
            Strategy::Newcomer,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            Strategy::Saver => "Saver",
            Strategy::Spender => "Spender",
            Strategy::CommunityFirst => "CommunityFirst",
            Strategy::FreeRider => "FreeRider",
            Strategy::Gamer => "Gamer",
            Strategy::Diversifier => "Diversifier",
            Strategy::Newcomer => "Newcomer",
        }
    }
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
    fn with_hysteresis(score: f64, current: Tier) -> Tier {
        let promoted = if score >= 0.8 + HYSTERESIS_MARGIN {
            Tier::Guardian
        } else if score >= 0.6 + HYSTERESIS_MARGIN {
            Tier::Steward
        } else if score >= 0.4 + HYSTERESIS_MARGIN {
            Tier::Citizen
        } else if score >= 0.3 + HYSTERESIS_MARGIN {
            Tier::Participant
        } else {
            Tier::Observer
        };

        let demoted = if score < 0.3 - HYSTERESIS_MARGIN {
            Tier::Observer
        } else if score < 0.4 - HYSTERESIS_MARGIN {
            Tier::Participant
        } else if score < 0.6 - HYSTERESIS_MARGIN {
            Tier::Citizen
        } else if score < 0.8 - HYSTERESIS_MARGIN {
            Tier::Steward
        } else {
            Tier::Guardian
        };

        if promoted > current {
            promoted
        } else if demoted < current {
            demoted
        } else {
            current
        }
    }
}

// ============================================================================
// AGENT
// ============================================================================

#[derive(Clone)]
struct Agent {
    id: u32,
    strategy: Strategy,

    // Currency balances
    sap_balance: f64,
    tend_balance: i32, // mutual credit (can be negative)
    mycel_score: f64,

    // MYCEL components
    participation: f64,
    recognition_received: f64,
    validation_score: f64,
    longevity_days: u32,

    // Recognition tracking
    recognitions_given_this_month: u32,

    // 4D consciousness profile
    identity: f64,
    reputation: f64,
    community: f64,
    engagement: f64,

    // Tier
    tier: Tier,

    // State
    active: bool,
    days_inactive: u32,
    asset_classes: usize,
    blacklisted: bool,
    total_slashes: u32,

    // Tracking
    transactions_today: u32,
    sap_spent_today: f64,
}

impl Agent {
    fn combined_score(&self) -> f64 {
        self.identity * WEIGHT_IDENTITY
            + self.reputation * WEIGHT_REPUTATION
            + self.community * WEIGHT_COMMUNITY
            + self.engagement * WEIGHT_ENGAGEMENT
    }

    fn fee_rate(&self) -> f64 {
        if self.mycel_score > 0.7 {
            FEE_STEWARD
        } else if self.mycel_score >= 0.3 {
            FEE_MEMBER
        } else {
            FEE_NEWCOMER
        }
    }

    fn tend_limit(&self, vitality: f64) -> i32 {
        if vitality >= 41.0 {
            TEND_LIMIT_NORMAL
        } else if vitality >= 21.0 {
            TEND_LIMIT_ELEVATED
        } else if vitality >= 11.0 {
            TEND_LIMIT_HIGH
        } else {
            TEND_LIMIT_EMERGENCY
        }
    }

    fn update_mycel(&mut self) {
        self.mycel_score = self.participation * 0.4
            + self.recognition_received.min(1.0) * 0.2
            + self.validation_score * 0.2
            + (self.longevity_days as f64 / 365.0).min(1.0) * 0.2;
        self.mycel_score = self.mycel_score.clamp(0.0, 1.0);
    }

    fn update_tier(&mut self) {
        self.tier = Tier::with_hysteresis(self.combined_score(), self.tier);
    }

    fn apply_demurrage(&mut self, base_rate: f64, exempt_floor: f64) -> f64 {
        if self.sap_balance <= exempt_floor {
            return 0.0;
        }
        let eligible = self.sap_balance - exempt_floor;
        let effective_rate = compute_effective_demurrage(base_rate, self.asset_classes);
        // Daily rate from annual
        let daily_decay = 1.0 - (1.0 - effective_rate).powf(1.0 / 365.0);
        let deduction = eligible * daily_decay;
        self.sap_balance -= deduction;
        deduction.max(0.0)
    }
}

fn compute_effective_demurrage(base_rate: f64, distinct_classes: usize) -> f64 {
    if distinct_classes < DISCOUNT_MIN_CLASSES {
        return base_rate;
    }
    let bonus = distinct_classes.saturating_sub(DISCOUNT_MIN_CLASSES - 1);
    let discount = (bonus as f64 * DISCOUNT_PER_CLASS).min(DISCOUNT_CAP);
    (base_rate - discount).max(base_rate * 0.5)
}

// ============================================================================
// METRICS
// ============================================================================

struct DayMetrics {
    day: u32,
    // SAP
    total_sap: f64,
    sap_velocity: f64,
    sap_gini: f64,
    sap_p10: f64,
    sap_p50: f64,
    sap_p90: f64,
    // TEND
    tend_utilization: f64,
    tend_net_imbalance: i64,
    // MYCEL
    mycel_mean: f64,
    mycel_gini: f64,
    // Tiers
    tier_observer: usize,
    tier_participant: usize,
    tier_citizen: usize,
    tier_steward: usize,
    tier_guardian: usize,
    // Activity
    active_count: usize,
    inactive_pct: f64,
    // Fees & compost
    total_fees: f64,
    total_compost: f64,
    avg_fee_rate: f64,
    // Recognition
    recognition_count: u32,
    // Vitality
    metabolic_vitality: f64,
    tend_limit_tier: i32,
    // Jubilee
    jubilee_applied: bool,
}

impl DayMetrics {
    fn csv_header() -> &'static str {
        "day,total_sap,sap_velocity,sap_gini,sap_p10,sap_p50,sap_p90,\
         tend_utilization,tend_net_imbalance,\
         mycel_mean,mycel_gini,\
         tier_observer,tier_participant,tier_citizen,tier_steward,tier_guardian,\
         active_count,inactive_pct,\
         total_fees,total_compost,avg_fee_rate,\
         recognition_count,metabolic_vitality,tend_limit_tier,jubilee_applied"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{:.2},{:.4},{:.4},{:.2},{:.2},{:.2},\
             {:.4},{},\
             {:.4},{:.4},\
             {},{},{},{},{},\
             {},{:.2},\
             {:.4},{:.4},{:.6},\
             {},{:.2},{},{}",
            self.day,
            self.total_sap,
            self.sap_velocity,
            self.sap_gini,
            self.sap_p10,
            self.sap_p50,
            self.sap_p90,
            self.tend_utilization,
            self.tend_net_imbalance,
            self.mycel_mean,
            self.mycel_gini,
            self.tier_observer,
            self.tier_participant,
            self.tier_citizen,
            self.tier_steward,
            self.tier_guardian,
            self.active_count,
            self.inactive_pct,
            self.total_fees,
            self.total_compost,
            self.avg_fee_rate,
            self.recognition_count,
            self.metabolic_vitality,
            self.tend_limit_tier,
            self.jubilee_applied,
        )
    }
}

fn gini(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for (i, &v) in sorted.iter().enumerate() {
        sum += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * v;
    }
    sum / (n as f64 * n as f64 * mean)
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (values.len() - 1) as f64) as usize;
    values[idx.min(values.len() - 1)]
}

// ============================================================================
// SIMULATION
// ============================================================================

struct Simulation {
    agents: Vec<Agent>,
    day: u32,
    rng: StdRng,
    inactive_rate_multiplier: f64,
    demurrage_rate: f64,
    jubilee_cycle_days: u32,
    exempt_floor: f64,

    // Global state
    compost_local: f64,
    compost_regional: f64,
    compost_global: f64,
    metabolic_vitality: f64,

    // Tracking
    total_fees_today: f64,
    total_compost_today: f64,
    total_transactions_today: u32,
    total_recognitions_today: u32,
}

impl Simulation {
    fn new(n_agents: usize, seed: u64, inactive_rate_multiplier: f64, demurrage_rate: f64, jubilee_cycle_days: u32, exempt_floor: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agents = Vec::with_capacity(n_agents);

        // Strategy distribution: 20% each for Saver/Spender/Community, 15% FreeRider,
        // 10% Gamer, 5% Diversifier, 10% Newcomer
        let strategy_weights = [
            (Strategy::Saver, 0.20),
            (Strategy::Spender, 0.20),
            (Strategy::CommunityFirst, 0.20),
            (Strategy::FreeRider, 0.15),
            (Strategy::Gamer, 0.10),
            (Strategy::Diversifier, 0.05),
            (Strategy::Newcomer, 0.10),
        ];

        for i in 0..n_agents {
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut strategy = Strategy::Newcomer;
            for &(s, w) in &strategy_weights {
                cumulative += w;
                if r < cumulative {
                    strategy = s;
                    break;
                }
            }

            let (sap, mycel, identity, community, engagement, asset_classes) = match strategy {
                Strategy::Saver => (
                    rng.gen_range(3000.0..10000.0),
                    rng.gen_range(0.3..0.6),
                    rng.gen_range(0.4..0.7),
                    rng.gen_range(0.2..0.4),
                    rng.gen_range(0.1..0.3),
                    rng.gen_range(2..5),
                ),
                Strategy::Spender => (
                    rng.gen_range(500.0..2000.0),
                    rng.gen_range(0.3..0.5),
                    rng.gen_range(0.3..0.6),
                    rng.gen_range(0.3..0.5),
                    rng.gen_range(0.3..0.6),
                    rng.gen_range(1..3),
                ),
                Strategy::CommunityFirst => (
                    rng.gen_range(1000.0..4000.0),
                    rng.gen_range(0.4..0.7),
                    rng.gen_range(0.5..0.8),
                    rng.gen_range(0.5..0.8),
                    rng.gen_range(0.4..0.7),
                    rng.gen_range(1..4),
                ),
                Strategy::FreeRider => (
                    rng.gen_range(200.0..1500.0),
                    rng.gen_range(0.1..0.3),
                    rng.gen_range(0.1..0.3),
                    rng.gen_range(0.0..0.2),
                    rng.gen_range(0.0..0.2),
                    rng.gen_range(0..2),
                ),
                Strategy::Gamer => (
                    rng.gen_range(1000.0..5000.0),
                    rng.gen_range(0.2..0.5),
                    rng.gen_range(0.3..0.6),
                    rng.gen_range(0.1..0.4),
                    rng.gen_range(0.2..0.5),
                    rng.gen_range(1..3),
                ),
                Strategy::Diversifier => (
                    rng.gen_range(2000.0..8000.0),
                    rng.gen_range(0.4..0.6),
                    rng.gen_range(0.4..0.7),
                    rng.gen_range(0.3..0.5),
                    rng.gen_range(0.2..0.4),
                    rng.gen_range(4..7),
                ),
                Strategy::Newcomer => (
                    rng.gen_range(100.0..500.0),
                    rng.gen_range(0.0..0.2),
                    rng.gen_range(0.2..0.4),
                    rng.gen_range(0.0..0.1),
                    rng.gen_range(0.0..0.1),
                    0,
                ),
            };

            let reputation = rng.gen_range(0.2..0.6);

            let mut agent = Agent {
                id: i as u32,
                strategy,
                sap_balance: sap,
                tend_balance: 0,
                mycel_score: mycel,
                participation: engagement,
                recognition_received: 0.0,
                validation_score: rng.gen_range(0.1..0.5),
                longevity_days: 0,
                recognitions_given_this_month: 0,
                identity,
                reputation,
                community,
                engagement,
                tier: Tier::Observer,
                active: true,
                days_inactive: 0,
                asset_classes,
                blacklisted: false,
                total_slashes: 0,
                transactions_today: 0,
                sap_spent_today: 0.0,
            };
            agent.update_mycel();
            agent.update_tier();
            agents.push(agent);
        }

        Self {
            agents,
            day: 0,
            rng,
            inactive_rate_multiplier,
            demurrage_rate,
            jubilee_cycle_days,
            exempt_floor,
            compost_local: 0.0,
            compost_regional: 0.0,
            compost_global: 0.0,
            metabolic_vitality: 50.0,
            total_fees_today: 0.0,
            total_compost_today: 0.0,
            total_transactions_today: 0,
            total_recognitions_today: 0,
        }
    }

    fn step(&mut self) -> DayMetrics {
        self.day += 1;
        self.total_fees_today = 0.0;
        self.total_compost_today = 0.0;
        self.total_transactions_today = 0;
        self.total_recognitions_today = 0;

        let n = self.agents.len();

        // Reset daily counters
        for agent in &mut self.agents {
            agent.transactions_today = 0;
            agent.sap_spent_today = 0.0;
        }

        // Monthly reset (day 1 of each month)
        if self.day % 30 == 1 {
            for agent in &mut self.agents {
                agent.recognitions_given_this_month = 0;
            }
        }

        // 1. Activity check
        self.update_activity();

        // 2. SAP transactions
        self.run_sap_transactions();

        // 3. TEND exchanges
        self.run_tend_exchanges();

        // 4. Recognition
        self.run_recognition();

        // 5. MYCEL update
        for agent in &mut self.agents {
            agent.longevity_days += 1;
            // Participation tracks engagement
            if agent.active {
                agent.participation = (agent.participation + 0.001).min(1.0);
            } else {
                agent.participation = (agent.participation - 0.002).max(0.0);
            }
            agent.update_mycel();
        }

        // 6. Annual decay on MYCEL (applied daily as 1/365 fraction)
        let daily_decay = 1.0 - (MYCEL_ANNUAL_DECAY / 365.0);
        for agent in &mut self.agents {
            agent.mycel_score *= daily_decay;
        }

        // 7. Jubilee
        let jubilee_applied = self.jubilee_cycle_days > 0 && self.day > 0 && self.day % self.jubilee_cycle_days == 0;
        if jubilee_applied {
            for agent in &mut self.agents {
                agent.mycel_score *= JUBILEE_FACTOR;
            }
            eprintln!("Day {}: JUBILEE applied (0.8× MYCEL compression)", self.day);
        }

        // 8. Demurrage
        let demurrage = self.demurrage_rate;
        let floor = self.exempt_floor;
        for agent in &mut self.agents {
            let compost = agent.apply_demurrage(demurrage, floor);
            if compost > 0.0 {
                self.compost_local += compost * COMPOST_LOCAL_PCT;
                self.compost_regional += compost * COMPOST_REGIONAL_PCT;
                self.compost_global += compost * COMPOST_GLOBAL_PCT;
                self.total_compost_today += compost;
            }
        }

        // 9. Reputation decay
        for agent in &mut self.agents {
            agent.reputation *= REPUTATION_DECAY_PER_DAY;
            if agent.active {
                agent.reputation = (agent.reputation + 0.0005).min(1.0);
            }
        }

        // 10. Tier update
        for agent in &mut self.agents {
            agent.update_tier();
        }

        // 11. Update metabolic vitality
        let active_ratio = self.agents.iter().filter(|a| a.active).count() as f64 / n as f64;
        let tx_velocity = self.total_transactions_today as f64 / n as f64;
        self.metabolic_vitality = (active_ratio * 50.0 + tx_velocity * 10.0).clamp(0.0, 100.0);

        // Collect metrics
        self.collect_metrics(jubilee_applied)
    }

    fn update_activity(&mut self) {
        for agent in &mut self.agents {
            let inactivity_prob = match agent.strategy {
                Strategy::FreeRider => 0.05,  // 5% daily → ~82% monthly inactive
                Strategy::Newcomer => 0.03,
                Strategy::Saver => 0.01,
                _ => 0.005,
            };
            if self.rng.gen::<f64>() < inactivity_prob * self.inactive_rate_multiplier {
                agent.active = false;
                agent.days_inactive += 1;
            } else {
                agent.active = true;
                agent.days_inactive = 0;
            }
        }
    }

    fn run_sap_transactions(&mut self) {
        let n = self.agents.len();
        // Determine number of transactions per agent based on strategy
        let mut tx_pairs: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            if !self.agents[i].active || self.agents[i].blacklisted {
                continue;
            }

            let n_tx = match self.agents[i].strategy {
                Strategy::Spender => self.rng.gen_range(2..6),
                Strategy::CommunityFirst => self.rng.gen_range(1..4),
                Strategy::Saver => self.rng.gen_range(0..2),
                Strategy::FreeRider => self.rng.gen_range(0..2),
                Strategy::Gamer => self.rng.gen_range(1..3),
                Strategy::Diversifier => self.rng.gen_range(1..3),
                Strategy::Newcomer => self.rng.gen_range(0..2),
            };

            for _ in 0..n_tx {
                let receiver = self.rng.gen_range(0..n);
                if receiver == i {
                    continue;
                }
                let max_amount = match self.agents[i].strategy {
                    Strategy::Spender => (self.agents[i].sap_balance * 0.1).min(200.0),
                    Strategy::Saver => (self.agents[i].sap_balance * 0.02).min(50.0),
                    _ => (self.agents[i].sap_balance * 0.05).min(100.0),
                };
                if max_amount < 1.0 {
                    continue;
                }
                let amount = self.rng.gen_range(1.0..max_amount);
                tx_pairs.push((i, receiver, amount));
            }
        }

        // Execute transactions
        for (sender, receiver, amount) in tx_pairs {
            let fee_rate = self.agents[sender].fee_rate();
            let fee = amount * fee_rate;
            let total = amount + fee;

            if self.agents[sender].sap_balance >= total {
                self.agents[sender].sap_balance -= total;
                self.agents[sender].sap_spent_today += amount;
                self.agents[sender].transactions_today += 1;
                self.agents[receiver].sap_balance += amount;
                self.total_fees_today += fee;
                self.total_transactions_today += 1;
            }
        }
    }

    fn run_tend_exchanges(&mut self) {
        let n = self.agents.len();
        let vitality = self.metabolic_vitality;
        let mut exchanges: Vec<(usize, usize, i32)> = Vec::new();

        for i in 0..n {
            if !self.agents[i].active {
                continue;
            }

            // CommunityFirst agents use TEND more
            let should_exchange = match self.agents[i].strategy {
                Strategy::CommunityFirst => self.rng.gen::<f64>() < 0.3,
                Strategy::Spender => self.rng.gen::<f64>() < 0.15,
                Strategy::FreeRider => self.rng.gen::<f64>() < 0.05,
                _ => self.rng.gen::<f64>() < 0.10,
            };

            if !should_exchange {
                continue;
            }

            let partner = self.rng.gen_range(0..n);
            if partner == i || !self.agents[partner].active {
                continue;
            }

            let hours = self.rng.gen_range(1..4);
            let limit_i = self.agents[i].tend_limit(vitality);
            let limit_p = self.agents[partner].tend_limit(vitality);

            // i provides hours to partner: i's balance decreases, partner's increases
            if self.agents[i].tend_balance - hours >= -limit_i
                && self.agents[partner].tend_balance + hours <= limit_p
            {
                exchanges.push((i, partner, hours));
            }
        }

        for (provider, recipient, hours) in exchanges {
            self.agents[provider].tend_balance -= hours;
            self.agents[recipient].tend_balance += hours;
            self.total_transactions_today += 1;
        }
    }

    fn run_recognition(&mut self) {
        let n = self.agents.len();
        let mut recognitions: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            if !self.agents[i].active {
                continue;
            }
            if self.agents[i].mycel_score < MIN_MYCEL_TO_GIVE {
                continue;
            }
            if self.agents[i].recognitions_given_this_month >= MAX_RECOGNITIONS_PER_MONTH {
                continue;
            }

            // Probability of giving recognition
            let prob = match self.agents[i].strategy {
                Strategy::CommunityFirst => 0.15,
                Strategy::Gamer => 0.20, // gamers give lots of recognition to build network
                Strategy::Saver | Strategy::Diversifier => 0.03,
                Strategy::FreeRider => 0.01,
                _ => 0.05,
            };

            if self.rng.gen::<f64>() < prob {
                let target = self.rng.gen_range(0..n);
                if target != i {
                    let weight = self.agents[i].mycel_score * RECOGNITION_BASE_WEIGHT;
                    recognitions.push((i, target, weight));
                }
            }
        }

        for (giver, receiver, weight) in recognitions {
            self.agents[giver].recognitions_given_this_month += 1;
            // Accumulate recognition, normalize to [0,1] via diminishing returns
            let current = self.agents[receiver].recognition_received;
            self.agents[receiver].recognition_received = current + weight * 0.1 * (1.0 - current);
            self.agents[receiver].community =
                (self.agents[receiver].community + 0.005).min(1.0);
            self.total_recognitions_today += 1;
        }
    }

    fn collect_metrics(&self, jubilee_applied: bool) -> DayMetrics {
        let n = self.agents.len();

        let mut sap_values: Vec<f64> = self.agents.iter().map(|a| a.sap_balance).collect();
        let mycel_values: Vec<f64> = self.agents.iter().map(|a| a.mycel_score).collect();

        let total_sap: f64 = sap_values.iter().sum();
        let sap_velocity = self.total_transactions_today as f64 / n as f64;

        // TEND
        let tend_total_abs: i32 = self.agents.iter().map(|a| a.tend_balance.abs()).sum();
        let tend_limit_sum: i32 = self.agents.iter().map(|a| a.tend_limit(self.metabolic_vitality)).sum();
        let tend_utilization = if tend_limit_sum > 0 {
            tend_total_abs as f64 / tend_limit_sum as f64
        } else {
            0.0
        };
        let tend_net: i64 = self.agents.iter().map(|a| a.tend_balance as i64).sum();

        // Tiers
        let mut tier_counts = [0usize; 5];
        for agent in &self.agents {
            tier_counts[agent.tier as usize] += 1;
        }

        let active_count = self.agents.iter().filter(|a| a.active).count();
        let inactive_pct = 100.0 * (n - active_count) as f64 / n as f64;

        let fee_rates: Vec<f64> = self.agents.iter().map(|a| a.fee_rate()).collect();
        let avg_fee_rate = fee_rates.iter().sum::<f64>() / n as f64;

        let vitality = self.metabolic_vitality;
        let tend_limit = if vitality >= 41.0 {
            TEND_LIMIT_NORMAL
        } else if vitality >= 21.0 {
            TEND_LIMIT_ELEVATED
        } else if vitality >= 11.0 {
            TEND_LIMIT_HIGH
        } else {
            TEND_LIMIT_EMERGENCY
        };

        DayMetrics {
            day: self.day,
            total_sap,
            sap_velocity,
            sap_gini: gini(&sap_values),
            sap_p10: percentile(&mut sap_values, 10.0),
            sap_p50: percentile(&mut sap_values, 50.0),
            sap_p90: percentile(&mut sap_values, 90.0),
            tend_utilization,
            tend_net_imbalance: tend_net,
            mycel_mean: mycel_values.iter().sum::<f64>() / n as f64,
            mycel_gini: gini(&mycel_values),
            tier_observer: tier_counts[0],
            tier_participant: tier_counts[1],
            tier_citizen: tier_counts[2],
            tier_steward: tier_counts[3],
            tier_guardian: tier_counts[4],
            active_count,
            inactive_pct,
            total_fees: self.total_fees_today,
            total_compost: self.total_compost_today,
            avg_fee_rate,
            recognition_count: self.total_recognitions_today,
            metabolic_vitality: vitality,
            tend_limit_tier: tend_limit,
            jubilee_applied,
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n_agents = 500;
    let mut n_days = 365;
    let mut seed = 42u64;
    let mut inactive_rate = 1.0f64;
    let mut demurrage_rate = DEMURRAGE_RATE;
    let mut jubilee_years = 4u32;
    let mut exempt_floor = 200.0f64; // Reduced from 1000 so demurrage actually affects behavior

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agents" => {
                i += 1;
                n_agents = args[i].parse().unwrap_or(500);
            }
            "--days" => {
                i += 1;
                n_days = args[i].parse().unwrap_or(365);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(42);
            }
            "--inactive-rate" => {
                i += 1;
                inactive_rate = args[i].parse().unwrap_or(1.0);
            }
            "--demurrage-rate" => {
                i += 1;
                demurrage_rate = args[i].parse().unwrap_or(DEMURRAGE_RATE);
            }
            "--jubilee-years" => {
                i += 1;
                jubilee_years = args[i].parse().unwrap_or(4);
            }
            "--exempt-floor" => {
                i += 1;
                exempt_floor = args[i].parse().unwrap_or(200.0);
            }
            "--help" | "-h" => {
                eprintln!("Mycelix Multi-Currency Macro Economy Simulation");
                eprintln!();
                eprintln!("USAGE:");
                eprintln!("  cargo run --release -- [OPTIONS]");
                eprintln!();
                eprintln!("OPTIONS:");
                eprintln!("  --agents <N>   Number of agents (default: 500)");
                eprintln!("  --days <N>     Simulation days (default: 365)");
                eprintln!("  --seed <N>     Random seed (default: 42)");
                eprintln!("  --inactive-rate <F>  Inactivity multiplier (default: 1.0, 0=none)");
                eprintln!("  --demurrage-rate <F> Annual SAP demurrage rate (default: 0.02)");
                eprintln!("  --jubilee-years <N>  Years between jubilee compression (default: 4, 0=never)");
                eprintln!("  --exempt-floor <F>   SAP exempt from demurrage (default: 200)");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!("=== Mycelix Macro Economy Simulation ===");
    eprintln!("Agents: {}, Days: {}, Seed: {}", n_agents, n_days, seed);

    let jubilee_days = if jubilee_years == 0 { 0 } else { jubilee_years * 365 };
    let mut sim = Simulation::new(n_agents, seed, inactive_rate, demurrage_rate, jubilee_days, exempt_floor);

    // Strategy distribution
    let mut strategy_counts: HashMap<&str, usize> = HashMap::new();
    for agent in &sim.agents {
        *strategy_counts.entry(agent.strategy.as_str()).or_default() += 1;
    }
    eprintln!("Strategy distribution:");
    for s in Strategy::all() {
        eprintln!(
            "  {}: {} ({:.1}%)",
            s.as_str(),
            strategy_counts.get(s.as_str()).unwrap_or(&0),
            100.0 * *strategy_counts.get(s.as_str()).unwrap_or(&0) as f64 / n_agents as f64
        );
    }
    eprintln!();

    // CSV header
    println!("{}", DayMetrics::csv_header());

    for _day in 0..n_days {
        let metrics = sim.step();
        println!("{}", metrics.to_csv());
    }

    // Final summary
    eprintln!();
    eprintln!("=== Final State (Day {}) ===", sim.day);

    let total_sap: f64 = sim.agents.iter().map(|a| a.sap_balance).sum();
    let tend_net: i64 = sim.agents.iter().map(|a| a.tend_balance as i64).sum();
    let mycel_mean: f64 = sim.agents.iter().map(|a| a.mycel_score).sum::<f64>() / n_agents as f64;

    eprintln!("Total SAP: {:.2}", total_sap);
    eprintln!("TEND net imbalance: {} (should be 0)", tend_net);
    eprintln!("MYCEL mean: {:.4}", mycel_mean);
    eprintln!("Compost pools: local={:.2}, regional={:.2}, global={:.2}",
        sim.compost_local, sim.compost_regional, sim.compost_global);

    let mut tier_counts = [0usize; 5];
    for agent in &sim.agents {
        tier_counts[agent.tier as usize] += 1;
    }
    eprintln!("Tier distribution: Observer={}, Participant={}, Citizen={}, Steward={}, Guardian={}",
        tier_counts[0], tier_counts[1], tier_counts[2], tier_counts[3], tier_counts[4]);

    // Per-strategy SAP
    eprintln!();
    eprintln!("Per-strategy SAP mean:");
    for s in Strategy::all() {
        let agents: Vec<&Agent> = sim.agents.iter().filter(|a| a.strategy == *s).collect();
        if !agents.is_empty() {
            let mean: f64 = agents.iter().map(|a| a.sap_balance).sum::<f64>() / agents.len() as f64;
            let mycel: f64 = agents.iter().map(|a| a.mycel_score).sum::<f64>() / agents.len() as f64;
            eprintln!("  {}: SAP={:.2}, MYCEL={:.4}", s.as_str(), mean, mycel);
        }
    }

    // Invariant check
    if tend_net != 0 {
        eprintln!("WARNING: TEND zero-sum invariant VIOLATED (net={})", tend_net);
    } else {
        eprintln!("TEND zero-sum invariant: OK");
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tend_zero_sum() {
        let mut sim = Simulation::new(200, 42, 1.0, DEMURRAGE_RATE, JUBILEE_CYCLE_DAYS, EXEMPT_FLOOR);
        for _ in 0..365 {
            sim.step();
        }
        let tend_net: i64 = sim.agents.iter().map(|a| a.tend_balance as i64).sum();
        assert_eq!(tend_net, 0, "TEND must be zero-sum, got {}", tend_net);
    }

    #[test]
    fn test_sap_non_negative() {
        let mut sim = Simulation::new(100, 42, 1.0, DEMURRAGE_RATE, JUBILEE_CYCLE_DAYS, EXEMPT_FLOOR);
        for _ in 0..365 {
            sim.step();
        }
        for agent in &sim.agents {
            assert!(
                agent.sap_balance >= 0.0,
                "Agent {} SAP is negative: {}",
                agent.id,
                agent.sap_balance
            );
        }
    }

    #[test]
    fn test_mycel_bounded() {
        let mut sim = Simulation::new(100, 42, 1.0, DEMURRAGE_RATE, JUBILEE_CYCLE_DAYS, EXEMPT_FLOOR);
        for _ in 0..365 {
            sim.step();
        }
        for agent in &sim.agents {
            assert!(
                agent.mycel_score >= 0.0 && agent.mycel_score <= 1.0,
                "Agent {} MYCEL out of bounds: {}",
                agent.id,
                agent.mycel_score
            );
        }
    }

    #[test]
    fn test_tend_within_max_limit() {
        // TEND balances were valid when exchanged but vitality can shift tiers,
        // so we check against the maximum possible limit (EMERGENCY=120).
        let mut sim = Simulation::new(100, 42, 1.0, DEMURRAGE_RATE, JUBILEE_CYCLE_DAYS, EXEMPT_FLOOR);
        for _ in 0..365 {
            sim.step();
        }
        for agent in &sim.agents {
            assert!(
                agent.tend_balance.abs() <= TEND_LIMIT_EMERGENCY,
                "Agent {} TEND {} exceeds max limit {}",
                agent.id,
                agent.tend_balance,
                TEND_LIMIT_EMERGENCY
            );
        }
    }

    #[test]
    fn test_demurrage_reduces_sap() {
        let mut agent = Agent {
            id: 0,
            strategy: Strategy::Saver,
            sap_balance: 5000.0,
            tend_balance: 0,
            mycel_score: 0.5,
            participation: 0.5,
            recognition_received: 0.0,
            validation_score: 0.5,
            longevity_days: 100,
            recognitions_given_this_month: 0,
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
            tier: Tier::Citizen,
            active: true,
            days_inactive: 0,
            asset_classes: 1,
            blacklisted: false,
            total_slashes: 0,
            transactions_today: 0,
            sap_spent_today: 0.0,
        };

        let initial = agent.sap_balance;
        let compost = agent.apply_demurrage(DEMURRAGE_RATE, EXEMPT_FLOOR);
        assert!(compost > 0.0, "Demurrage should produce compost");
        assert!(
            agent.sap_balance < initial,
            "SAP should decrease after demurrage"
        );
        assert!(
            agent.sap_balance >= EXEMPT_FLOOR,
            "SAP should not go below exempt floor"
        );
    }

    #[test]
    fn test_gini_coefficient() {
        // Perfect equality
        assert!((gini(&[100.0, 100.0, 100.0, 100.0]) - 0.0).abs() < 0.01);
        // High inequality
        let g = gini(&[0.0, 0.0, 0.0, 1000.0]);
        assert!(g > 0.5, "High inequality Gini should be > 0.5, got {}", g);
    }

    #[test]
    fn test_jubilee_reduces_mycel() {
        let mut sim = Simulation::new(50, 42, 1.0, DEMURRAGE_RATE, JUBILEE_CYCLE_DAYS, EXEMPT_FLOOR);
        // Run to just before jubilee
        for _ in 0..(JUBILEE_CYCLE_DAYS - 1) {
            sim.step();
        }
        let pre_mycel: Vec<f64> = sim.agents.iter().map(|a| a.mycel_score).collect();
        sim.step(); // jubilee day
        for (i, agent) in sim.agents.iter().enumerate() {
            // Allow for daily decay on top of jubilee
            assert!(
                agent.mycel_score <= pre_mycel[i] * 1.01,
                "Jubilee should not increase MYCEL"
            );
        }
    }

    #[test]
    fn test_effective_demurrage_discount() {
        let base = compute_effective_demurrage(0.02, 0);
        assert_eq!(base, 0.02);

        let with_3 = compute_effective_demurrage(0.02, 3);
        assert!(with_3 < 0.02, "3 asset classes should reduce rate");

        let with_10 = compute_effective_demurrage(0.02, 10);
        assert!(
            with_10 >= 0.02 * 0.5,
            "Discount should not exceed 50% of base"
        );
    }
}
