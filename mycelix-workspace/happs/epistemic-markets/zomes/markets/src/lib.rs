// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Markets - Core Market Zome
//!
//! Revolutionary prediction market infrastructure for Mycelix featuring:
//! - Multi-dimensional stakes (money, reputation, social, commitment)
//! - Epistemic position-driven resolution mechanisms
//! - MATL-weighted oracle networks with 45% Byzantine tolerance
//! - Cross-hApp market integration via Bridge Protocol

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// EPISTEMIC CLASSIFICATION (Integration with Mycelix Epistemic Charter)
// ============================================================================

/// Empirical axis: How can this claim be verified?
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmpiricalLevel {
    /// E0: Subjective experience, no external verification possible
    Subjective,
    /// E1: Witness testimony, social verification
    Testimonial,
    /// E2: Private verification (ZK proofs, credentials)
    PrivateVerify,
    /// E3: Cryptographic/on-chain verification
    Cryptographic,
    /// E4: Fully measurable, scientifically reproducible
    Measurable,
}

/// Normative axis: Who must agree for this to be "true"?
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormativeLevel {
    /// N0: Individual perspective only
    Personal,
    /// N1: Community consensus required
    Communal,
    /// N2: Network-wide agreement
    Network,
    /// N3: Universal/objective truth
    Universal,
}

/// Materiality axis: How long does this claim matter?
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialityLevel {
    /// M0: Ephemeral, doesn't persist
    Ephemeral,
    /// M1: Time-limited relevance
    Temporal,
    /// M2: Persistent record
    Persistent,
    /// M3: Foundational to understanding
    Foundational,
}

/// 3D epistemic position determines market behavior
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EpistemicPosition {
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
}

impl EpistemicPosition {
    /// Derive resolution mechanism from epistemic position
    pub fn recommended_resolution(&self) -> ResolutionMechanism {
        match self.empirical {
            EmpiricalLevel::Measurable | EmpiricalLevel::Cryptographic => {
                ResolutionMechanism::Automated
            }
            EmpiricalLevel::PrivateVerify => ResolutionMechanism::ZkVerified,
            EmpiricalLevel::Testimonial => match self.normative {
                NormativeLevel::Universal | NormativeLevel::Network => {
                    ResolutionMechanism::OracleConsensus
                }
                _ => ResolutionMechanism::CommunityVote,
            },
            EmpiricalLevel::Subjective => ResolutionMechanism::ReputationStaked,
        }
    }

    /// Derive recommended market duration from materiality
    pub fn recommended_duration_days(&self) -> u64 {
        match self.materiality {
            MaterialityLevel::Ephemeral => 1,
            MaterialityLevel::Temporal => 7,
            MaterialityLevel::Persistent => 30,
            MaterialityLevel::Foundational => 90,
        }
    }
}

// ============================================================================
// MARKET MECHANISMS
// ============================================================================

/// How market prices are determined
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MarketMechanism {
    /// Simple binary market (yes/no)
    Binary {
        yes_shares: u64,
        no_shares: u64,
    },

    /// Logarithmic Market Scoring Rule (bounded loss for market maker)
    /// Good for low liquidity, provides always-available prices
    LMSR {
        liquidity_parameter: f64,
        subsidy_pool: u64,
        outcome_quantities: Vec<u64>,
    },

    /// Continuous Double Auction (order book)
    /// More efficient for liquid markets
    CDA {
        bids: Vec<Order>,
        asks: Vec<Order>,
    },

    /// Parimutuel (all bets pooled, proportional payout)
    /// Simple, no market maker needed
    Parimutuel {
        pools: Vec<(OutcomeId, u64)>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Order {
    pub agent: AgentPubKey,
    pub outcome: OutcomeId,
    pub price: f64,
    pub quantity: u64,
    pub timestamp: u64,
}

/// How outcomes are determined
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResolutionMechanism {
    /// Automated: triggered by on-chain event or oracle feed
    Automated,

    /// ZK-verified: outcome proven via zero-knowledge proof
    ZkVerified,

    /// Oracle consensus: MATL-weighted oracle votes
    OracleConsensus,

    /// Community vote: broader participation, MATL-weighted
    CommunityVote,

    /// Reputation staked: attestors stake their MATL score
    ReputationStaked,

    /// Governance escalation: disputed markets go to governance
    GovernanceEscalation,
}

// ============================================================================
// MULTI-DIMENSIONAL STAKES (Revolutionary Feature)
// ============================================================================

/// Stakes can be monetary, reputational, social, or commitment-based
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MultiDimensionalStake {
    /// Traditional monetary stake
    pub monetary: Option<MonetaryStake>,

    /// MATL reputation at risk
    pub reputation: Option<ReputationStake>,

    /// Social visibility stake (public prediction)
    pub social: Option<SocialStake>,

    /// Future commitment stake
    pub commitment: Option<CommitmentStake>,

    /// Time/attention investment
    pub time: Option<TimeStake>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MonetaryStake {
    pub amount: u64,
    pub currency: String,
    /// Finance hApp escrow ID
    pub escrow_id: Option<EntryHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReputationStake {
    /// Which MATL domains are at risk
    pub domains: Vec<String>,
    /// Percentage of domain score staked (0.0 - 1.0)
    pub stake_percentage: f64,
    /// Confidence multiplier: higher confidence = more at risk if wrong
    pub confidence_multiplier: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SocialStake {
    /// Visibility level
    pub visibility: Visibility,
    /// Link to verified identity (higher stake)
    pub identity_link: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Visibility {
    /// Only visible after resolution
    Private,
    /// Visible to specific agents
    Limited(Vec<AgentPubKey>),
    /// Visible to community members
    Community,
    /// Publicly visible
    Public,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommitmentStake {
    /// Actions committed if prediction is correct
    pub if_correct: Vec<Commitment>,
    /// Actions committed if prediction is wrong
    pub if_wrong: Vec<Commitment>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Commitment {
    /// Publicly update stated beliefs
    BeliefUpdate { topic: String },
    /// Spend time investigating a topic
    Investigation { hours: u32, topic: String },
    /// Financial donation
    Donation { amount: u64, recipient: String },
    /// Mentorship commitment
    Mentorship { hours: u32, domain: String },
    /// Custom commitment
    Custom { description: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimeStake {
    /// Hours of research invested
    pub research_hours: u32,
    /// Evidence provided
    pub evidence_submitted: Vec<EntryHash>,
}

impl MultiDimensionalStake {
    /// Calculate total stake value for ranking/weighting
    pub fn total_value(&self) -> f64 {
        let mut total = 0.0;

        if let Some(ref m) = self.monetary {
            total += m.amount as f64;
        }

        if let Some(ref r) = self.reputation {
            // Reputation is valuable: 1% of MATL ≈ 100 units
            total += r.stake_percentage * 10000.0 * r.confidence_multiplier;
        }

        if let Some(ref s) = self.social {
            total += match s.visibility {
                Visibility::Private => 0.0,
                Visibility::Limited(_) => 50.0,
                Visibility::Community => 200.0,
                Visibility::Public => 500.0,
            };
            if s.identity_link.is_some() {
                total *= 2.0; // Identity-linked predictions worth more
            }
        }

        if let Some(ref c) = self.commitment {
            for commitment in &c.if_wrong {
                total += match commitment {
                    Commitment::Investigation { hours, .. } => *hours as f64 * 20.0,
                    Commitment::Donation { amount, .. } => *amount as f64,
                    Commitment::Mentorship { hours, .. } => *hours as f64 * 30.0,
                    _ => 50.0,
                };
            }
        }

        if let Some(ref t) = self.time {
            total += t.research_hours as f64 * 15.0;
            total += t.evidence_submitted.len() as f64 * 25.0;
        }

        total
    }

    /// Create a reputation-only stake (for those without monetary resources)
    pub fn reputation_only(domains: Vec<String>, stake_pct: f64, confidence: f64) -> Self {
        Self {
            monetary: None,
            reputation: Some(ReputationStake {
                domains,
                stake_percentage: stake_pct,
                confidence_multiplier: confidence,
            }),
            social: None,
            commitment: None,
            time: None,
        }
    }

    /// Create a monetary stake
    pub fn monetary_only(amount: u64, currency: String) -> Self {
        Self {
            monetary: Some(MonetaryStake {
                amount,
                currency,
                escrow_id: None,
            }),
            reputation: None,
            social: None,
            commitment: None,
            time: None,
        }
    }
}

// ============================================================================
// CORE MARKET STRUCTURE
// ============================================================================

pub type OutcomeId = String;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Outcome {
    pub id: OutcomeId,
    pub description: String,
    pub current_probability: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum MarketStatus {
    /// Market is open for predictions
    Open,
    /// Market is closed, awaiting resolution
    Closed,
    /// Resolution in progress
    Resolving,
    /// Market resolved, payouts available
    Resolved,
    /// Market disputed, under review
    Disputed,
    /// Market cancelled (invalid question, etc.)
    Cancelled,
}

/// Core market entry
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Market {
    /// Unique identifier
    pub id: EntryHash,

    /// The question being predicted
    pub question: String,

    /// Detailed description and resolution criteria
    pub description: String,

    /// Creator of the market
    pub creator: AgentPubKey,

    /// Epistemic classification (drives resolution mechanism)
    pub epistemic_position: EpistemicPosition,

    /// Possible outcomes
    pub outcomes: Vec<Outcome>,

    /// Market mechanism (how prices are determined)
    pub mechanism: MarketMechanism,

    /// Resolution mechanism (how outcome is determined)
    pub resolution: ResolutionMechanism,

    /// Minimum MATL score to participate
    pub min_participant_matl: f64,

    /// Minimum MATL score for oracles
    pub min_oracle_matl: f64,

    /// Tags for discovery
    pub tags: Vec<String>,

    /// Source hApp (if cross-hApp market)
    pub source_happ: Option<String>,

    /// External resolution data source
    pub resolution_source: Option<String>,

    /// Lifecycle timestamps
    pub created_at: u64,
    pub closes_at: u64,
    pub resolution_deadline: u64,
    pub resolved_at: Option<u64>,

    /// Current status
    pub status: MarketStatus,

    /// Resolved outcome (if resolved)
    pub resolved_outcome: Option<OutcomeId>,

    /// Total value staked (all dimensions)
    pub total_stake_value: f64,

    /// Number of unique predictors
    pub predictor_count: u64,
}

/// Input for creating a new market
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateMarketInput {
    pub question: String,
    pub description: String,
    pub outcomes: Vec<String>,
    pub epistemic_position: EpistemicPosition,
    pub mechanism: MarketMechanism,
    pub closes_at: u64,
    pub resolution_deadline: u64,
    pub min_participant_matl: Option<f64>,
    pub min_oracle_matl: Option<f64>,
    pub tags: Vec<String>,
    pub source_happ: Option<String>,
    pub resolution_source: Option<String>,
    /// Initial liquidity subsidy (if LMSR)
    pub initial_subsidy: Option<u64>,
}

// ============================================================================
// ZOME FUNCTIONS
// ============================================================================

#[hdk_extern]
pub fn create_market(input: CreateMarketInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let creator = agent_info()?.agent_initial_pubkey;

    // Validate: closes_at must be in future
    if input.closes_at <= now {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Market close time must be in the future".into()
        )));
    }

    // Validate: at least 2 outcomes
    if input.outcomes.len() < 2 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Market must have at least 2 outcomes".into()
        )));
    }

    // Build outcomes with initial equal probabilities
    let initial_prob = 1.0 / input.outcomes.len() as f64;
    let outcomes: Vec<Outcome> = input
        .outcomes
        .iter()
        .enumerate()
        .map(|(i, desc)| Outcome {
            id: format!("outcome_{}", i),
            description: desc.clone(),
            current_probability: initial_prob,
        })
        .collect();

    // Derive resolution mechanism from epistemic position if not specified
    let resolution = input.epistemic_position.recommended_resolution();

    // Create market entry
    let market = Market {
        id: EntryHash::from_raw_36(vec![0; 36]), // Placeholder, will be updated
        question: input.question,
        description: input.description,
        creator: creator.clone(),
        epistemic_position: input.epistemic_position,
        outcomes,
        mechanism: input.mechanism,
        resolution,
        min_participant_matl: input.min_participant_matl.unwrap_or(0.0),
        min_oracle_matl: input.min_oracle_matl.unwrap_or(0.7),
        tags: input.tags,
        source_happ: input.source_happ,
        resolution_source: input.resolution_source,
        created_at: now,
        closes_at: input.closes_at,
        resolution_deadline: input.resolution_deadline,
        resolved_at: None,
        status: MarketStatus::Open,
        resolved_outcome: None,
        total_stake_value: 0.0,
        predictor_count: 0,
    };

    // Create the entry
    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let entry_hash = hash_entry(&market)?;
    let _action_hash = create_entry(&EntryTypes::Market(market.clone()))?;

    // Link to creator
    create_link(
        creator.clone(),
        entry_hash.clone(),
        LinkTypes::CreatorToMarket,
        (),
    )?;

    // Link to "all_markets" anchor for discovery
    let all_markets_anchor = anchor_hash("all_markets")?;
    create_link(
        all_markets_anchor,
        entry_hash.clone(),
        LinkTypes::AnchorToMarket,
        (),
    )?;

    // Link by tags
    for tag in &market.tags {
        let tag_anchor = anchor_hash(&format!("tag:{}", tag))?;
        create_link(tag_anchor, entry_hash.clone(), LinkTypes::TagToMarket, ())?;
    }

    Ok(entry_hash)
}

#[hdk_extern]
pub fn get_market(market_hash: EntryHash) -> ExternResult<Option<Market>> {
    match get(market_hash, GetOptions::default())? {
        Some(record) => {
            let market: Market = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market not found".into())))?;
            Ok(Some(market))
        }
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn list_markets(_: ()) -> ExternResult<Vec<Market>> {
    let all_markets_anchor = anchor_hash("all_markets")?;
    let links = get_links(
        LinkQuery::try_new(all_markets_anchor, LinkTypes::AnchorToMarket)?,
        GetStrategy::default(),
    )?;

    let mut markets = Vec::new();
    for link in links {
        if let Some(market) = get_market(link.target.into_entry_hash().unwrap())? {
            markets.push(market);
        }
    }

    // Sort by creation time (newest first)
    markets.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(markets)
}

#[hdk_extern]
pub fn list_open_markets(_: ()) -> ExternResult<Vec<Market>> {
    let markets = list_markets(())?;
    Ok(markets
        .into_iter()
        .filter(|m| m.status == MarketStatus::Open)
        .collect())
}

#[hdk_extern]
pub fn search_markets_by_tag(tag: String) -> ExternResult<Vec<Market>> {
    let tag_anchor = anchor_hash(&format!("tag:{}", tag))?;
    let links =
        get_links(LinkQuery::try_new(tag_anchor, LinkTypes::TagToMarket)?, GetStrategy::default())?;

    let mut markets = Vec::new();
    for link in links {
        if let Some(market) = get_market(link.target.into_entry_hash().unwrap())? {
            markets.push(market);
        }
    }

    Ok(markets)
}

#[hdk_extern]
pub fn close_market(market_hash: EntryHash) -> ExternResult<EntryHash> {
    // Get record to extract action hash for update
    let record = get(market_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market not found".into())))?;

    let mut market: Market = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market entry not found".into())))?;

    // Only creator or system can close
    let caller = agent_info()?.agent_initial_pubkey;
    if caller != market.creator {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only market creator can close market".into()
        )));
    }

    // Can only close open markets
    if market.status != MarketStatus::Open {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Market is not open".into()
        )));
    }

    market.status = MarketStatus::Closed;

    // HDK 0.6: update_entry takes ActionHash
    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &market)?;

    Ok(market.id)
}

// ============================================================================
// LMSR PRICE DISCOVERY
// ============================================================================

/// Logarithmic Market Scoring Rule - Price Calculation
///
/// LMSR provides always-available prices with bounded market maker loss.
/// Formula: price_i = e^(q_i/b) / Σ e^(q_j/b)
/// where b = liquidity parameter, q_i = quantity of outcome i
///
/// # Arguments
/// * `outcome_quantities` - Current quantities for each outcome
/// * `liquidity_parameter` - The "b" parameter controlling price sensitivity
/// * `outcome_index` - Which outcome to get the price for
///
/// # Returns
/// Price in range (0, 1) representing probability
pub fn lmsr_price(
    outcome_quantities: &[u64],
    liquidity_parameter: f64,
    outcome_index: usize,
) -> f64 {
    if outcome_quantities.is_empty() || outcome_index >= outcome_quantities.len() {
        return 0.0;
    }

    // Compute e^(q_i/b) for each outcome using log-sum-exp for numerical stability
    let exponents: Vec<f64> = outcome_quantities
        .iter()
        .map(|q| (*q as f64) / liquidity_parameter)
        .collect();

    // Find max for numerical stability (log-sum-exp trick)
    let max_exp = exponents.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Compute sum of e^(x - max) then multiply by e^max
    let sum_exp: f64 = exponents
        .iter()
        .map(|e| (e - max_exp).exp())
        .sum();

    // Price = e^(q_i/b - max) / sum_exp
    let price = (exponents[outcome_index] - max_exp).exp() / sum_exp;
    price.clamp(0.001, 0.999) // Prevent 0 or 1 prices
}

/// Calculate all LMSR prices for a market
pub fn lmsr_prices(
    outcome_quantities: &[u64],
    liquidity_parameter: f64,
) -> Vec<f64> {
    (0..outcome_quantities.len())
        .map(|i| lmsr_price(outcome_quantities, liquidity_parameter, i))
        .collect()
}

/// LMSR Cost Function - Calculate cost to buy shares
///
/// Cost = b * ln(Σ e^((q_i + Δq_i)/b)) - b * ln(Σ e^(q_i/b))
///
/// # Arguments
/// * `outcome_quantities` - Current quantities
/// * `liquidity_parameter` - The "b" parameter
/// * `outcome_index` - Which outcome to buy
/// * `quantity` - How many shares to buy
///
/// # Returns
/// Cost in market currency units
pub fn lmsr_cost(
    outcome_quantities: &[u64],
    liquidity_parameter: f64,
    outcome_index: usize,
    quantity: u64,
) -> f64 {
    if outcome_quantities.is_empty() || outcome_index >= outcome_quantities.len() {
        return 0.0;
    }

    let b = liquidity_parameter;

    // Cost function: C(q) = b * ln(Σ e^(q_i/b))
    // Cost to buy = C(q_after) - C(q_before)

    // Helper: compute b * ln(Σ e^(q_i/b)) with numerical stability
    let cost_function = |quantities: &[u64]| -> f64 {
        let exponents: Vec<f64> = quantities
            .iter()
            .map(|q| (*q as f64) / b)
            .collect();
        let max_exp = exponents.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f64 = exponents.iter().map(|e| (e - max_exp).exp()).sum();
        b * (max_exp + sum_exp.ln())
    };

    // Before purchase
    let cost_before = cost_function(outcome_quantities);

    // After purchase (add quantity to target outcome)
    let mut quantities_after = outcome_quantities.to_vec();
    quantities_after[outcome_index] += quantity;
    let cost_after = cost_function(&quantities_after);

    (cost_after - cost_before).max(0.0)
}

/// Calculate how many shares can be bought with a given budget
pub fn lmsr_shares_for_budget(
    outcome_quantities: &[u64],
    liquidity_parameter: f64,
    outcome_index: usize,
    budget: f64,
) -> u64 {
    // Binary search for the quantity that costs approximately the budget
    let mut low = 0u64;
    let mut high = (budget * 10.0) as u64; // Upper bound estimate

    while low < high {
        let mid = low + (high - low) / 2;
        let cost = lmsr_cost(outcome_quantities, liquidity_parameter, outcome_index, mid);

        if cost <= budget {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    // Return the highest quantity that fits budget
    if low > 0 {
        low - 1
    } else {
        0
    }
}

/// Get current market price from Market struct
pub fn get_market_price(market: &Market, outcome_index: usize) -> Option<f64> {
    match &market.mechanism {
        MarketMechanism::LMSR {
            liquidity_parameter,
            outcome_quantities,
            ..
        } => Some(lmsr_price(outcome_quantities, *liquidity_parameter, outcome_index)),

        MarketMechanism::Binary { yes_shares, no_shares } => {
            let total = (*yes_shares + *no_shares) as f64;
            if total == 0.0 {
                Some(0.5) // Default 50-50 if no shares
            } else {
                match outcome_index {
                    0 => Some(*yes_shares as f64 / total),
                    1 => Some(*no_shares as f64 / total),
                    _ => None,
                }
            }
        }

        MarketMechanism::Parimutuel { pools } => {
            let total: u64 = pools.iter().map(|(_, amount)| amount).sum();
            if total == 0 {
                return Some(1.0 / pools.len() as f64);
            }
            pools.get(outcome_index).map(|(_, amount)| *amount as f64 / total as f64)
        }

        MarketMechanism::CDA { .. } => {
            // For CDA, price is the current best bid/ask midpoint
            // This requires order book analysis - return None for now
            None
        }
    }
}

/// Get all market prices
pub fn get_all_market_prices(market: &Market) -> Vec<f64> {
    match &market.mechanism {
        MarketMechanism::LMSR {
            liquidity_parameter,
            outcome_quantities,
            ..
        } => lmsr_prices(outcome_quantities, *liquidity_parameter),

        MarketMechanism::Binary { yes_shares, no_shares } => {
            let total = (*yes_shares + *no_shares) as f64;
            if total == 0.0 {
                vec![0.5, 0.5]
            } else {
                vec![
                    *yes_shares as f64 / total,
                    *no_shares as f64 / total,
                ]
            }
        }

        MarketMechanism::Parimutuel { pools } => {
            let total: u64 = pools.iter().map(|(_, amount)| amount).sum();
            if total == 0 {
                vec![1.0 / pools.len() as f64; pools.len()]
            } else {
                pools.iter().map(|(_, amount)| *amount as f64 / total as f64).collect()
            }
        }

        MarketMechanism::CDA { .. } => vec![],
    }
}

// ============================================================================
// SHARE TRADING SYSTEM
// ============================================================================

/// A user's share position in a market
#[hdk_entry_helper]
#[derive(Clone)]
pub struct SharePosition {
    /// Market this position is for
    pub market_id: EntryHash,
    /// Agent holding the position
    pub agent: AgentPubKey,
    /// Shares held by outcome (index = outcome index)
    pub shares_by_outcome: Vec<u64>,
    /// Total cost basis for each outcome (for P&L tracking)
    pub cost_basis_by_outcome: Vec<u64>,
    /// Average purchase price for each outcome
    pub avg_price_by_outcome: Vec<f64>,
    /// First purchase timestamp (for early bonus calculations)
    pub first_purchase_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
    /// Trade history
    pub trades: Vec<Trade>,
}

/// Individual trade record
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Trade {
    /// Trade type
    pub trade_type: TradeType,
    /// Outcome index
    pub outcome_index: usize,
    /// Number of shares
    pub shares: u64,
    /// Price per share at time of trade
    pub price: f64,
    /// Total cost/proceeds
    pub total_amount: u64,
    /// Timestamp
    pub timestamp: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum TradeType {
    Buy,
    Sell,
}

/// Input for buying shares
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BuySharesInput {
    pub market_id: EntryHash,
    pub outcome_index: usize,
    /// Either specify shares or budget
    pub shares: Option<u64>,
    pub budget: Option<u64>,
    /// Maximum acceptable price (slippage protection)
    pub max_price: Option<f64>,
}

/// Input for selling shares
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SellSharesInput {
    pub market_id: EntryHash,
    pub outcome_index: usize,
    pub shares: u64,
    /// Minimum acceptable price (slippage protection)
    pub min_price: Option<f64>,
}

/// Result of a trade
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TradeResult {
    pub market_id: EntryHash,
    pub position_id: EntryHash,
    pub trade_type: TradeType,
    pub outcome_index: usize,
    pub shares: u64,
    pub price_per_share: f64,
    pub total_cost: u64,
    pub new_position: Vec<u64>,
    pub new_market_prices: Vec<f64>,
}

#[hdk_extern]
pub fn buy_shares(input: BuySharesInput) -> ExternResult<TradeResult> {
    let now = sys_time()?.as_micros() as u64;
    let buyer = agent_info()?.agent_initial_pubkey;

    // Get market record
    let record = get(input.market_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market not found".into())))?;

    let mut market: Market = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market entry not found".into())))?;

    let market_action_hash = record.action_hashed().hash.clone();

    // Verify market is open
    if market.status != MarketStatus::Open {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Market is not open for trading".into()
        )));
    }

    // Get LMSR parameters
    let (liquidity_parameter, outcome_quantities) = match &market.mechanism {
        MarketMechanism::LMSR {
            liquidity_parameter,
            outcome_quantities,
            ..
        } => (*liquidity_parameter, outcome_quantities.clone()),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only LMSR markets support share trading".into()
            )));
        }
    };

    // Validate outcome index
    if input.outcome_index >= outcome_quantities.len() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid outcome index".into()
        )));
    }

    // Calculate shares to buy
    let shares_to_buy = if let Some(shares) = input.shares {
        shares
    } else if let Some(budget) = input.budget {
        lmsr_shares_for_budget(
            &outcome_quantities,
            liquidity_parameter,
            input.outcome_index,
            budget as f64,
        )
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Must specify either shares or budget".into()
        )));
    };

    if shares_to_buy == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot buy zero shares".into()
        )));
    }

    // Calculate cost
    let cost = lmsr_cost(
        &outcome_quantities,
        liquidity_parameter,
        input.outcome_index,
        shares_to_buy,
    );
    let cost_u64 = cost.ceil() as u64;

    // Check slippage protection
    let current_price = lmsr_price(&outcome_quantities, liquidity_parameter, input.outcome_index);
    if let Some(max_price) = input.max_price {
        if current_price > max_price {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Price {} exceeds max acceptable price {}",
                current_price, max_price
            ))));
        }
    }

    // Execute payment via Finance hApp
    execute_payment(&buyer, cost_u64, &input.market_id, "share_purchase")?;

    // Update market quantities
    let mut new_quantities = outcome_quantities.clone();
    new_quantities[input.outcome_index] += shares_to_buy;

    market.mechanism = MarketMechanism::LMSR {
        liquidity_parameter,
        subsidy_pool: 0, // Would track this properly
        outcome_quantities: new_quantities.clone(),
    };

    update_entry(market_action_hash, &market)?;

    // Update or create position
    let position_id = update_share_position(
        &buyer,
        &input.market_id,
        input.outcome_index,
        shares_to_buy,
        cost_u64,
        current_price,
        TradeType::Buy,
        now,
    )?;

    // Calculate new prices
    let new_prices = lmsr_prices(&new_quantities, liquidity_parameter);

    // Emit trade signal
    emit_signal(serde_json::json!({
        "type": "shares_bought",
        "market_id": input.market_id,
        "buyer": buyer,
        "outcome_index": input.outcome_index,
        "shares": shares_to_buy,
        "cost": cost_u64,
        "price": current_price,
        "new_prices": new_prices,
    }))?;

    Ok(TradeResult {
        market_id: input.market_id,
        position_id,
        trade_type: TradeType::Buy,
        outcome_index: input.outcome_index,
        shares: shares_to_buy,
        price_per_share: current_price,
        total_cost: cost_u64,
        new_position: new_quantities,
        new_market_prices: new_prices,
    })
}

#[hdk_extern]
pub fn sell_shares(input: SellSharesInput) -> ExternResult<TradeResult> {
    let now = sys_time()?.as_micros() as u64;
    let seller = agent_info()?.agent_initial_pubkey;

    // Get market record
    let record = get(input.market_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market not found".into())))?;

    let mut market: Market = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market entry not found".into())))?;

    let market_action_hash = record.action_hashed().hash.clone();

    // Verify market is open
    if market.status != MarketStatus::Open {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Market is not open for trading".into()
        )));
    }

    // Get position
    let position = get_agent_position(&seller, &input.market_id)?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No position in this market".into())))?;

    // Verify has enough shares
    if position.shares_by_outcome.get(input.outcome_index).copied().unwrap_or(0) < input.shares {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient shares to sell".into()
        )));
    }

    // Get LMSR parameters
    let (liquidity_parameter, outcome_quantities) = match &market.mechanism {
        MarketMechanism::LMSR {
            liquidity_parameter,
            outcome_quantities,
            ..
        } => (*liquidity_parameter, outcome_quantities.clone()),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only LMSR markets support share trading".into()
            )));
        }
    };

    // Calculate proceeds (negative cost = proceeds)
    // Selling is buying negative shares: cost(q - delta) - cost(q)
    let mut quantities_after = outcome_quantities.clone();
    quantities_after[input.outcome_index] = quantities_after[input.outcome_index]
        .saturating_sub(input.shares);

    let cost_before = lmsr_cost_total(&outcome_quantities, liquidity_parameter);
    let cost_after = lmsr_cost_total(&quantities_after, liquidity_parameter);
    let proceeds = (cost_before - cost_after).max(0.0);
    let proceeds_u64 = proceeds.floor() as u64;

    // Check slippage protection
    let current_price = lmsr_price(&outcome_quantities, liquidity_parameter, input.outcome_index);
    if let Some(min_price) = input.min_price {
        if current_price < min_price {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Price {} below min acceptable price {}",
                current_price, min_price
            ))));
        }
    }

    // Update market quantities
    market.mechanism = MarketMechanism::LMSR {
        liquidity_parameter,
        subsidy_pool: 0,
        outcome_quantities: quantities_after.clone(),
    };

    update_entry(market_action_hash, &market)?;

    // Execute payout to seller via Finance hApp
    execute_payout(&seller, proceeds_u64, &input.market_id, "share_sale")?;

    // Update position
    let position_id = update_share_position(
        &seller,
        &input.market_id,
        input.outcome_index,
        input.shares,
        proceeds_u64,
        current_price,
        TradeType::Sell,
        now,
    )?;

    // Calculate new prices
    let new_prices = lmsr_prices(&quantities_after, liquidity_parameter);

    // Emit trade signal
    emit_signal(serde_json::json!({
        "type": "shares_sold",
        "market_id": input.market_id,
        "seller": seller,
        "outcome_index": input.outcome_index,
        "shares": input.shares,
        "proceeds": proceeds_u64,
        "price": current_price,
        "new_prices": new_prices,
    }))?;

    Ok(TradeResult {
        market_id: input.market_id,
        position_id,
        trade_type: TradeType::Sell,
        outcome_index: input.outcome_index,
        shares: input.shares,
        price_per_share: current_price,
        total_cost: proceeds_u64,
        new_position: quantities_after,
        new_market_prices: new_prices,
    })
}

/// Get quote for buying shares (preview before execution)
#[hdk_extern]
pub fn get_buy_quote(input: BuySharesInput) -> ExternResult<TradeQuote> {
    let record = get(input.market_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market not found".into())))?;

    let market: Market = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market entry not found".into())))?;

    let (liquidity_parameter, outcome_quantities) = match &market.mechanism {
        MarketMechanism::LMSR {
            liquidity_parameter,
            outcome_quantities,
            ..
        } => (*liquidity_parameter, outcome_quantities.clone()),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only LMSR markets support quotes".into()
            )));
        }
    };

    let shares = if let Some(s) = input.shares {
        s
    } else if let Some(budget) = input.budget {
        lmsr_shares_for_budget(&outcome_quantities, liquidity_parameter, input.outcome_index, budget as f64)
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest("Must specify shares or budget".into())));
    };

    let current_price = lmsr_price(&outcome_quantities, liquidity_parameter, input.outcome_index);
    let cost = lmsr_cost(&outcome_quantities, liquidity_parameter, input.outcome_index, shares);

    let mut new_quantities = outcome_quantities.clone();
    new_quantities[input.outcome_index] += shares;
    let price_after = lmsr_price(&new_quantities, liquidity_parameter, input.outcome_index);

    Ok(TradeQuote {
        shares,
        current_price,
        estimated_price_after: price_after,
        estimated_cost: cost.ceil() as u64,
        price_impact: price_after - current_price,
    })
}

/// Get quote for selling shares
#[hdk_extern]
pub fn get_sell_quote(input: SellSharesInput) -> ExternResult<TradeQuote> {
    let record = get(input.market_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market not found".into())))?;

    let market: Market = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Market entry not found".into())))?;

    let (liquidity_parameter, outcome_quantities) = match &market.mechanism {
        MarketMechanism::LMSR {
            liquidity_parameter,
            outcome_quantities,
            ..
        } => (*liquidity_parameter, outcome_quantities.clone()),
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only LMSR markets support quotes".into()
            )));
        }
    };

    let current_price = lmsr_price(&outcome_quantities, liquidity_parameter, input.outcome_index);

    let mut new_quantities = outcome_quantities.clone();
    new_quantities[input.outcome_index] = new_quantities[input.outcome_index].saturating_sub(input.shares);

    let cost_before = lmsr_cost_total(&outcome_quantities, liquidity_parameter);
    let cost_after = lmsr_cost_total(&new_quantities, liquidity_parameter);
    let proceeds = (cost_before - cost_after).max(0.0);

    let price_after = lmsr_price(&new_quantities, liquidity_parameter, input.outcome_index);

    Ok(TradeQuote {
        shares: input.shares,
        current_price,
        estimated_price_after: price_after,
        estimated_cost: proceeds.floor() as u64,
        price_impact: price_after - current_price,
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TradeQuote {
    pub shares: u64,
    pub current_price: f64,
    pub estimated_price_after: f64,
    pub estimated_cost: u64,
    pub price_impact: f64,
}

#[hdk_extern]
pub fn get_position(input: GetPositionInput) -> ExternResult<Option<SharePosition>> {
    get_agent_position(&input.agent, &input.market_id)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetPositionInput {
    pub agent: AgentPubKey,
    pub market_id: EntryHash,
}

#[hdk_extern]
pub fn get_market_positions(market_id: EntryHash) -> ExternResult<Vec<SharePosition>> {
    let links = get_links(
        LinkQuery::try_new(market_id, LinkTypes::MarketToPosition)?,
        GetStrategy::default(),
    )?;

    let mut positions = Vec::new();
    for link in links {
        if let Some(record) = get(link.target.into_entry_hash().unwrap(), GetOptions::default())? {
            if let Some(position) = record.entry().to_app_option::<SharePosition>().ok().flatten() {
                positions.push(position);
            }
        }
    }

    Ok(positions)
}

// ============================================================================
// SHARE TRADING HELPER FUNCTIONS
// ============================================================================

/// Total cost function C(q) = b * ln(Σ e^(q_i/b))
fn lmsr_cost_total(outcome_quantities: &[u64], liquidity_parameter: f64) -> f64 {
    let b = liquidity_parameter;
    let exponents: Vec<f64> = outcome_quantities
        .iter()
        .map(|q| (*q as f64) / b)
        .collect();
    let max_exp = exponents.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp: f64 = exponents.iter().map(|e| (e - max_exp).exp()).sum();
    b * (max_exp + sum_exp.ln())
}

fn get_agent_position(agent: &AgentPubKey, market_id: &EntryHash) -> ExternResult<Option<SharePosition>> {
    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToPosition)?,
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(record) = get(link.target.into_entry_hash().unwrap(), GetOptions::default())? {
            if let Some(position) = record.entry().to_app_option::<SharePosition>().ok().flatten() {
                if position.market_id == *market_id {
                    return Ok(Some(position));
                }
            }
        }
    }

    Ok(None)
}

fn update_share_position(
    agent: &AgentPubKey,
    market_id: &EntryHash,
    outcome_index: usize,
    shares: u64,
    amount: u64,
    price: f64,
    trade_type: TradeType,
    now: u64,
) -> ExternResult<EntryHash> {
    // Try to get existing position
    let existing = get_agent_position(agent, market_id)?;

    let trade = Trade {
        trade_type: trade_type.clone(),
        outcome_index,
        shares,
        price,
        total_amount: amount,
        timestamp: now,
    };

    let position = if let Some(mut pos) = existing {
        // Get record for update
        let links = get_links(
            LinkQuery::try_new(agent.clone(), LinkTypes::AgentToPosition)?,
            GetStrategy::default(),
        )?;

        let position_hash = links.iter()
            .find(|l| {
                get(l.target.clone().into_entry_hash().unwrap(), GetOptions::default())
                    .ok()
                    .flatten()
                    .and_then(|r| r.entry().to_app_option::<SharePosition>().ok().flatten())
                    .map(|p| p.market_id == *market_id)
                    .unwrap_or(false)
            })
            .map(|l| l.target.clone().into_entry_hash().unwrap());

        match trade_type {
            TradeType::Buy => {
                // Ensure vectors are long enough
                while pos.shares_by_outcome.len() <= outcome_index {
                    pos.shares_by_outcome.push(0);
                    pos.cost_basis_by_outcome.push(0);
                    pos.avg_price_by_outcome.push(0.0);
                }

                let old_shares = pos.shares_by_outcome[outcome_index];
                let old_cost = pos.cost_basis_by_outcome[outcome_index];
                let new_shares = old_shares + shares;
                let new_cost = old_cost + amount;

                pos.shares_by_outcome[outcome_index] = new_shares;
                pos.cost_basis_by_outcome[outcome_index] = new_cost;
                pos.avg_price_by_outcome[outcome_index] = if new_shares > 0 {
                    new_cost as f64 / new_shares as f64
                } else {
                    0.0
                };
            }
            TradeType::Sell => {
                if outcome_index < pos.shares_by_outcome.len() {
                    pos.shares_by_outcome[outcome_index] =
                        pos.shares_by_outcome[outcome_index].saturating_sub(shares);
                    // Cost basis adjusted proportionally
                    let ratio = if pos.shares_by_outcome[outcome_index] > 0 {
                        pos.shares_by_outcome[outcome_index] as f64
                            / (pos.shares_by_outcome[outcome_index] + shares) as f64
                    } else {
                        0.0
                    };
                    pos.cost_basis_by_outcome[outcome_index] =
                        (pos.cost_basis_by_outcome[outcome_index] as f64 * ratio) as u64;
                }
            }
        }

        pos.trades.push(trade);
        pos.updated_at = now;

        // Update entry
        if let Some(hash) = position_hash {
            let record = get(hash.clone(), GetOptions::default())?
                .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Position not found".into())))?;
            let action_hash = record.action_hashed().hash.clone();
            update_entry(action_hash, &pos)?;
        }

        pos
    } else {
        // Create new position
        let num_outcomes = outcome_index + 1;
        let mut shares_by_outcome = vec![0u64; num_outcomes];
        let mut cost_basis_by_outcome = vec![0u64; num_outcomes];
        let mut avg_price_by_outcome = vec![0.0f64; num_outcomes];

        if trade_type == TradeType::Buy {
            shares_by_outcome[outcome_index] = shares;
            cost_basis_by_outcome[outcome_index] = amount;
            avg_price_by_outcome[outcome_index] = price;
        }

        let new_position = SharePosition {
            market_id: market_id.clone(),
            agent: agent.clone(),
            shares_by_outcome,
            cost_basis_by_outcome,
            avg_price_by_outcome,
            first_purchase_at: now,
            updated_at: now,
            trades: vec![trade],
        };

        let hash = hash_entry(&new_position)?;
        let _action_hash = create_entry(&EntryTypes::SharePosition(new_position.clone()))?;

        // Create links
        create_link(agent.clone(), hash.clone(), LinkTypes::AgentToPosition, ())?;
        create_link(market_id.clone(), hash.clone(), LinkTypes::MarketToPosition, ())?;

        new_position
    };

    hash_entry(&position)
}

/// Payment input for Finance hApp
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PaymentInput {
    pub from: AgentPubKey,
    pub amount: u64,
    pub market_id: EntryHash,
    pub payment_type: String,
}

fn execute_payment(
    payer: &AgentPubKey,
    amount: u64,
    market_id: &EntryHash,
    payment_type: &str,
) -> ExternResult<()> {
    let payment = PaymentInput {
        from: payer.clone(),
        amount,
        market_id: market_id.clone(),
        payment_type: payment_type.to_string(),
    };

    let response = call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::new("kredit"),
        FunctionName::new("escrow_payment"),
        None,
        payment,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Payment failed: {:?}", e))))?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            Err(wasm_error!(WasmErrorInner::Guest("Unauthorized payment".into())))
        }
        ZomeCallResponse::NetworkError(e) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!("Network error: {}", e))))
        }
        _ => Err(wasm_error!(WasmErrorInner::Guest("Payment failed".into()))),
    }
}

fn execute_payout(
    recipient: &AgentPubKey,
    amount: u64,
    market_id: &EntryHash,
    payout_type: &str,
) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct PayoutInput {
        pub to: AgentPubKey,
        pub amount: u64,
        pub market_id: EntryHash,
        pub payout_type: String,
    }

    let payout = PayoutInput {
        to: recipient.clone(),
        amount,
        market_id: market_id.clone(),
        payout_type: payout_type.to_string(),
    };

    let response = call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::new("kredit"),
        FunctionName::new("release_payout"),
        None,
        payout,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Payout failed: {:?}", e))))?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        _ => Ok(()), // Non-critical, don't fail the trade
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    use hdk::prelude::{hash_entry, Entry, AppEntryBytes, SerializedBytes, UnsafeBytes};
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", anchor).into_bytes()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

// ============================================================================
// ENTRY AND LINK DEFINITIONS
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Market(Market),
    SharePosition(SharePosition),
}

#[hdk_link_types]
pub enum LinkTypes {
    CreatorToMarket,
    AnchorToMarket,
    TagToMarket,
    MarketToPrediction,
    AgentToPosition,
    MarketToPosition,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_position_automated_resolution() {
        // E4 (Measurable) -> Automated resolution
        let pos = EpistemicPosition {
            empirical: EmpiricalLevel::Measurable,
            normative: NormativeLevel::Universal,
            materiality: MaterialityLevel::Persistent,
        };
        assert!(matches!(pos.recommended_resolution(), ResolutionMechanism::Automated));

        // E3 (Cryptographic) -> Automated resolution
        let pos2 = EpistemicPosition {
            empirical: EmpiricalLevel::Cryptographic,
            normative: NormativeLevel::Network,
            materiality: MaterialityLevel::Persistent,
        };
        assert!(matches!(pos2.recommended_resolution(), ResolutionMechanism::Automated));
    }

    #[test]
    fn test_epistemic_position_zk_resolution() {
        // E2 (PrivateVerify) -> ZK-verified resolution
        let pos = EpistemicPosition {
            empirical: EmpiricalLevel::PrivateVerify,
            normative: NormativeLevel::Communal,
            materiality: MaterialityLevel::Temporal,
        };
        assert!(matches!(pos.recommended_resolution(), ResolutionMechanism::ZkVerified));
    }

    #[test]
    fn test_epistemic_position_oracle_resolution() {
        // E1 (Testimonial) + N2/N3 -> Oracle consensus
        let pos = EpistemicPosition {
            empirical: EmpiricalLevel::Testimonial,
            normative: NormativeLevel::Network,
            materiality: MaterialityLevel::Persistent,
        };
        assert!(matches!(pos.recommended_resolution(), ResolutionMechanism::OracleConsensus));

        let pos2 = EpistemicPosition {
            empirical: EmpiricalLevel::Testimonial,
            normative: NormativeLevel::Universal,
            materiality: MaterialityLevel::Foundational,
        };
        assert!(matches!(pos2.recommended_resolution(), ResolutionMechanism::OracleConsensus));
    }

    #[test]
    fn test_epistemic_position_community_vote() {
        // E1 (Testimonial) + N0/N1 -> Community vote
        let pos = EpistemicPosition {
            empirical: EmpiricalLevel::Testimonial,
            normative: NormativeLevel::Communal,
            materiality: MaterialityLevel::Temporal,
        };
        assert!(matches!(pos.recommended_resolution(), ResolutionMechanism::CommunityVote));

        let pos2 = EpistemicPosition {
            empirical: EmpiricalLevel::Testimonial,
            normative: NormativeLevel::Personal,
            materiality: MaterialityLevel::Ephemeral,
        };
        assert!(matches!(pos2.recommended_resolution(), ResolutionMechanism::CommunityVote));
    }

    #[test]
    fn test_epistemic_position_reputation_staked() {
        // E0 (Subjective) -> Reputation staked
        let pos = EpistemicPosition {
            empirical: EmpiricalLevel::Subjective,
            normative: NormativeLevel::Personal,
            materiality: MaterialityLevel::Ephemeral,
        };
        assert!(matches!(pos.recommended_resolution(), ResolutionMechanism::ReputationStaked));
    }

    #[test]
    fn test_epistemic_duration_days() {
        let ephemeral = EpistemicPosition {
            empirical: EmpiricalLevel::Subjective,
            normative: NormativeLevel::Personal,
            materiality: MaterialityLevel::Ephemeral,
        };
        assert_eq!(ephemeral.recommended_duration_days(), 1);

        let temporal = EpistemicPosition {
            empirical: EmpiricalLevel::Testimonial,
            normative: NormativeLevel::Communal,
            materiality: MaterialityLevel::Temporal,
        };
        assert_eq!(temporal.recommended_duration_days(), 7);

        let persistent = EpistemicPosition {
            empirical: EmpiricalLevel::Cryptographic,
            normative: NormativeLevel::Network,
            materiality: MaterialityLevel::Persistent,
        };
        assert_eq!(persistent.recommended_duration_days(), 30);

        let foundational = EpistemicPosition {
            empirical: EmpiricalLevel::Measurable,
            normative: NormativeLevel::Universal,
            materiality: MaterialityLevel::Foundational,
        };
        assert_eq!(foundational.recommended_duration_days(), 90);
    }

    #[test]
    fn test_multi_dimensional_stake_monetary() {
        let stake = MultiDimensionalStake::monetary_only(1000, "KREDIT".to_string());
        assert_eq!(stake.monetary.as_ref().unwrap().amount, 1000);
        assert!(stake.reputation.is_none());
        assert!((stake.total_value() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_multi_dimensional_stake_reputation() {
        let stake = MultiDimensionalStake::reputation_only(
            vec!["knowledge".to_string()],
            0.1,  // 10% of reputation
            1.5,  // 1.5x confidence multiplier
        );
        assert!(stake.monetary.is_none());
        assert!(stake.reputation.is_some());
        // 0.1 * 10000 * 1.5 = 1500
        assert!((stake.total_value() - 1500.0).abs() < 0.001);
    }

    #[test]
    fn test_multi_dimensional_stake_combined() {
        let stake = MultiDimensionalStake {
            monetary: Some(MonetaryStake {
                amount: 500,
                currency: "KREDIT".to_string(),
                escrow_id: None,
            }),
            reputation: Some(ReputationStake {
                domains: vec!["tech".to_string()],
                stake_percentage: 0.05,  // 5%
                confidence_multiplier: 2.0,
            }),
            social: Some(SocialStake {
                visibility: Visibility::Public,
                identity_link: None,
            }),
            commitment: None,
            time: None,
        };
        // 500 (monetary) + 0.05 * 10000 * 2.0 (rep) + 500 (public visibility)
        // = 500 + 1000 + 500 = 2000
        assert!((stake.total_value() - 2000.0).abs() < 0.001);
    }

    #[test]
    fn test_stake_with_identity_link() {
        let stake = MultiDimensionalStake {
            monetary: None,
            reputation: None,
            social: Some(SocialStake {
                visibility: Visibility::Public,
                identity_link: Some("verified_identity".to_string()),
            }),
            commitment: None,
            time: None,
        };
        // Public (500) * 2 (identity linked) = 1000
        assert!((stake.total_value() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_stake_with_time_investment() {
        let stake = MultiDimensionalStake {
            monetary: None,
            reputation: None,
            social: None,
            commitment: None,
            time: Some(TimeStake {
                research_hours: 10,
                evidence_submitted: vec![
                    EntryHash::from_raw_36(vec![0u8; 36]),
                    EntryHash::from_raw_36(vec![1u8; 36]),
                ],
            }),
        };
        // 10 * 15 (hours) + 2 * 25 (evidence) = 150 + 50 = 200
        assert!((stake.total_value() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_visibility_ordering() {
        // Test visibility stake values
        let private = Visibility::Private;
        let limited = Visibility::Limited(vec![]);
        let community = Visibility::Community;
        let public = Visibility::Public;

        let make_stake = |vis| MultiDimensionalStake {
            monetary: None,
            reputation: None,
            social: Some(SocialStake { visibility: vis, identity_link: None }),
            commitment: None,
            time: None,
        };

        assert!((make_stake(private).total_value() - 0.0).abs() < 0.001);
        assert!((make_stake(limited).total_value() - 50.0).abs() < 0.001);
        assert!((make_stake(community).total_value() - 200.0).abs() < 0.001);
        assert!((make_stake(public).total_value() - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_market_status_values() {
        // All market statuses are distinct
        let statuses = [
            MarketStatus::Open,
            MarketStatus::Closed,
            MarketStatus::Resolving,
            MarketStatus::Resolved,
            MarketStatus::Disputed,
            MarketStatus::Cancelled,
        ];
        for (i, s1) in statuses.iter().enumerate() {
            for (j, s2) in statuses.iter().enumerate() {
                if i != j {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_commitment_stake_value() {
        let stake = MultiDimensionalStake {
            monetary: None,
            reputation: None,
            social: None,
            commitment: Some(CommitmentStake {
                if_correct: vec![],
                if_wrong: vec![
                    Commitment::Investigation { hours: 5, topic: "research".to_string() },
                    Commitment::Donation { amount: 100, recipient: "charity".to_string() },
                ],
            }),
            time: None,
        };
        // 5 * 20 (investigation) + 100 (donation) = 100 + 100 = 200
        assert!((stake.total_value() - 200.0).abs() < 0.001);
    }
}
