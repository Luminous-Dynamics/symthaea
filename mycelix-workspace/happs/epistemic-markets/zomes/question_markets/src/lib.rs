// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Question Markets - Trade on What's Worth Knowing
//!
//! Revolutionary mechanism: Before predicting answers, discover which questions matter.
//!
//! Traditional prediction markets assume we know what to predict.
//! Question markets surface *which uncertainties matter most*, directing
//! collective attention to high-value information gaps.

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// QUESTION VALUE DIMENSIONS
// ============================================================================

/// Multi-dimensional value assessment for questions
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QuestionValue {
    /// How many decisions depend on knowing this answer?
    pub decision_relevance: f64,

    /// How much uncertainty exists currently? (high = more valuable to resolve)
    pub current_uncertainty: f64,

    /// How many agents have expressed interest?
    pub curiosity_signals: u64,

    /// How actionable is the answer likely to be?
    pub actionability: f64,

    /// Time sensitivity: how much does value decay over time?
    pub time_sensitivity: f64,

    /// Domain importance: foundational vs peripheral knowledge
    pub domain_importance: f64,
}

impl QuestionValue {
    /// Calculate composite question value
    pub fn composite_value(&self) -> f64 {
        // Weighted combination of value dimensions
        let weights = QuestionValueWeights::default();

        self.decision_relevance * weights.decision
            + self.current_uncertainty * weights.uncertainty
            + (self.curiosity_signals as f64).ln().max(0.0) * weights.curiosity
            + self.actionability * weights.actionability
            + self.time_sensitivity * weights.time_sensitivity
            + self.domain_importance * weights.domain_importance
    }

    /// Initial value estimate from question metadata
    pub fn estimate_initial(
        question: &str,
        proposer_matl: f64,
        initial_curiosity: u64,
    ) -> Self {
        // Heuristics for initial estimation
        let _word_count = question.split_whitespace().count();
        let has_quantifiable = question.contains("how many")
            || question.contains("what percentage")
            || question.contains("by when");

        Self {
            decision_relevance: 0.5, // Unknown initially
            current_uncertainty: 0.8, // Assume high uncertainty for new questions
            curiosity_signals: initial_curiosity,
            actionability: if has_quantifiable { 0.7 } else { 0.4 },
            time_sensitivity: 0.5,
            domain_importance: proposer_matl * 0.8, // Trust proposer's judgment
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QuestionValueWeights {
    pub decision: f64,
    pub uncertainty: f64,
    pub curiosity: f64,
    pub actionability: f64,
    pub time_sensitivity: f64,
    pub domain_importance: f64,
}

impl Default for QuestionValueWeights {
    fn default() -> Self {
        Self {
            decision: 0.25,
            uncertainty: 0.20,
            curiosity: 0.15,
            actionability: 0.15,
            time_sensitivity: 0.10,
            domain_importance: 0.15,
        }
    }
}

// ============================================================================
// QUESTION MARKET STRUCTURE
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum QuestionStatus {
    /// Question is open for value trading
    Trading,
    /// Question reached threshold, prediction market spawned
    Spawned,
    /// Question expired without reaching threshold
    Expired,
    /// Question merged with similar question
    Merged { into: EntryHash },
    /// Question rejected (duplicate, invalid, etc.)
    Rejected { reason: String },
}

/// A market for question value, not answer probability
#[hdk_entry_helper]
#[derive(Clone)]
pub struct QuestionMarket {
    /// Unique identifier
    pub id: EntryHash,

    /// The question being evaluated
    pub question_text: String,

    /// Detailed context and why this question matters
    pub context: String,

    /// Who proposed this question
    pub proposer: AgentPubKey,

    /// Proposer's MATL score at time of proposal
    pub proposer_matl: f64,

    /// Multi-dimensional value assessment
    pub value: QuestionValue,

    /// Current market price (expected value of knowing the answer)
    pub current_price: f64,

    /// Total shares outstanding
    pub total_shares: u64,

    /// Price history for charting
    pub price_history: Vec<PricePoint>,

    /// Threshold to spawn a prediction market
    pub spawn_threshold: f64,

    /// If spawned, the resulting prediction market
    pub spawned_market: Option<EntryHash>,

    /// Related questions (for duplicate detection and linking)
    pub related_questions: Vec<EntryHash>,

    /// Domain tags
    pub domains: Vec<String>,

    /// Lifecycle
    pub created_at: u64,
    pub expires_at: u64,
    pub status: QuestionStatus,

    /// Curiosity signals (lightweight interest indicators)
    pub curiosity_count: u64,

    /// Discussion/reasoning about question value
    pub discussion_entries: Vec<EntryHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PricePoint {
    pub timestamp: u64,
    pub price: f64,
    pub volume: u64,
}

/// A position in a question market
#[hdk_entry_helper]
#[derive(Clone)]
pub struct QuestionPosition {
    /// The question market
    pub question_market: EntryHash,

    /// The holder
    pub holder: AgentPubKey,

    /// Number of value shares held
    pub shares: u64,

    /// Average purchase price
    pub avg_price: f64,

    /// Acquisition timestamp
    pub acquired_at: u64,

    /// Reasoning for position (optional, for reputation)
    pub reasoning: Option<String>,
}

/// Lightweight signal of interest (no stake required)
#[hdk_entry_helper]
#[derive(Clone)]
pub struct CuriositySignal {
    pub question_market: EntryHash,
    pub signaler: AgentPubKey,
    pub timestamp: u64,
    /// Optional: why are you curious?
    pub reason: Option<String>,
}

// ============================================================================
// SPAWN CONFIGURATION
// ============================================================================

/// Configuration for spawning prediction markets from questions
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpawnConfig {
    /// Minimum price to trigger spawn
    pub price_threshold: f64,

    /// Minimum unique traders
    pub min_traders: u64,

    /// Minimum curiosity signals
    pub min_curiosity: u64,

    /// How much of the question market value becomes prediction market subsidy
    pub liquidity_transfer_ratio: f64,

    /// Domains that auto-spawn (no threshold needed)
    pub auto_spawn_domains: Vec<String>,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            price_threshold: 100.0,
            min_traders: 5,
            min_curiosity: 10,
            liquidity_transfer_ratio: 0.5,
            auto_spawn_domains: vec!["governance".to_string(), "safety".to_string()],
        }
    }
}

/// Request to spawn a prediction market from a question
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarketSpawnRequest {
    pub question: String,
    pub initial_liquidity: u64,
    pub suggested_outcomes: Vec<String>,
    pub resolution_deadline: u64,
    pub source_question_market: EntryHash,
}

// ============================================================================
// ZOME FUNCTIONS
// ============================================================================

#[hdk_extern]
pub fn propose_question(input: ProposeQuestionInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let proposer = agent_info()?.agent_initial_pubkey;

    // Get proposer's MATL score (bridge call in production)
    let proposer_matl = get_matl_score(&proposer)?;

    // Initial value estimate
    let value = QuestionValue::estimate_initial(
        &input.question_text,
        proposer_matl,
        1, // Self counts as first curiosity signal
    );

    let initial_price = value.composite_value();

    let question_market = QuestionMarket {
        id: EntryHash::from_raw_36(vec![0; 36]),
        question_text: input.question_text,
        context: input.context,
        proposer: proposer.clone(),
        proposer_matl,
        value,
        current_price: initial_price,
        total_shares: 0,
        price_history: vec![PricePoint {
            timestamp: now,
            price: initial_price,
            volume: 0,
        }],
        spawn_threshold: input.spawn_threshold.unwrap_or(100.0),
        spawned_market: None,
        related_questions: vec![],
        domains: input.domains,
        created_at: now,
        expires_at: input.expires_at.unwrap_or(now + 30 * 24 * 60 * 60 * 1_000_000), // 30 days default
        status: QuestionStatus::Trading,
        curiosity_count: 1,
        discussion_entries: vec![],
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let entry_hash = hash_entry(&question_market)?;
    let _action_hash = create_entry(&EntryTypes::QuestionMarket(question_market))?;

    // Link to proposer
    create_link(
        proposer,
        entry_hash.clone(),
        LinkTypes::ProposerToQuestion,
        (),
    )?;

    // Link to all_questions anchor
    let anchor = anchor_hash("all_questions")?;
    create_link(anchor, entry_hash.clone(), LinkTypes::AnchorToQuestion, ())?;

    Ok(entry_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProposeQuestionInput {
    pub question_text: String,
    pub context: String,
    pub domains: Vec<String>,
    pub spawn_threshold: Option<f64>,
    pub expires_at: Option<u64>,
}

#[hdk_extern]
pub fn signal_curiosity(input: SignalCuriosityInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let signaler = agent_info()?.agent_initial_pubkey;

    // Create curiosity signal
    let signal = CuriositySignal {
        question_market: input.question_market.clone(),
        signaler: signaler.clone(),
        timestamp: now,
        reason: input.reason,
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let signal_entry_hash = hash_entry(&signal)?;
    let _signal_action_hash = create_entry(&EntryTypes::CuriositySignal(signal))?;

    // Update question market curiosity count
    // Get the record for action hash
    if let Some(record) = get(input.question_market.clone(), GetOptions::default())? {
        if let Some(mut market) = record.entry().to_app_option::<QuestionMarket>().ok().flatten() {
            market.curiosity_count += 1;
            market.value.curiosity_signals += 1;

            // Recalculate price based on new curiosity
            market.current_price = market.value.composite_value();

            // Add price point
            market.price_history.push(PricePoint {
                timestamp: now,
                price: market.current_price,
                volume: 0,
            });

            // Check if we should spawn
            check_spawn_threshold(&mut market)?;

            // HDK 0.6: update_entry takes ActionHash
            let action_hash = record.action_hashed().hash.clone();
            update_entry(action_hash, &market)?;
        }
    }

    Ok(signal_entry_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignalCuriosityInput {
    pub question_market: EntryHash,
    pub reason: Option<String>,
}

#[hdk_extern]
pub fn buy_question_shares(input: BuySharesInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let buyer = agent_info()?.agent_initial_pubkey;

    // Get the record for action hash
    let record = get(input.question_market.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Question market not found".into())))?;

    let mut market: QuestionMarket = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Question market entry not found".into())))?;

    if market.status != QuestionStatus::Trading {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Question market is not trading".into()
        )));
    }

    // Simple AMM pricing: price increases with shares bought
    // Using constant product formula variant
    let price_impact = calculate_price_impact(market.total_shares, input.shares);
    let execution_price = market.current_price * (1.0 + price_impact);

    // Update market state
    market.total_shares += input.shares;
    market.current_price = execution_price;

    // Record price point
    market.price_history.push(PricePoint {
        timestamp: now,
        price: market.current_price,
        volume: input.shares,
    });

    // Check spawn threshold
    check_spawn_threshold(&mut market)?;

    // HDK 0.6: update_entry takes ActionHash
    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &market)?;

    // Create position entry
    let position = QuestionPosition {
        question_market: input.question_market,
        holder: buyer.clone(),
        shares: input.shares,
        avg_price: execution_price,
        acquired_at: now,
        reasoning: input.reasoning,
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let position_entry_hash = hash_entry(&position)?;
    let _position_action_hash = create_entry(&EntryTypes::QuestionPosition(position))?;

    // Link position to buyer
    create_link(buyer, position_entry_hash.clone(), LinkTypes::HolderToPosition, ())?;

    Ok(position_entry_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BuySharesInput {
    pub question_market: EntryHash,
    pub shares: u64,
    pub max_price: Option<f64>,
    pub reasoning: Option<String>,
}

#[hdk_extern]
pub fn get_question_market(hash: EntryHash) -> ExternResult<Option<QuestionMarket>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let market: QuestionMarket = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Question market not found".into()))
                })?;
            Ok(Some(market))
        }
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn list_question_markets(_: ()) -> ExternResult<Vec<QuestionMarket>> {
    let anchor = anchor_hash("all_questions")?;
    let links =
        get_links(LinkQuery::try_new(anchor, LinkTypes::AnchorToQuestion)?, GetStrategy::default())?;

    let mut markets = Vec::new();
    for link in links {
        if let Some(market) = get_question_market(link.target.into_entry_hash().unwrap())? {
            markets.push(market);
        }
    }

    // Sort by current price (most valuable questions first)
    markets.sort_by(|a, b| {
        b.current_price
            .partial_cmp(&a.current_price)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(markets)
}

#[hdk_extern]
pub fn get_top_questions(limit: usize) -> ExternResult<Vec<QuestionMarket>> {
    let mut markets = list_question_markets(())?;
    markets.truncate(limit);
    Ok(markets)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Input for creating a spawned prediction market
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SpawnMarketInput {
    pub question: String,
    pub description: String,
    pub outcomes: Vec<String>,
    pub closes_at: u64,
    pub initial_liquidity: u64,
    pub source_question_market: EntryHash,
    pub domains: Vec<String>,
}

fn check_spawn_threshold(market: &mut QuestionMarket) -> ExternResult<()> {
    let config = SpawnConfig::default();

    // Check if any auto-spawn domain applies
    let auto_spawn = market
        .domains
        .iter()
        .any(|d| config.auto_spawn_domains.contains(d));

    let should_spawn = auto_spawn
        || (market.current_price >= config.price_threshold
            && market.curiosity_count >= config.min_curiosity);

    if should_spawn && market.spawned_market.is_none() {
        // Create spawn request for markets zome
        let spawn_input = SpawnMarketInput {
            question: market.question_text.clone(),
            description: market.context.clone(),
            outcomes: vec!["Yes".to_string(), "No".to_string()],
            closes_at: market.expires_at + 30 * 24 * 60 * 60 * 1_000_000, // +30 days
            initial_liquidity: (market.current_price * config.liquidity_transfer_ratio) as u64,
            source_question_market: market.id.clone(),
            domains: market.domains.clone(),
        };

        // Bridge call to markets zome to spawn the prediction market
        // HDK 0.6: Handle ZomeCallResponse variants
        let response = call(
            CallTargetCell::Local,
            ZomeName::new("markets"),
            FunctionName::new("create_market_from_question"),
            None,
            spawn_input.clone(),
        )
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to spawn market: {:?}", e))))?;

        let spawned_hash: EntryHash = match response {
            ZomeCallResponse::Ok(extern_io) => extern_io.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode spawned market hash: {:?}", e))))?,
            ZomeCallResponse::Unauthorized(_, _, _, _) => {
                return Err(wasm_error!(WasmErrorInner::Guest("Unauthorized to spawn market".into())));
            }
            ZomeCallResponse::NetworkError(e) => {
                return Err(wasm_error!(WasmErrorInner::Guest(format!("Network error: {}", e))));
            }
            ZomeCallResponse::CountersigningSession(e) => {
                return Err(wasm_error!(WasmErrorInner::Guest(format!("Countersigning error: {}", e))));
            }
            ZomeCallResponse::AuthenticationFailed(_, _) => {
                return Err(wasm_error!(WasmErrorInner::Guest("Authentication failed".into())));
            }
        };

        market.spawned_market = Some(spawned_hash.clone());
        market.status = QuestionStatus::Spawned;

        // Create spawn request record for tracking
        let spawn_request = MarketSpawnRequest {
            question: market.question_text.clone(),
            initial_liquidity: (market.current_price * config.liquidity_transfer_ratio) as u64,
            suggested_outcomes: vec!["Yes".to_string(), "No".to_string()],
            resolution_deadline: market.expires_at + 30 * 24 * 60 * 60 * 1_000_000,
            source_question_market: market.id.clone(),
        };

        // Emit signal for UI updates
        emit_signal(serde_json::json!({
            "type": "question_spawned",
            "question_market": market.id,
            "question": market.question_text,
            "spawned_market": spawned_hash,
            "spawn_request": spawn_request,
        }))?;
    }

    Ok(())
}

fn calculate_price_impact(current_shares: u64, new_shares: u64) -> f64 {
    // Bonding curve: price increases with square root of shares
    let before = (current_shares as f64).sqrt();
    let after = ((current_shares + new_shares) as f64).sqrt();
    (after - before) / before.max(1.0)
}

/// MATL score response from identity hApp
#[derive(Serialize, Deserialize, Debug, Clone)]
struct MatlScoreResponse {
    pub quality: f64,
    pub consistency: f64,
    pub reputation: f64,
    pub composite: f64,
    pub domain_scores: Vec<(String, f64)>,
}

fn get_matl_score(agent: &AgentPubKey) -> ExternResult<f64> {
    // Bridge call to MATL system in identity hApp
    // HDK 0.6: Handle ZomeCallResponse variants
    let response = call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::new("matl"),
        FunctionName::new("get_agent_matl_score"),
        None,
        agent.clone(),
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to get MATL score: {:?}", e))))?;

    let matl_result: Result<MatlScoreResponse, _> = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode(),
        _ => Err(holochain_serialized_bytes::SerializedBytesError::Deserialize(
            "Call failed".to_string()
        )),
    };

    // Return the composite score or a conservative default
    Ok(matl_result
        .map(|score| score.composite)
        .unwrap_or(0.5)) // Conservative default for unknown agents
}

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
    QuestionMarket(QuestionMarket),
    QuestionPosition(QuestionPosition),
    CuriositySignal(CuriositySignal),
}

#[hdk_link_types]
pub enum LinkTypes {
    ProposerToQuestion,
    AnchorToQuestion,
    HolderToPosition,
    QuestionToDiscussion,
    RelatedQuestions,
}
