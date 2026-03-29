// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resolution Zome - MATL-Weighted Oracle Networks
//!
//! Revolutionary resolution mechanism leveraging Mycelix's 45% Byzantine tolerance.
//!
//! Key innovations:
//! - MATL-weighted oracle voting (quality × consistency × reputation)
//! - Adaptive Byzantine detection (45% tolerance vs classical 33%)
//! - Epistemic position-driven resolution mechanisms
//! - Multi-tier escalation for disputed outcomes
//! - Reputation staking for attestors

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// MARKET DATA (From Markets Zome)
// ============================================================================

/// Market data retrieved from markets zome for resolution configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarketData {
    pub id: EntryHash,
    pub question: String,
    pub outcomes: Vec<String>,
    pub epistemic_level: u8, // 0-4 for E0-E4
    pub domain: String,
    pub closes_at: u64,
    pub resolution_config_override: Option<ResolutionConfig>,
}

// ============================================================================
// MATL INTEGRATION (From Mycelix Core)
// ============================================================================

/// MATL composite score components
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MatlScore {
    /// Quality of contributions (0.0 - 1.0)
    pub quality: f64,
    /// Consistency over time (0.0 - 1.0)
    pub consistency: f64,
    /// Historical reputation (0.0 - 1.0)
    pub reputation: f64,
    /// Composite score using default weights (0.4, 0.3, 0.3)
    pub composite: f64,
    /// Domain-specific scores
    pub domain_scores: Vec<(String, f64)>,
}

impl MatlScore {
    pub fn calculate_composite(&self) -> f64 {
        0.4 * self.quality + 0.3 * self.consistency + 0.3 * self.reputation
    }

    /// Weight for oracle voting
    pub fn oracle_weight(&self) -> f64 {
        // Oracles weighted by composite score squared (amplify quality differences)
        self.composite.powi(2)
    }
}

// ============================================================================
// ORACLE NETWORK
// ============================================================================

/// An oracle that can vote on market outcomes
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Oracle {
    pub agent: AgentPubKey,
    pub matl_score: MatlScore,
    /// Domains of expertise
    pub domains: Vec<String>,
    /// Track record in resolution
    pub resolution_accuracy: f64,
    /// Number of resolutions participated in
    pub resolution_count: u64,
    /// Reputation staked on current votes
    pub reputation_at_stake: f64,
}

/// A vote on market outcome
#[hdk_entry_helper]
#[derive(Clone)]
pub struct OracleVote {
    pub market_id: EntryHash,
    pub oracle: AgentPubKey,
    pub outcome: String,
    pub confidence: f64,
    pub reasoning: String,
    pub evidence: Vec<EvidenceLink>,
    pub timestamp: u64,
    /// MATL weight at time of vote
    pub matl_weight: f64,
    /// Reputation staked on this vote
    pub reputation_stake: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvidenceLink {
    pub source: String,
    pub url: Option<String>,
    pub description: String,
    pub epistemic_level: u8, // 0-4 matching EmpiricalLevel
}

// ============================================================================
// RESOLUTION MECHANISMS
// ============================================================================

/// Resolution configuration based on epistemic position
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResolutionConfig {
    /// Minimum number of oracles required
    pub min_oracles: u32,
    /// Minimum MATL score for oracle participation
    pub min_oracle_matl: f64,
    /// Consensus threshold (weighted votes / total weight)
    pub consensus_threshold: f64,
    /// Time allowed for voting
    pub voting_period_seconds: u64,
    /// Maximum dispute escalations
    pub max_escalations: u32,
    /// Byzantine tolerance (Mycelix revolutionary: 0.45)
    pub byzantine_tolerance: f64,
}

impl ResolutionConfig {
    /// Standard config for E3/E4 (cryptographic/measurable) outcomes
    pub fn automated() -> Self {
        Self {
            min_oracles: 1,
            min_oracle_matl: 0.5,
            consensus_threshold: 0.0, // Single source sufficient
            voting_period_seconds: 0,
            max_escalations: 1,
            byzantine_tolerance: 0.45,
        }
    }

    /// Config for E1/E2 (testimonial/private) outcomes
    pub fn oracle_consensus() -> Self {
        Self {
            min_oracles: 5,
            min_oracle_matl: 0.7,
            consensus_threshold: 0.67,
            voting_period_seconds: 7 * 24 * 60 * 60, // 7 days
            max_escalations: 2,
            byzantine_tolerance: 0.45,
        }
    }

    /// Config for E0 (subjective) outcomes
    pub fn community_vote() -> Self {
        Self {
            min_oracles: 10,
            min_oracle_matl: 0.5,
            consensus_threshold: 0.6,
            voting_period_seconds: 14 * 24 * 60 * 60, // 14 days
            max_escalations: 3,
            byzantine_tolerance: 0.45,
        }
    }
}

/// Current state of resolution process
/// Note: f64 fields prevent deriving Eq, use PartialEq only
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResolutionState {
    /// Not yet started
    Pending,
    /// Voting in progress
    Voting {
        started_at: u64,
        deadline: u64,
    },
    /// Consensus reached
    Resolved {
        outcome: String,
        confidence: f64,
        resolved_at: u64,
    },
    /// Disputed, under review
    Disputed {
        disputed_at: u64,
        disputer: AgentPubKey,
        reason: String,
        escalation_level: u32,
    },
    /// Escalated to governance
    EscalatedToGovernance {
        proposal_id: EntryHash,
    },
    /// Failed to resolve (no consensus)
    Failed {
        reason: String,
    },
}

/// Resolution process for a market
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ResolutionProcess {
    pub market_id: EntryHash,
    pub config: ResolutionConfig,
    pub state: ResolutionState,
    pub votes: Vec<EntryHash>, // Links to OracleVote entries
    pub outcome_tallies: Vec<OutcomeTally>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutcomeTally {
    pub outcome: String,
    /// Sum of MATL-weighted votes
    pub weighted_votes: f64,
    /// Number of oracles voting for this outcome
    pub oracle_count: u64,
    /// Average confidence of votes
    pub avg_confidence: f64,
    /// Evidence quality score
    pub evidence_score: f64,
}

// ============================================================================
// BYZANTINE DETECTION (45% Tolerance)
// ============================================================================

/// Byzantine detection using Mycelix's adaptive threshold system
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ByzantineAnalysis {
    /// Detected potentially byzantine oracles
    pub suspicious_oracles: Vec<SuspiciousOracle>,
    /// Network health assessment
    pub network_health: f64,
    /// Recommended action
    pub recommendation: ByzantineRecommendation,
    /// Effective tolerance based on network state
    pub effective_tolerance: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SuspiciousOracle {
    pub agent: AgentPubKey,
    pub suspicion_score: f64,
    pub reasons: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum ByzantineRecommendation {
    /// Proceed normally
    Proceed,
    /// Increase oracle requirements
    IncreaseQuorum,
    /// Extend voting period
    ExtendVoting,
    /// Escalate to governance
    Escalate,
    /// Halt resolution
    Halt,
}

impl ByzantineAnalysis {
    /// Analyze votes for Byzantine behavior
    pub fn analyze(
        votes: &[OracleVote],
        config: &ResolutionConfig,
    ) -> Self {
        let mut suspicious = Vec::new();

        // Group votes by outcome
        let mut outcome_groups: std::collections::HashMap<String, Vec<&OracleVote>> =
            std::collections::HashMap::new();
        for vote in votes {
            outcome_groups
                .entry(vote.outcome.clone())
                .or_default()
                .push(vote);
        }

        // Calculate total weight
        let total_weight: f64 = votes.iter().map(|v| v.matl_weight).sum();

        // Find minority outcomes
        for (outcome, outcome_votes) in &outcome_groups {
            let outcome_weight: f64 = outcome_votes.iter().map(|v| v.matl_weight).sum();
            let outcome_ratio = outcome_weight / total_weight;

            // Check for coordinated low-confidence voting (potential manipulation)
            let avg_confidence: f64 = outcome_votes.iter().map(|v| v.confidence).sum::<f64>()
                / outcome_votes.len() as f64;

            // Suspicion: high weight + low confidence = possible manipulation
            if outcome_ratio > 0.2 && avg_confidence < 0.5 {
                for vote in outcome_votes {
                    suspicious.push(SuspiciousOracle {
                        agent: vote.oracle.clone(),
                        suspicion_score: (1.0 - avg_confidence) * outcome_ratio,
                        reasons: vec![format!(
                            "Low confidence ({:.2}) on minority outcome '{}'",
                            vote.confidence, outcome
                        )],
                    });
                }
            }
        }

        // Check for voting patterns (same timestamp, similar reasoning)
        // This is simplified; real implementation would be more sophisticated

        let suspicious_weight: f64 = suspicious.iter().map(|s| s.suspicion_score).sum();
        let suspicious_ratio = suspicious_weight / (votes.len() as f64).max(1.0);

        // Network health: inverse of suspicion
        let network_health = 1.0 - suspicious_ratio.min(1.0);

        // Effective tolerance adjusts based on network health
        let effective_tolerance = config.byzantine_tolerance * network_health.max(0.5);

        // Recommendation based on analysis
        let recommendation = if suspicious_ratio > config.byzantine_tolerance {
            ByzantineRecommendation::Halt
        } else if suspicious_ratio > config.byzantine_tolerance * 0.8 {
            ByzantineRecommendation::Escalate
        } else if suspicious_ratio > config.byzantine_tolerance * 0.5 {
            ByzantineRecommendation::IncreaseQuorum
        } else if suspicious_ratio > config.byzantine_tolerance * 0.3 {
            ByzantineRecommendation::ExtendVoting
        } else {
            ByzantineRecommendation::Proceed
        };

        Self {
            suspicious_oracles: suspicious,
            network_health,
            recommendation,
            effective_tolerance,
        }
    }
}

// ============================================================================
// ZOME FUNCTIONS
// ============================================================================

#[hdk_extern]
pub fn start_resolution(market_id: EntryHash) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;

    // Get market from markets zome to determine resolution config based on epistemic level
    // HDK 0.6: Handle ZomeCallResponse variants
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("markets"),
        FunctionName::new("get_market"),
        None,
        market_id.clone(),
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to get market data: {:?}", e))))?;

    let market_data: Option<MarketData> = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().ok(),
        _ => None,
    };

    // Determine resolution config based on market's epistemic level
    let config = if let Some(market) = market_data {
        // Use override if provided, otherwise select based on epistemic level
        market.resolution_config_override.unwrap_or_else(|| {
            match market.epistemic_level {
                // E3/E4: Cryptographic or measurable outcomes - automated resolution
                3 | 4 => ResolutionConfig::automated(),
                // E1/E2: Testimonial or private outcomes - oracle consensus
                1 | 2 => ResolutionConfig::oracle_consensus(),
                // E0: Subjective outcomes - community vote
                _ => ResolutionConfig::community_vote(),
            }
        })
    } else {
        // Fallback to oracle consensus if market not found
        ResolutionConfig::oracle_consensus()
    };

    let process = ResolutionProcess {
        market_id: market_id.clone(),
        config: config.clone(),
        state: ResolutionState::Voting {
            started_at: now,
            deadline: now + config.voting_period_seconds * 1_000_000,
        },
        votes: vec![],
        outcome_tallies: vec![],
        created_at: now,
        updated_at: now,
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let entry_hash = hash_entry(&process)?;
    let _action_hash = create_entry(&EntryTypes::ResolutionProcess(process))?;

    // Link to market
    create_link(market_id.clone(), entry_hash.clone(), LinkTypes::MarketToResolution, ())?;

    // Emit signal for oracle notification
    emit_signal(serde_json::json!({
        "type": "resolution_started",
        "market_id": market_id,
        "resolution_id": entry_hash,
        "voting_deadline": now + config.voting_period_seconds * 1_000_000,
    }))?;

    Ok(entry_hash)
}

#[hdk_extern]
pub fn submit_oracle_vote(input: SubmitVoteInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let oracle = agent_info()?.agent_initial_pubkey;

    // Get resolution process record for action hash
    let record = get(input.resolution_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution not found".into())))?;

    let mut process: ResolutionProcess = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution entry not found".into())))?;

    // Verify voting is open
    match &process.state {
        ResolutionState::Voting { deadline, .. } => {
            if now > *deadline {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Voting period has ended".into()
                )));
            }
        }
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Resolution is not in voting state".into()
            )));
        }
    }

    // Get oracle's MATL score
    let matl_score = get_oracle_matl(&oracle)?;

    // Verify minimum MATL requirement
    if matl_score.composite < process.config.min_oracle_matl {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "MATL score {:.2} below minimum {:.2}",
            matl_score.composite, process.config.min_oracle_matl
        ))));
    }

    // Calculate reputation stake based on confidence
    let reputation_stake = input.confidence * input.reputation_stake_pct.unwrap_or(0.1);

    let vote = OracleVote {
        market_id: process.market_id.clone(),
        oracle: oracle.clone(),
        outcome: input.outcome.clone(),
        confidence: input.confidence,
        reasoning: input.reasoning,
        evidence: input.evidence,
        timestamp: now,
        matl_weight: matl_score.oracle_weight(),
        reputation_stake,
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let vote_entry_hash = hash_entry(&vote)?;
    let _vote_action_hash = create_entry(&EntryTypes::OracleVote(vote.clone()))?;

    // Add vote to process
    process.votes.push(vote_entry_hash.clone());
    process.updated_at = now;

    // Update tallies
    update_outcome_tallies(&mut process, &vote);

    // Check for consensus
    check_consensus(&mut process)?;

    // HDK 0.6: update_entry takes ActionHash
    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &process)?;

    // Link vote to oracle
    create_link(oracle, vote_entry_hash.clone(), LinkTypes::OracleToVote, ())?;

    Ok(vote_entry_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubmitVoteInput {
    pub resolution_id: EntryHash,
    pub outcome: String,
    pub confidence: f64,
    pub reasoning: String,
    pub evidence: Vec<EvidenceLink>,
    pub reputation_stake_pct: Option<f64>,
}

#[hdk_extern]
pub fn finalize_resolution(resolution_id: EntryHash) -> ExternResult<ResolutionResult> {
    let now = sys_time()?.as_micros() as u64;

    // Get record for action hash
    let record = get(resolution_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution not found".into())))?;

    let mut process: ResolutionProcess = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution entry not found".into())))?;

    // HDK 0.6: Get action hash for updates
    let action_hash = record.action_hashed().hash.clone();

    // Collect votes
    let votes = collect_votes(&process)?;

    // Run Byzantine analysis
    let byzantine_analysis = ByzantineAnalysis::analyze(&votes, &process.config);

    // Check Byzantine recommendation
    match byzantine_analysis.recommendation {
        ByzantineRecommendation::Halt => {
            process.state = ResolutionState::Failed {
                reason: "Byzantine activity detected above tolerance threshold".into(),
            };
            update_entry(action_hash, &process)?;
            return Ok(ResolutionResult {
                success: false,
                outcome: None,
                confidence: 0.0,
                byzantine_analysis,
            });
        }
        ByzantineRecommendation::Escalate => {
            // Escalate to governance
            let proposal_id = escalate_to_governance(&process)?;
            process.state = ResolutionState::EscalatedToGovernance { proposal_id };
            update_entry(action_hash, &process)?;
            return Ok(ResolutionResult {
                success: false,
                outcome: None,
                confidence: 0.0,
                byzantine_analysis,
            });
        }
        _ => {}
    }

    // Find winning outcome
    let total_weight: f64 = process.outcome_tallies.iter().map(|t| t.weighted_votes).sum();
    let winning_tally = process
        .outcome_tallies
        .iter()
        .max_by(|a, b| {
            a.weighted_votes
                .partial_cmp(&b.weighted_votes)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some(winner) = winning_tally {
        let consensus_ratio = winner.weighted_votes / total_weight;

        if consensus_ratio >= process.config.consensus_threshold {
            process.state = ResolutionState::Resolved {
                outcome: winner.outcome.clone(),
                confidence: consensus_ratio,
                resolved_at: now,
            };

            update_entry(action_hash.clone(), &process)?;

            // Emit resolution signal
            emit_signal(serde_json::json!({
                "type": "market_resolved",
                "market_id": process.market_id,
                "outcome": winner.outcome,
                "confidence": consensus_ratio,
            }))?;

            // Update oracle reputations based on accuracy
            update_oracle_reputations(&process, &winner.outcome)?;

            return Ok(ResolutionResult {
                success: true,
                outcome: Some(winner.outcome.clone()),
                confidence: consensus_ratio,
                byzantine_analysis,
            });
        }
    }

    // No consensus reached
    process.state = ResolutionState::Failed {
        reason: "Consensus threshold not reached".into(),
    };
    update_entry(action_hash, &process)?;

    Ok(ResolutionResult {
        success: false,
        outcome: None,
        confidence: 0.0,
        byzantine_analysis,
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResolutionResult {
    pub success: bool,
    pub outcome: Option<String>,
    pub confidence: f64,
    pub byzantine_analysis: ByzantineAnalysis,
}

// ============================================================================
// PAYOUT DISTRIBUTION SYSTEM
// ============================================================================

/// Individual predictor payout calculation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictorPayout {
    /// Agent who made the prediction
    pub agent: AgentPubKey,
    /// Outcome they predicted
    pub predicted_outcome: String,
    /// Number of shares held in winning outcome
    pub winning_shares: u64,
    /// Number of shares held in losing outcomes
    pub losing_shares: u64,
    /// Total KREDIT to receive
    pub payout_amount: u64,
    /// Breakdown of payout calculation
    pub calculation_details: PayoutCalculation,
}

/// Detailed breakdown of payout calculation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PayoutCalculation {
    /// Base payout from winning shares
    pub base_payout: u64,
    /// MATL reputation bonus (0-15% based on reputation)
    pub matl_bonus: u64,
    /// Early prediction bonus (0-10% for predictions in first 25% of market)
    pub early_bonus: u64,
    /// Wisdom seed activation bonus (for high-confidence correct predictions)
    pub wisdom_bonus: u64,
    /// Total before fees
    pub gross_payout: u64,
    /// Market maker fee (typically 1-2%)
    pub market_fee: u64,
    /// Protocol fee (0.5%)
    pub protocol_fee: u64,
    /// Net payout after fees
    pub net_payout: u64,
}

/// Complete payout record for a resolved market
#[hdk_entry_helper]
#[derive(Clone)]
pub struct PayoutRecord {
    /// Market that was resolved
    pub market_id: EntryHash,
    /// Resolution that triggered payout
    pub resolution_id: EntryHash,
    /// Winning outcome
    pub winning_outcome: String,
    /// Resolution confidence (affects payout certainty)
    pub resolution_confidence: f64,
    /// Total pool to distribute
    pub total_pool: u64,
    /// Individual payouts
    pub payouts: Vec<PredictorPayout>,
    /// Market maker share
    pub market_maker_share: u64,
    /// Protocol treasury share
    pub protocol_share: u64,
    /// Timestamp of payout calculation
    pub calculated_at: u64,
    /// Status of payout execution
    pub execution_status: PayoutExecutionStatus,
}

/// Status of payout execution
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PayoutExecutionStatus {
    /// Calculated but not yet executed
    Pending,
    /// Currently being processed
    Processing,
    /// All payouts executed successfully
    Completed,
    /// Partial execution (some failed)
    PartiallyCompleted {
        successful: u32,
        failed: u32,
    },
    /// Execution failed
    Failed {
        reason: String,
    },
}

/// Input for payout calculation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CalculatePayoutsInput {
    pub resolution_id: EntryHash,
    /// Optional: override payout parameters
    pub config_override: Option<PayoutConfig>,
}

/// Configuration for payout calculation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PayoutConfig {
    /// Market maker fee percentage (default: 1.5%)
    pub market_maker_fee_pct: f64,
    /// Protocol fee percentage (default: 0.5%)
    pub protocol_fee_pct: f64,
    /// Maximum MATL bonus percentage (default: 15%)
    pub max_matl_bonus_pct: f64,
    /// Maximum early prediction bonus percentage (default: 10%)
    pub max_early_bonus_pct: f64,
    /// Wisdom seed activation threshold (default: 0.8 confidence)
    pub wisdom_threshold: f64,
    /// Wisdom bonus percentage (default: 5%)
    pub wisdom_bonus_pct: f64,
}

impl Default for PayoutConfig {
    fn default() -> Self {
        Self {
            market_maker_fee_pct: 0.015,
            protocol_fee_pct: 0.005,
            max_matl_bonus_pct: 0.15,
            max_early_bonus_pct: 0.10,
            wisdom_threshold: 0.8,
            wisdom_bonus_pct: 0.05,
        }
    }
}

/// Share position for a predictor (retrieved from predictions zome)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictorPosition {
    pub agent: AgentPubKey,
    pub outcome: String,
    pub shares: u64,
    pub avg_purchase_price: f64,
    pub first_purchase_at: u64,
    pub prediction_confidence: f64,
}

/// Market pool data (retrieved from markets zome)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarketPoolData {
    pub market_id: EntryHash,
    pub total_pool: u64,
    pub outcome_pools: Vec<(String, u64)>,
    pub market_maker: Option<AgentPubKey>,
    pub created_at: u64,
    pub closes_at: u64,
}

#[hdk_extern]
pub fn calculate_payouts(input: CalculatePayoutsInput) -> ExternResult<PayoutRecord> {
    let now = sys_time()?.as_micros() as u64;
    let config = input.config_override.unwrap_or_default();

    // Get resolution process
    let resolution_record = get(input.resolution_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution not found".into())))?;

    let process: ResolutionProcess = resolution_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution entry not found".into())))?;

    // Verify resolution is complete
    let (winning_outcome, resolution_confidence) = match &process.state {
        ResolutionState::Resolved { outcome, confidence, .. } => {
            (outcome.clone(), *confidence)
        }
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot calculate payouts: resolution not complete".into()
            )));
        }
    };

    // Get market pool data from markets zome
    let pool_data = get_market_pool_data(&process.market_id)?;

    // Get all predictor positions from predictions zome
    let positions = get_predictor_positions(&process.market_id)?;

    // Calculate individual payouts
    let mut payouts = Vec::new();
    let mut total_winning_shares: u64 = 0;

    // First pass: count total winning shares
    for position in &positions {
        if position.outcome == winning_outcome {
            total_winning_shares += position.shares;
        }
    }

    // Calculate distributable pool (after fees)
    let gross_pool = pool_data.total_pool;
    let market_maker_share = (gross_pool as f64 * config.market_maker_fee_pct) as u64;
    let protocol_share = (gross_pool as f64 * config.protocol_fee_pct) as u64;
    let distributable_pool = gross_pool - market_maker_share - protocol_share;

    // Second pass: calculate individual payouts
    for position in &positions {
        let is_winner = position.outcome == winning_outcome;

        if !is_winner {
            // Losing positions get nothing, but record them for transparency
            payouts.push(PredictorPayout {
                agent: position.agent.clone(),
                predicted_outcome: position.outcome.clone(),
                winning_shares: 0,
                losing_shares: position.shares,
                payout_amount: 0,
                calculation_details: PayoutCalculation {
                    base_payout: 0,
                    matl_bonus: 0,
                    early_bonus: 0,
                    wisdom_bonus: 0,
                    gross_payout: 0,
                    market_fee: 0,
                    protocol_fee: 0,
                    net_payout: 0,
                },
            });
            continue;
        }

        // Calculate base payout proportional to share ownership
        let share_ratio = position.shares as f64 / total_winning_shares as f64;
        let base_payout = (distributable_pool as f64 * share_ratio) as u64;

        // Get MATL score for bonus calculation
        let matl_score = get_oracle_matl(&position.agent)?;
        let matl_bonus = (base_payout as f64 * config.max_matl_bonus_pct * matl_score.composite) as u64;

        // Calculate early prediction bonus
        let market_duration = pool_data.closes_at.saturating_sub(pool_data.created_at);
        let prediction_timing = position.first_purchase_at.saturating_sub(pool_data.created_at);
        let early_ratio = 1.0 - (prediction_timing as f64 / market_duration as f64).min(1.0);
        let early_bonus = if early_ratio > 0.75 {
            // Prediction in first 25% of market gets full bonus
            (base_payout as f64 * config.max_early_bonus_pct * early_ratio) as u64
        } else {
            0
        };

        // Calculate wisdom seed bonus (high confidence correct predictions)
        let wisdom_bonus = if position.prediction_confidence >= config.wisdom_threshold {
            (base_payout as f64 * config.wisdom_bonus_pct) as u64
        } else {
            0
        };

        let gross_payout = base_payout + matl_bonus + early_bonus + wisdom_bonus;

        // Individual fee calculation (for transparency, though fees are pool-level)
        let individual_market_fee = (gross_payout as f64 * config.market_maker_fee_pct) as u64;
        let individual_protocol_fee = (gross_payout as f64 * config.protocol_fee_pct) as u64;
        let net_payout = gross_payout - individual_market_fee - individual_protocol_fee;

        payouts.push(PredictorPayout {
            agent: position.agent.clone(),
            predicted_outcome: position.outcome.clone(),
            winning_shares: position.shares,
            losing_shares: 0,
            payout_amount: net_payout,
            calculation_details: PayoutCalculation {
                base_payout,
                matl_bonus,
                early_bonus,
                wisdom_bonus,
                gross_payout,
                market_fee: individual_market_fee,
                protocol_fee: individual_protocol_fee,
                net_payout,
            },
        });
    }

    // Create payout record
    let payout_record = PayoutRecord {
        market_id: process.market_id.clone(),
        resolution_id: input.resolution_id.clone(),
        winning_outcome,
        resolution_confidence,
        total_pool: gross_pool,
        payouts,
        market_maker_share,
        protocol_share,
        calculated_at: now,
        execution_status: PayoutExecutionStatus::Pending,
    };

    // Store payout record
    let entry_hash = hash_entry(&payout_record)?;
    let _action_hash = create_entry(&EntryTypes::PayoutRecord(payout_record.clone()))?;

    // Link to resolution
    create_link(
        input.resolution_id.clone(),
        entry_hash.clone(),
        LinkTypes::ResolutionToPayout,
        (),
    )?;

    // Emit signal for payout calculation complete
    emit_signal(serde_json::json!({
        "type": "payouts_calculated",
        "market_id": process.market_id,
        "resolution_id": input.resolution_id,
        "payout_record_id": entry_hash,
        "total_payouts": payout_record.payouts.len(),
        "total_pool": gross_pool,
    }))?;

    Ok(payout_record)
}

#[hdk_extern]
pub fn execute_payouts(payout_record_id: EntryHash) -> ExternResult<PayoutExecutionResult> {
    let now = sys_time()?.as_micros() as u64;

    // Get payout record
    let record = get(payout_record_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Payout record not found".into())))?;

    let mut payout_record: PayoutRecord = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Payout record entry not found".into())))?;

    let action_hash = record.action_hashed().hash.clone();

    // Verify not already executed
    if payout_record.execution_status != PayoutExecutionStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Payouts already executed or in progress".into()
        )));
    }

    // Mark as processing
    payout_record.execution_status = PayoutExecutionStatus::Processing;
    update_entry(action_hash.clone(), &payout_record)?;

    // Execute individual payouts via Finance hApp bridge
    let mut successful = 0u32;
    let mut failed = 0u32;
    let mut execution_results = Vec::new();

    for payout in &payout_record.payouts {
        if payout.payout_amount == 0 {
            continue; // Skip zero payouts
        }

        let result = execute_kredit_transfer(
            &payout.agent,
            payout.payout_amount,
            &payout_record.market_id,
            "market_payout",
        );

        match result {
            Ok(_) => {
                successful += 1;
                execution_results.push(PayoutExecutionDetail {
                    agent: payout.agent.clone(),
                    amount: payout.payout_amount,
                    success: true,
                    error: None,
                    executed_at: now,
                });
            }
            Err(e) => {
                failed += 1;
                execution_results.push(PayoutExecutionDetail {
                    agent: payout.agent.clone(),
                    amount: payout.payout_amount,
                    success: false,
                    error: Some(format!("{:?}", e)),
                    executed_at: now,
                });
            }
        }
    }

    // Execute market maker share
    if let Some(market_maker) = get_market_maker(&payout_record.market_id)? {
        let _ = execute_kredit_transfer(
            &market_maker,
            payout_record.market_maker_share,
            &payout_record.market_id,
            "market_maker_fee",
        );
    }

    // Execute protocol share (to treasury)
    let _ = execute_protocol_fee_transfer(
        payout_record.protocol_share,
        &payout_record.market_id,
    );

    // Update execution status
    payout_record.execution_status = if failed == 0 {
        PayoutExecutionStatus::Completed
    } else if successful == 0 {
        PayoutExecutionStatus::Failed {
            reason: "All transfers failed".into(),
        }
    } else {
        PayoutExecutionStatus::PartiallyCompleted { successful, failed }
    };

    update_entry(action_hash, &payout_record)?;

    // Emit completion signal
    emit_signal(serde_json::json!({
        "type": "payouts_executed",
        "market_id": payout_record.market_id,
        "payout_record_id": payout_record_id,
        "successful": successful,
        "failed": failed,
        "status": format!("{:?}", payout_record.execution_status),
    }))?;

    Ok(PayoutExecutionResult {
        payout_record_id,
        successful,
        failed,
        execution_details: execution_results,
        status: payout_record.execution_status,
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PayoutExecutionResult {
    pub payout_record_id: EntryHash,
    pub successful: u32,
    pub failed: u32,
    pub execution_details: Vec<PayoutExecutionDetail>,
    pub status: PayoutExecutionStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PayoutExecutionDetail {
    pub agent: AgentPubKey,
    pub amount: u64,
    pub success: bool,
    pub error: Option<String>,
    pub executed_at: u64,
}

#[hdk_extern]
pub fn get_payout_record(hash: EntryHash) -> ExternResult<Option<PayoutRecord>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let payout: PayoutRecord = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Payout record not found".into()))
                })?;
            Ok(Some(payout))
        }
        None => Ok(None),
    }
}

// ============================================================================
// PAYOUT HELPER FUNCTIONS
// ============================================================================

fn get_market_pool_data(market_id: &EntryHash) -> ExternResult<MarketPoolData> {
    // Bridge call to markets zome
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("markets"),
        FunctionName::new("get_market_pool"),
        None,
        market_id.clone(),
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to get market pool: {:?}", e))))?;

    let pool_data: MarketPoolData = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode pool data: {:?}", e))))?,
        _ => {
            // Return default if call fails (for development/testing)
            MarketPoolData {
                market_id: market_id.clone(),
                total_pool: 0,
                outcome_pools: vec![],
                market_maker: None,
                created_at: 0,
                closes_at: 0,
            }
        }
    };

    Ok(pool_data)
}

fn get_predictor_positions(market_id: &EntryHash) -> ExternResult<Vec<PredictorPosition>> {
    // Bridge call to predictions zome
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("predictions"),
        FunctionName::new("get_market_positions"),
        None,
        market_id.clone(),
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to get positions: {:?}", e))))?;

    let positions: Vec<PredictorPosition> = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().unwrap_or_default(),
        _ => vec![],
    };

    Ok(positions)
}

fn get_market_maker(market_id: &EntryHash) -> ExternResult<Option<AgentPubKey>> {
    let pool_data = get_market_pool_data(market_id)?;
    Ok(pool_data.market_maker)
}

/// KREDIT transfer input for Finance hApp
#[derive(Serialize, Deserialize, Debug, Clone)]
struct KreditTransferInput {
    pub recipient: AgentPubKey,
    pub amount: u64,
    pub source_market: EntryHash,
    pub transfer_type: String,
    pub memo: String,
}

fn execute_kredit_transfer(
    recipient: &AgentPubKey,
    amount: u64,
    market_id: &EntryHash,
    transfer_type: &str,
) -> ExternResult<()> {
    let transfer_input = KreditTransferInput {
        recipient: recipient.clone(),
        amount,
        source_market: market_id.clone(),
        transfer_type: transfer_type.to_string(),
        memo: format!("Epistemic market payout: {}", transfer_type),
    };

    // Bridge call to Finance hApp
    let response = call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::new("kredit"),
        FunctionName::new("transfer"),
        None,
        transfer_input,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("KREDIT transfer failed: {:?}", e))))?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            Err(wasm_error!(WasmErrorInner::Guest("Unauthorized transfer".into())))
        }
        ZomeCallResponse::NetworkError(e) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!("Network error: {}", e))))
        }
        _ => Err(wasm_error!(WasmErrorInner::Guest("Transfer failed".into()))),
    }
}

/// Protocol fee transfer input
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ProtocolFeeInput {
    pub amount: u64,
    pub source_market: EntryHash,
    pub fee_type: String,
}

fn execute_protocol_fee_transfer(amount: u64, market_id: &EntryHash) -> ExternResult<()> {
    let fee_input = ProtocolFeeInput {
        amount,
        source_market: market_id.clone(),
        fee_type: "market_resolution".to_string(),
    };

    // Bridge call to Finance hApp treasury
    let response = call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::new("treasury"),
        FunctionName::new("receive_protocol_fee"),
        None,
        fee_input,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Protocol fee transfer failed: {:?}", e))))?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        _ => Ok(()), // Non-critical, don't fail payout
    }
}

#[hdk_extern]
pub fn dispute_resolution(input: DisputeInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let disputer = agent_info()?.agent_initial_pubkey;

    // Get record for action hash
    let record = get(input.resolution_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution not found".into())))?;

    let mut process: ResolutionProcess = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Resolution entry not found".into())))?;

    // Can only dispute resolved outcomes
    let current_escalation = match &process.state {
        ResolutionState::Resolved { .. } => 0,
        ResolutionState::Disputed {
            escalation_level, ..
        } => *escalation_level,
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot dispute: resolution not in valid state".into()
            )));
        }
    };

    if current_escalation >= process.config.max_escalations {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Maximum escalation level reached".into()
        )));
    }

    // Verify disputer has sufficient MATL
    let disputer_matl = get_oracle_matl(&disputer)?;
    if disputer_matl.composite < process.config.min_oracle_matl {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient MATL to dispute".into()
        )));
    }

    process.state = ResolutionState::Disputed {
        disputed_at: now,
        disputer: disputer.clone(),
        reason: input.reason.clone(),
        escalation_level: current_escalation + 1,
    };
    process.updated_at = now;

    // Reset votes for new round
    process.votes.clear();
    process.outcome_tallies.clear();

    // Increase oracle requirements for disputed resolution
    process.config.min_oracles = (process.config.min_oracles as f64 * 1.5) as u32;
    process.config.min_oracle_matl = (process.config.min_oracle_matl + 0.1).min(0.95);

    // HDK 0.6: update_entry takes ActionHash
    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &process)?;

    // Emit dispute signal
    emit_signal(serde_json::json!({
        "type": "resolution_disputed",
        "market_id": process.market_id,
        "disputer": disputer,
        "reason": input.reason,
        "escalation_level": current_escalation + 1,
    }))?;

    Ok(input.resolution_id)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisputeInput {
    pub resolution_id: EntryHash,
    pub reason: String,
    pub evidence: Vec<EvidenceLink>,
}

#[hdk_extern]
pub fn get_resolution_process(hash: EntryHash) -> ExternResult<Option<ResolutionProcess>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let process: ResolutionProcess = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Resolution not found".into()))
                })?;
            Ok(Some(process))
        }
        None => Ok(None),
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn update_outcome_tallies(process: &mut ResolutionProcess, vote: &OracleVote) {
    let existing = process
        .outcome_tallies
        .iter_mut()
        .find(|t| t.outcome == vote.outcome);

    if let Some(tally) = existing {
        let old_weight = tally.weighted_votes;
        let old_count = tally.oracle_count;

        tally.weighted_votes += vote.matl_weight;
        tally.oracle_count += 1;
        tally.avg_confidence =
            (tally.avg_confidence * old_count as f64 + vote.confidence) / (old_count + 1) as f64;

        // Update evidence score based on evidence quality
        let evidence_quality: f64 = vote
            .evidence
            .iter()
            .map(|e| e.epistemic_level as f64 / 4.0)
            .sum::<f64>()
            / vote.evidence.len().max(1) as f64;
        tally.evidence_score = (old_weight * tally.evidence_score + vote.matl_weight * evidence_quality)
            / (old_weight + vote.matl_weight);
    } else {
        let evidence_quality: f64 = vote
            .evidence
            .iter()
            .map(|e| e.epistemic_level as f64 / 4.0)
            .sum::<f64>()
            / vote.evidence.len().max(1) as f64;

        process.outcome_tallies.push(OutcomeTally {
            outcome: vote.outcome.clone(),
            weighted_votes: vote.matl_weight,
            oracle_count: 1,
            avg_confidence: vote.confidence,
            evidence_score: evidence_quality,
        });
    }
}

fn check_consensus(process: &mut ResolutionProcess) -> ExternResult<()> {
    let total_weight: f64 = process.outcome_tallies.iter().map(|t| t.weighted_votes).sum();

    for tally in &process.outcome_tallies {
        let ratio = tally.weighted_votes / total_weight;
        if ratio >= process.config.consensus_threshold
            && tally.oracle_count >= process.config.min_oracles as u64
        {
            let now = sys_time()?.as_micros() as u64;
            process.state = ResolutionState::Resolved {
                outcome: tally.outcome.clone(),
                confidence: ratio,
                resolved_at: now,
            };

            emit_signal(serde_json::json!({
                "type": "consensus_reached",
                "market_id": process.market_id,
                "outcome": tally.outcome,
                "confidence": ratio,
            }))?;

            break;
        }
    }

    Ok(())
}

fn collect_votes(process: &ResolutionProcess) -> ExternResult<Vec<OracleVote>> {
    let mut votes = Vec::new();
    for vote_hash in &process.votes {
        if let Some(record) = get(vote_hash.clone(), GetOptions::default())? {
            if let Some(vote) = record.entry().to_app_option::<OracleVote>().ok().flatten() {
                votes.push(vote);
            }
        }
    }
    Ok(votes)
}

fn get_oracle_matl(oracle: &AgentPubKey) -> ExternResult<MatlScore> {
    // Bridge call to MATL system in the identity hApp
    // HDK 0.6: Handle ZomeCallResponse variants
    let response = call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::new("matl"),
        FunctionName::new("get_agent_matl_score"),
        None,
        oracle.clone(),
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to get MATL score: {:?}", e))))?;

    let matl_result: Result<MatlScore, _> = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode(),
        _ => Err(holochain_serialized_bytes::SerializedBytesError::Deserialize(
            "Call failed".to_string()
        )),
    };

    // Return the MATL score or a default if the call fails
    Ok(matl_result.unwrap_or_else(|_| MatlScore {
        quality: 0.5,  // Conservative default
        consistency: 0.5,
        reputation: 0.5,
        composite: 0.5,
        domain_scores: vec![],
    }))
}

/// Input for creating a governance proposal for disputed resolution
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GovernanceEscalationInput {
    pub market_id: EntryHash,
    pub market_question: String,
    pub dispute_reason: String,
    pub current_tallies: Vec<OutcomeTally>,
    pub escalation_level: u32,
    pub proposal_type: String,
}

fn escalate_to_governance(process: &ResolutionProcess) -> ExternResult<EntryHash> {
    // Extract dispute reason from current state
    let (dispute_reason, escalation_level) = match &process.state {
        ResolutionState::Disputed { reason, escalation_level, .. } => {
            (reason.clone(), *escalation_level)
        }
        _ => ("Escalated due to Byzantine activity".to_string(), 0),
    };

    // Build escalation input for governance hApp
    let escalation_input = GovernanceEscalationInput {
        market_id: process.market_id.clone(),
        market_question: String::new(), // Would be fetched from market data
        dispute_reason,
        current_tallies: process.outcome_tallies.clone(),
        escalation_level,
        proposal_type: "market_resolution_dispute".to_string(),
    };

    // Bridge call to governance hApp to create resolution proposal
    // HDK 0.6: Handle ZomeCallResponse variants
    let response = call(
        CallTargetCell::OtherRole("governance".into()),
        ZomeName::new("proposals"),
        FunctionName::new("create_resolution_dispute_proposal"),
        None,
        escalation_input,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to escalate to governance: {:?}", e))))?;

    let proposal_id: EntryHash = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode proposal ID: {:?}", e))))?,
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            return Err(wasm_error!(WasmErrorInner::Guest("Unauthorized to escalate to governance".into())));
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

    // Emit signal for tracking
    emit_signal(serde_json::json!({
        "type": "resolution_escalated_to_governance",
        "market_id": process.market_id,
        "proposal_id": proposal_id,
    }))?;

    Ok(proposal_id)
}

/// Reputation update for an oracle based on prediction accuracy
#[derive(Serialize, Deserialize, Debug, Clone)]
struct OracleReputationUpdate {
    pub agent: AgentPubKey,
    pub domain: String,
    pub delta: f64,
    pub reason: String,
    pub market_id: EntryHash,
    pub staked_amount: f64,
}

/// Batch update input for MATL system
#[derive(Serialize, Deserialize, Debug, Clone)]
struct BatchReputationUpdate {
    pub updates: Vec<OracleReputationUpdate>,
    pub source: String,
}

fn update_oracle_reputations(process: &ResolutionProcess, winning_outcome: &str) -> ExternResult<()> {
    // Collect all votes to determine which oracles voted correctly
    let votes = collect_votes(process)?;

    let mut updates = Vec::new();

    for vote in &votes {
        let voted_correctly = vote.outcome == winning_outcome;
        let confidence_factor = vote.confidence;
        let stake_factor = vote.reputation_stake;

        // Calculate reputation delta:
        // - Correct vote with high confidence = large positive delta
        // - Incorrect vote with high stake = larger negative delta (accountability)
        let base_delta = if voted_correctly {
            0.05 * confidence_factor * vote.matl_weight
        } else {
            -0.08 * stake_factor * vote.matl_weight // Penalize incorrect votes with stake
        };

        updates.push(OracleReputationUpdate {
            agent: vote.oracle.clone(),
            domain: "epistemic_markets".to_string(),
            delta: base_delta,
            reason: if voted_correctly {
                format!("Correct oracle vote on market resolution (confidence: {:.2})", confidence_factor)
            } else {
                format!("Incorrect oracle vote on market resolution (stake: {:.2})", stake_factor)
            },
            market_id: process.market_id.clone(),
            staked_amount: vote.reputation_stake,
        });
    }

    // Bridge call to MATL system to update reputations
    let batch_update = BatchReputationUpdate {
        updates: updates.clone(),
        source: "epistemic_markets_resolution".to_string(),
    };

    // HDK 0.6: Handle ZomeCallResponse variants
    let response = call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::new("matl"),
        FunctionName::new("batch_update_reputation"),
        None,
        batch_update,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to update reputations: {:?}", e))))?;

    let _result: () = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().unwrap_or(()),
        _ => (),
    };

    // Emit signal for UI updates
    emit_signal(serde_json::json!({
        "type": "oracle_reputations_updated",
        "market_id": process.market_id,
        "winning_outcome": winning_outcome,
        "updates_count": updates.len(),
    }))?;

    Ok(())
}

// ============================================================================
// ENTRY AND LINK DEFINITIONS
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    ResolutionProcess(ResolutionProcess),
    OracleVote(OracleVote),
    PayoutRecord(PayoutRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    MarketToResolution,
    OracleToVote,
    ResolutionToVote,
    ResolutionToPayout,
}
