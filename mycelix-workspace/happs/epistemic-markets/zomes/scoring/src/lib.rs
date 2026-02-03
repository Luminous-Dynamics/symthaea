//! Scoring Zome - Calibration, Brier Scores, and Collective Intelligence Metrics
//!
//! Advanced mechanisms for:
//! - Individual calibration tracking and training
//! - Cascade detection (information herding)
//! - Disagreement mining (finding productive conflicts)
//! - Long-horizon commitment scoring
//! - Human-AI collaboration metrics

use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CALIBRATION SYSTEM
// ============================================================================

/// Complete calibration profile for a predictor
#[hdk_entry_helper]
#[derive(Clone)]
pub struct CalibrationProfile {
    pub agent: AgentPubKey,

    /// Overall Brier score (lower is better, 0 = perfect)
    pub brier_score: f64,

    /// Calibration curve: for predictions at X% confidence, what % were correct?
    pub calibration_buckets: Vec<CalibrationBucket>,

    /// Domain-specific scores
    pub domain_scores: HashMap<String, DomainScore>,

    /// Resolution: ability to distinguish true from false
    pub resolution_score: f64,

    /// Reliability: consistency of confidence levels
    pub reliability_score: f64,

    /// Total predictions tracked
    pub total_predictions: u64,
    pub resolved_predictions: u64,

    /// Performance over time
    pub performance_history: Vec<PerformanceSnapshot>,

    /// Last updated
    pub updated_at: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CalibrationBucket {
    /// Confidence range (e.g., 0.6 to 0.7)
    pub confidence_min: f64,
    pub confidence_max: f64,

    /// Number of predictions in this bucket
    pub count: u64,

    /// Number that were correct
    pub correct: u64,

    /// Actual accuracy rate
    pub actual_rate: f64,

    /// Calibration error for this bucket
    pub calibration_error: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DomainScore {
    pub domain: String,
    pub brier_score: f64,
    pub predictions: u64,
    pub calibration_error: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: u64,
    pub brier_score: f64,
    pub calibration_error: f64,
    pub predictions_count: u64,
}

impl CalibrationProfile {
    /// Calculate overall calibration error
    pub fn calibration_error(&self) -> f64 {
        if self.calibration_buckets.is_empty() {
            return 1.0;
        }

        let total_weight: u64 = self.calibration_buckets.iter().map(|b| b.count).sum();
        if total_weight == 0 {
            return 1.0;
        }

        self.calibration_buckets
            .iter()
            .map(|b| b.calibration_error * b.count as f64)
            .sum::<f64>()
            / total_weight as f64
    }

    /// Brier score decomposition
    pub fn brier_decomposition(&self) -> BrierDecomposition {
        // Brier = Reliability - Resolution + Uncertainty
        let base_rate = self.calculate_base_rate();
        let uncertainty = base_rate * (1.0 - base_rate);

        BrierDecomposition {
            reliability: self.reliability_score,
            resolution: self.resolution_score,
            uncertainty,
            total: self.reliability_score - self.resolution_score + uncertainty,
        }
    }

    fn calculate_base_rate(&self) -> f64 {
        let total: u64 = self.calibration_buckets.iter().map(|b| b.count).sum();
        let correct: u64 = self.calibration_buckets.iter().map(|b| b.correct).sum();
        if total == 0 {
            0.5
        } else {
            correct as f64 / total as f64
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BrierDecomposition {
    /// How well-calibrated (lower is better)
    pub reliability: f64,
    /// Ability to distinguish outcomes (higher is better)
    pub resolution: f64,
    /// Inherent uncertainty in the domain
    pub uncertainty: f64,
    /// Total Brier score
    pub total: f64,
}

// ============================================================================
// CASCADE DETECTION
// ============================================================================

/// Detect information cascades (herding behavior)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CascadeAnalysis {
    pub market_id: EntryHash,
    pub analysis_timestamp: u64,

    /// Overall cascade score (0 = independent, 1 = complete herding)
    pub cascade_score: f64,

    /// Detected cascade events
    pub cascade_events: Vec<CascadeEvent>,

    /// Independent information score
    pub independence_score: f64,

    /// Recommendations
    pub recommendations: Vec<CascadeRecommendation>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CascadeEvent {
    pub timestamp: u64,
    pub trigger_prediction: EntryHash,
    pub following_predictions: Vec<EntryHash>,

    /// How similar are the following predictions?
    pub similarity_score: f64,

    /// Time window of herding (seconds)
    pub window_seconds: u64,

    /// Was there new information in this window?
    pub new_information: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CascadeRecommendation {
    /// Delay showing predictions to prevent cascades
    DelayDisplay { suggested_delay_seconds: u64 },

    /// Require reasoning before showing others' predictions
    RequireReasoningFirst,

    /// Show confidence distribution without identifying predictors
    AnonymizeInitialDisplay,

    /// Incentivize contrarian predictions
    ContrarianBonus { bonus_percentage: f64 },

    /// Reduce weight of cascade-influenced predictions
    ReduceCascadeWeight { reduction_factor: f64 },
}

impl CascadeAnalysis {
    /// Analyze predictions for cascade behavior
    pub fn analyze(predictions: &[PredictionRecord], information_events: &[InformationEvent]) -> Self {
        let mut cascade_events = Vec::new();
        let mut total_cascade_score = 0.0;

        // Sort predictions by time
        let mut sorted_predictions = predictions.to_vec();
        sorted_predictions.sort_by_key(|p| p.timestamp);

        // Sliding window analysis
        let window_size = 10;
        for window in sorted_predictions.windows(window_size) {
            let first = &window[0];
            let rest = &window[1..];

            // Check similarity of predictions following the first
            let avg_similarity = rest
                .iter()
                .map(|p| prediction_similarity(first, p))
                .sum::<f64>()
                / rest.len() as f64;

            // Check if new information arrived
            let window_start = first.timestamp;
            let window_end = window.last().map(|p| p.timestamp).unwrap_or(window_start);
            let new_info = information_events
                .iter()
                .any(|e| e.timestamp >= window_start && e.timestamp <= window_end);

            // High similarity without new information = cascade
            if avg_similarity > 0.8 && !new_info {
                cascade_events.push(CascadeEvent {
                    timestamp: first.timestamp,
                    trigger_prediction: first.id.clone(),
                    following_predictions: rest.iter().map(|p| p.id.clone()).collect(),
                    similarity_score: avg_similarity,
                    window_seconds: (window_end - window_start) / 1_000_000,
                    new_information: new_info,
                });

                total_cascade_score += avg_similarity * (rest.len() as f64 / predictions.len() as f64);
            }
        }

        let cascade_score = total_cascade_score.min(1.0);
        let independence_score = 1.0 - cascade_score;

        let recommendations = Self::generate_recommendations(cascade_score, &cascade_events);

        CascadeAnalysis {
            market_id: EntryHash::from_raw_36(vec![0; 36]), // Placeholder
            analysis_timestamp: 0,
            cascade_score,
            cascade_events,
            independence_score,
            recommendations,
        }
    }

    fn generate_recommendations(score: f64, _events: &[CascadeEvent]) -> Vec<CascadeRecommendation> {
        let mut recs = Vec::new();

        if score > 0.7 {
            recs.push(CascadeRecommendation::RequireReasoningFirst);
            recs.push(CascadeRecommendation::ContrarianBonus {
                bonus_percentage: 20.0,
            });
        } else if score > 0.5 {
            recs.push(CascadeRecommendation::DelayDisplay {
                suggested_delay_seconds: 300,
            });
            recs.push(CascadeRecommendation::AnonymizeInitialDisplay);
        } else if score > 0.3 {
            recs.push(CascadeRecommendation::ReduceCascadeWeight {
                reduction_factor: 0.8,
            });
        }

        recs
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictionRecord {
    pub id: EntryHash,
    pub predictor: AgentPubKey,
    pub timestamp: u64,
    pub outcome: String,
    pub confidence: f64,
    pub reasoning_hash: Option<EntryHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InformationEvent {
    pub timestamp: u64,
    pub event_type: String,
    pub impact_estimate: f64,
}

fn prediction_similarity(a: &PredictionRecord, b: &PredictionRecord) -> f64 {
    let outcome_match = if a.outcome == b.outcome { 1.0 } else { 0.0 };
    let confidence_similarity = 1.0 - (a.confidence - b.confidence).abs();

    outcome_match * 0.7 + confidence_similarity * 0.3
}

// ============================================================================
// DISAGREEMENT MINING
// ============================================================================

/// Track and incentivize productive disagreement
#[hdk_entry_helper]
#[derive(Clone)]
pub struct DisagreementRecord {
    pub market_id: EntryHash,

    /// Agents on different sides
    pub thesis_holders: Vec<AgentPubKey>,
    pub antithesis_holders: Vec<AgentPubKey>,

    /// Confidence differential
    pub confidence_spread: f64,

    /// Identified cruxes (points that would change minds)
    pub cruxes: Vec<Crux>,

    /// Double cruxes found (shared pivot points)
    pub double_cruxes: Vec<DoubleCrux>,

    /// Has disagreement been productive?
    pub productivity_score: f64,

    /// Resolution contributions
    pub contributions: Vec<DisagreementContribution>,
}

#[hdk_entry_helper]
#[derive(Clone)]
pub struct Crux {
    pub id: EntryHash,
    pub proposer: AgentPubKey,
    pub description: String,

    /// What would you need to see to change your mind?
    pub operationalization: String,

    /// What probability shift if crux resolves one way vs other?
    pub if_true_shift: f64,
    pub if_false_shift: f64,

    /// How important is this crux to your position?
    pub importance: f64,

    /// Has this crux been resolved?
    pub status: CruxStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum CruxStatus {
    Proposed,
    Acknowledged,    // Other side agrees this is important
    BeingTested,     // Evidence gathering
    Resolved { outcome: bool },
    Rejected { reason: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DoubleCrux {
    /// A proposition BOTH sides would update on
    pub proposition: String,

    /// Person A's update
    pub a_agent: AgentPubKey,
    pub a_if_true: f64,
    pub a_if_false: f64,

    /// Person B's update
    pub b_agent: AgentPubKey,
    pub b_if_true: f64,
    pub b_if_false: f64,

    /// Status
    pub status: CruxStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisagreementContribution {
    pub agent: AgentPubKey,
    pub contribution_type: ContributionType,
    pub timestamp: u64,
    pub value: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ContributionType {
    /// Proposed a crux
    CruxProposal,
    /// Acknowledged opponent's valid point
    SteelMan,
    /// Found a double crux
    DoubleCruxDiscovery,
    /// Provided decisive evidence
    DecisiveEvidence,
    /// Updated beliefs appropriately
    AppropriateUpdate,
    /// Brought novel information
    NovelInformation,
}

impl DisagreementRecord {
    /// Calculate disagreement productivity score
    pub fn calculate_productivity(&self) -> f64 {
        let mut score = 0.0;

        // Reward crux identification
        score += self.cruxes.len() as f64 * 10.0;

        // Big reward for double cruxes
        score += self.double_cruxes.len() as f64 * 50.0;

        // Reward acknowledgments (steel-manning)
        let steel_mans = self
            .contributions
            .iter()
            .filter(|c| matches!(c.contribution_type, ContributionType::SteelMan))
            .count();
        score += steel_mans as f64 * 15.0;

        // Reward appropriate belief updates
        let updates = self
            .contributions
            .iter()
            .filter(|c| matches!(c.contribution_type, ContributionType::AppropriateUpdate))
            .count();
        score += updates as f64 * 20.0;

        // Penalize if disagreement persists without cruxes
        if self.cruxes.is_empty() && self.confidence_spread > 0.5 {
            score -= 30.0;
        }

        score.max(0.0)
    }

    /// Score individual contribution to disagreement resolution
    pub fn score_agent(&self, agent: &AgentPubKey) -> f64 {
        self.contributions
            .iter()
            .filter(|c| &c.agent == agent)
            .map(|c| c.value)
            .sum()
    }
}

// ============================================================================
// LONG-HORIZON SCORING
// ============================================================================

/// Track long-horizon prediction performance
#[hdk_entry_helper]
#[derive(Clone)]
pub struct LongHorizonTracker {
    pub predictor: AgentPubKey,

    /// Predictions by time horizon
    pub predictions_by_horizon: HashMap<HorizonCategory, Vec<TrackedPrediction>>,

    /// Accuracy by horizon (does accuracy decay with distance?)
    pub accuracy_by_horizon: HashMap<HorizonCategory, f64>,

    /// Commitment level tracking
    pub commitment_scores: Vec<CommitmentScore>,

    /// Update behavior (do they update appropriately over time?)
    pub update_quality: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq)]
pub enum HorizonCategory {
    Immediate,  // < 1 week
    Short,      // 1 week - 1 month
    Medium,     // 1-6 months
    Long,       // 6 months - 2 years
    Extended,   // 2-10 years
    Generational, // 10+ years
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrackedPrediction {
    pub prediction_id: EntryHash,
    pub made_at: u64,
    pub resolves_at: u64,
    pub horizon: HorizonCategory,
    pub initial_confidence: f64,

    /// Updates over time
    pub updates: Vec<PredictionUpdate>,

    /// Final outcome
    pub resolved: Option<bool>,
    pub brier_contribution: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictionUpdate {
    pub timestamp: u64,
    pub new_confidence: f64,
    pub reasoning: String,
    pub triggered_by: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommitmentScore {
    pub prediction_id: EntryHash,
    pub commitment_type: CommitmentType,
    pub held_until_resolution: bool,
    pub early_exit_penalty: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CommitmentType {
    Standard,
    Locked,
    Vesting,
    Legacy,
}

impl LongHorizonTracker {
    /// Calculate if predictor is better at short or long horizons
    pub fn horizon_comparative_advantage(&self) -> HashMap<HorizonCategory, f64> {
        let overall_accuracy = self.overall_accuracy();

        self.accuracy_by_horizon
            .iter()
            .map(|(horizon, accuracy)| {
                let advantage = accuracy - overall_accuracy;
                (horizon.clone(), advantage)
            })
            .collect()
    }

    fn overall_accuracy(&self) -> f64 {
        let total_count: usize = self
            .accuracy_by_horizon
            .values()
            .count();
        if total_count == 0 {
            return 0.5;
        }

        self.accuracy_by_horizon.values().sum::<f64>() / total_count as f64
    }

    /// Score update quality (do they update appropriately?)
    pub fn calculate_update_quality(&self) -> f64 {
        let mut quality_scores = Vec::new();

        for predictions in self.predictions_by_horizon.values() {
            for pred in predictions {
                if pred.updates.is_empty() {
                    continue;
                }

                // Good updates: move toward final outcome
                if let Some(resolved) = pred.resolved {
                    let target = if resolved { 1.0 } else { 0.0 };

                    let mut prev_conf = pred.initial_confidence;
                    for update in &pred.updates {
                        let prev_distance = (prev_conf - target).abs();
                        let new_distance = (update.new_confidence - target).abs();

                        // Good update moves closer to truth
                        let update_quality = if new_distance < prev_distance {
                            1.0
                        } else {
                            -0.5
                        };

                        quality_scores.push(update_quality);
                        prev_conf = update.new_confidence;
                    }
                }
            }
        }

        if quality_scores.is_empty() {
            0.5
        } else {
            quality_scores.iter().sum::<f64>() / quality_scores.len() as f64
        }
    }
}

// ============================================================================
// ZOME FUNCTIONS
// ============================================================================

#[hdk_extern]
pub fn record_prediction_outcome(input: RecordOutcomeInput) -> ExternResult<EntryHash> {
    let predictor = input.predictor;
    let mut profile = get_or_create_calibration_profile(predictor.clone())?;

    // Update Brier score
    let outcome_value = if input.was_correct { 1.0 } else { 0.0 };
    let squared_error = (input.confidence - outcome_value).powi(2);

    // Exponential moving average for Brier score
    profile.brier_score = profile.brier_score * 0.95 + squared_error * 0.05;

    // Update calibration bucket
    let bucket_idx = (input.confidence * 10.0).floor() as usize;
    let bucket_idx = bucket_idx.min(9);

    if let Some(bucket) = profile.calibration_buckets.get_mut(bucket_idx) {
        bucket.count += 1;
        if input.was_correct {
            bucket.correct += 1;
        }
        bucket.actual_rate = bucket.correct as f64 / bucket.count as f64;
        let bucket_midpoint = (bucket.confidence_min + bucket.confidence_max) / 2.0;
        bucket.calibration_error = (bucket.actual_rate - bucket_midpoint).abs();
    }

    // Update domain score
    if let Some(domain) = &input.domain {
        let domain_score = profile
            .domain_scores
            .entry(domain.clone())
            .or_insert(DomainScore {
                domain: domain.clone(),
                brier_score: 0.25,
                predictions: 0,
                calibration_error: 0.0,
            });

        domain_score.predictions += 1;
        domain_score.brier_score = domain_score.brier_score * 0.9 + squared_error * 0.1;
    }

    profile.resolved_predictions += 1;
    profile.updated_at = sys_time()?.as_micros() as u64;

    // Save profile
    let hash = update_calibration_profile(&profile)?;

    // Emit calibration update signal
    emit_signal(serde_json::json!({
        "type": "calibration_updated",
        "agent": predictor,
        "brier_score": profile.brier_score,
        "calibration_error": profile.calibration_error(),
    }))?;

    Ok(hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecordOutcomeInput {
    pub predictor: AgentPubKey,
    pub prediction_id: EntryHash,
    pub confidence: f64,
    pub was_correct: bool,
    pub domain: Option<String>,
}

#[hdk_extern]
pub fn get_calibration_profile(agent: AgentPubKey) -> ExternResult<Option<CalibrationProfile>> {
    let anchor = calibration_anchor(&agent)?;
    let links =
        get_links(LinkQuery::try_new(anchor, LinkTypes::AgentToProfile)?, GetStrategy::default())?;

    if let Some(link) = links.first() {
        let hash = link.target.clone().into_entry_hash().unwrap();
        if let Some(record) = get(hash, GetOptions::default())? {
            return Ok(record.entry().to_app_option().ok().flatten());
        }
    }

    Ok(None)
}

#[hdk_extern]
pub fn analyze_cascade(market_id: EntryHash) -> ExternResult<CascadeAnalysis> {
    // Fetch predictions for market via bridge call to predictions zome
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("predictions"),
        FunctionName::new("get_predictions_for_market"),
        None,
        market_id.clone(),
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to fetch predictions: {:?}", e))))?;

    // HDK 0.6: Handle ZomeCallResponse variants
    let predictions: Vec<PredictionRecord> = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().unwrap_or_else(|_| vec![]),
        ZomeCallResponse::Unauthorized(_, _, _, _) => vec![],
        ZomeCallResponse::NetworkError(_) => vec![],
        ZomeCallResponse::CountersigningSession(_) => vec![],
        ZomeCallResponse::AuthenticationFailed(_, _) => vec![],
    };

    // Fetch information events from knowledge hApp for context
    let info_response = call(
        CallTargetCell::OtherRole("knowledge".into()),
        ZomeName::new("knowledge_base"),
        FunctionName::new("get_information_events"),
        None,
        market_id.clone(),
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to fetch information events: {:?}", e))))?;

    // HDK 0.6: Handle ZomeCallResponse variants
    let information_events: Vec<InformationEvent> = match info_response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().unwrap_or_else(|_| vec![]),
        ZomeCallResponse::Unauthorized(_, _, _, _) => vec![],
        ZomeCallResponse::NetworkError(_) => vec![],
        ZomeCallResponse::CountersigningSession(_) => vec![],
        ZomeCallResponse::AuthenticationFailed(_, _) => vec![],
    };

    let mut analysis = CascadeAnalysis::analyze(&predictions, &information_events);
    analysis.market_id = market_id;
    analysis.analysis_timestamp = sys_time()?.as_micros() as u64;

    Ok(analysis)
}

#[hdk_extern]
pub fn propose_crux(input: ProposeCruxInput) -> ExternResult<EntryHash> {
    let _now = sys_time()?.as_micros() as u64;
    let proposer = agent_info()?.agent_initial_pubkey;

    let crux = Crux {
        id: EntryHash::from_raw_36(vec![0; 36]),
        proposer,
        description: input.description,
        operationalization: input.operationalization,
        if_true_shift: input.if_true_shift,
        if_false_shift: input.if_false_shift,
        importance: input.importance,
        status: CruxStatus::Proposed,
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let entry_hash = hash_entry(&crux)?;
    let _action_hash = create_entry(&EntryTypes::Crux(crux))?;

    // Link to market
    create_link(input.market_id.clone(), entry_hash.clone(), LinkTypes::MarketToCrux, ())?;

    emit_signal(serde_json::json!({
        "type": "crux_proposed",
        "market_id": input.market_id,
        "crux_id": entry_hash,
    }))?;

    Ok(entry_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProposeCruxInput {
    pub market_id: EntryHash,
    pub description: String,
    pub operationalization: String,
    pub if_true_shift: f64,
    pub if_false_shift: f64,
    pub importance: f64,
}

#[hdk_extern]
pub fn get_leaderboard(input: LeaderboardInput) -> ExternResult<Vec<LeaderboardEntry>> {
    let anchor = anchor_hash("all_profiles")?;
    let links =
        get_links(LinkQuery::try_new(anchor, LinkTypes::AnchorToProfile)?, GetStrategy::default())?;

    let mut entries = Vec::new();
    for link in links {
        if let Some(profile) =
            get_calibration_profile(AgentPubKey::from_raw_36(link.target.into_inner()))?
        {
            // Filter by domain if specified
            if let Some(ref domain) = input.domain {
                if !profile.domain_scores.contains_key(domain) {
                    continue;
                }
            }

            // Filter by minimum predictions
            if profile.resolved_predictions < input.min_predictions.unwrap_or(10) {
                continue;
            }

            entries.push(LeaderboardEntry {
                agent: profile.agent.clone(),
                brier_score: profile.brier_score,
                calibration_error: profile.calibration_error(),
                predictions_count: profile.resolved_predictions,
                rank: 0, // Will be set after sorting
            });
        }
    }

    // Sort by Brier score (lower is better)
    entries.sort_by(|a, b| {
        a.brier_score
            .partial_cmp(&b.brier_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Set ranks
    for (i, entry) in entries.iter_mut().enumerate() {
        entry.rank = i as u32 + 1;
    }

    // Return top N
    entries.truncate(input.limit.unwrap_or(100) as usize);

    Ok(entries)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderboardInput {
    pub domain: Option<String>,
    pub min_predictions: Option<u64>,
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderboardEntry {
    pub agent: AgentPubKey,
    pub brier_score: f64,
    pub calibration_error: f64,
    pub predictions_count: u64,
    pub rank: u32,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn get_or_create_calibration_profile(agent: AgentPubKey) -> ExternResult<CalibrationProfile> {
    if let Some(profile) = get_calibration_profile(agent.clone())? {
        return Ok(profile);
    }

    // Create new profile with default buckets
    let now = sys_time()?.as_micros() as u64;

    let buckets: Vec<CalibrationBucket> = (0..10)
        .map(|i| CalibrationBucket {
            confidence_min: i as f64 * 0.1,
            confidence_max: (i + 1) as f64 * 0.1,
            count: 0,
            correct: 0,
            actual_rate: 0.0,
            calibration_error: 0.0,
        })
        .collect();

    let profile = CalibrationProfile {
        agent: agent.clone(),
        brier_score: 0.25, // Default (random guessing on binary)
        calibration_buckets: buckets,
        domain_scores: HashMap::new(),
        resolution_score: 0.0,
        reliability_score: 0.0,
        total_predictions: 0,
        resolved_predictions: 0,
        performance_history: vec![],
        updated_at: now,
    };

    let hash = create_entry(&EntryTypes::CalibrationProfile(profile.clone()))?;

    // Link to agent
    let anchor = calibration_anchor(&agent)?;
    create_link(anchor, hash.clone(), LinkTypes::AgentToProfile, ())?;

    // Link to all profiles
    let all_anchor = anchor_hash("all_profiles")?;
    create_link(all_anchor, hash, LinkTypes::AnchorToProfile, ())?;

    Ok(profile)
}

fn update_calibration_profile(profile: &CalibrationProfile) -> ExternResult<EntryHash> {
    let anchor = calibration_anchor(&profile.agent)?;
    let links =
        get_links(LinkQuery::try_new(anchor, LinkTypes::AgentToProfile)?, GetStrategy::default())?;

    if let Some(link) = links.first() {
        let entry_hash = link.target.clone().into_entry_hash().unwrap();
        // HDK 0.6: update_entry takes ActionHash, need to get record first
        if let Some(record) = get(entry_hash.clone(), GetOptions::default())? {
            let action_hash = record.action_hashed().hash.clone();
            update_entry(action_hash, profile)?;
        }
    }

    Ok(EntryHash::from_raw_36(vec![0; 36]))
}

fn calibration_anchor(agent: &AgentPubKey) -> ExternResult<EntryHash> {
    anchor_hash(&format!("calibration:{}", agent))
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
    CalibrationProfile(CalibrationProfile),
    DisagreementRecord(DisagreementRecord),
    LongHorizonTracker(LongHorizonTracker),
    Crux(Crux),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToProfile,
    AnchorToProfile,
    MarketToCrux,
    MarketToDisagreement,
}
