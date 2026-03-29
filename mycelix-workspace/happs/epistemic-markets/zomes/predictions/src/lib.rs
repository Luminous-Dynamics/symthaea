// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictions Zome - Foundations for Long-Term Epistemic Infrastructure
//!
//! This zome implements prediction submission with mechanisms that support
//! the long-term vision of collective intelligence:
//!
//! - Belief dependencies (seeds of belief graphs)
//! - Reasoning traces (for wisdom extraction)
//! - Temporal commitments (long-horizon support)
//! - Epistemic lineage (intergenerational transfer)
//! - Wisdom seeds (lessons for the future)

use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CORE PREDICTION WITH REASONING TRACE
// ============================================================================

/// A prediction with full reasoning trace for wisdom extraction
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Prediction {
    pub id: EntryHash,
    pub market_id: EntryHash,
    pub predictor: AgentPubKey,

    /// The prediction itself
    pub outcome: String,
    pub confidence: f64,

    /// Multi-dimensional stake
    pub stake: MultiDimensionalStake,

    /// CRITICAL: Full reasoning trace (for wisdom extraction)
    pub reasoning: ReasoningTrace,

    /// Belief dependencies (seed of belief graph)
    pub depends_on: Vec<BeliefDependency>,

    /// Temporal commitment level
    pub temporal_commitment: TemporalCommitment,

    /// For the future: what would we tell them?
    pub wisdom_seed: Option<WisdomSeed>,

    /// Lineage: what previous predictions/wisdom informed this?
    pub epistemic_lineage: Vec<EntryHash>,

    /// Lifecycle
    pub created_at: u64,
    pub updated_at: u64,
    pub updates: Vec<PredictionUpdate>,

    /// Resolution
    pub resolved: Option<PredictionResolution>,
}

// ============================================================================
// REASONING TRACES (Foundation for Wisdom Extraction)
// ============================================================================

/// Full reasoning trace - the gift we give to future wisdom extraction
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReasoningTrace {
    /// Natural language summary
    pub summary: String,

    /// Structured reasoning steps
    pub steps: Vec<ReasoningStep>,

    /// Key assumptions (explicit!)
    pub assumptions: Vec<Assumption>,

    /// What would make me update? (cruxes)
    pub update_triggers: Vec<UpdateTrigger>,

    /// Known weaknesses in my reasoning
    pub acknowledged_weaknesses: Vec<String>,

    /// Alternative framings considered
    pub alternatives_considered: Vec<AlternativeFraming>,

    /// Information sources consulted
    pub sources: Vec<InformationSource>,

    /// Confidence decomposition
    pub confidence_breakdown: ConfidenceBreakdown,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReasoningStep {
    pub step_number: u32,
    pub claim: String,
    pub support: StepSupport,
    pub confidence_contribution: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StepSupport {
    /// Empirical evidence
    Evidence { sources: Vec<String>, strength: f64 },
    /// Logical inference
    Inference { from_steps: Vec<u32>, rule: String },
    /// Expert testimony
    Testimony { source: String, credibility: f64 },
    /// Prior belief
    Prior { basis: String },
    /// Assumption (explicit!)
    Assumption { id: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Assumption {
    pub id: String,
    pub statement: String,
    /// How confident am I in this assumption?
    pub confidence: f64,
    /// What would falsify this assumption?
    pub falsification_criteria: String,
    /// How much does my conclusion depend on this?
    pub importance: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateTrigger {
    pub description: String,
    /// What evidence would trigger update?
    pub evidence_type: String,
    /// How much would I update?
    pub update_magnitude: f64,
    /// Direction of update
    pub direction: UpdateDirection,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UpdateDirection {
    Increase,
    Decrease,
    Depends { on: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlternativeFraming {
    pub framing: String,
    pub why_rejected: String,
    pub confidence_if_correct: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InformationSource {
    pub source_type: SourceType,
    pub reference: String,
    pub credibility_assessment: f64,
    pub key_claims: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SourceType {
    AcademicPaper,
    JournalisticReport,
    ExpertOpinion,
    DataSet,
    PredictionMarket,
    PersonalExperience,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConfidenceBreakdown {
    /// Base rate contribution
    pub base_rate: f64,
    /// Evidence adjustment
    pub evidence_adjustment: f64,
    /// Model uncertainty
    pub model_uncertainty: f64,
    /// Known unknowns
    pub known_unknowns: f64,
    /// Allowance for unknown unknowns
    pub unknown_unknowns_allowance: f64,
}

// ============================================================================
// BELIEF DEPENDENCIES (Seeds of Belief Graphs)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BeliefDependency {
    /// What this prediction depends on
    pub depends_on: DependencyTarget,
    /// Type of dependency
    pub dependency_type: DependencyType,
    /// Strength of dependency (-1.0 to 1.0)
    pub strength: f64,
    /// Conditional probabilities
    pub conditional: ConditionalProbability,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DependencyTarget {
    /// Another prediction in the system
    Prediction(EntryHash),
    /// A claim in the knowledge graph
    KnowledgeClaim(EntryHash),
    /// An external fact
    ExternalFact { description: String },
    /// A future event
    FutureEvent { description: String, expected_by: u64 },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DependencyType {
    /// If dependency is true, this is more likely
    PositiveEvidence,
    /// If dependency is true, this is less likely
    NegativeEvidence,
    /// Dependency must be true for this to be possible
    Prerequisite,
    /// Dependency causes this
    Causal,
    /// Correlated but not causal
    Correlated,
    /// This is a more specific version of dependency
    SpecializationOf,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConditionalProbability {
    /// P(this | dependency true)
    pub if_true: f64,
    /// P(this | dependency false)
    pub if_false: f64,
    /// Current P(dependency)
    pub dependency_probability: f64,
}

// ============================================================================
// TEMPORAL COMMITMENTS (Long-Horizon Support)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TemporalCommitment {
    /// Standard: can exit anytime
    Standard,

    /// Locked: stake locked until resolution
    Locked {
        lock_until: u64,
        early_exit_penalty: f64,
    },

    /// Vesting: stake unlocks gradually
    Vesting {
        schedule: Vec<VestingMilestone>,
    },

    /// Legacy: stake transfers to successors
    Legacy {
        successor_policy: SuccessorPolicy,
        knowledge_package: Option<EntryHash>,
    },

    /// Generational: explicitly multi-generational prediction
    Generational {
        generation: u32,
        cohort_id: EntryHash,
        expected_generations: u32,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VestingMilestone {
    pub timestamp: u64,
    pub percentage_unlocked: f64,
    pub condition: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SuccessorPolicy {
    /// Return stake to commons
    ReturnToCommons,
    /// Designated successor
    Designated {
        successor: AgentPubKey,
        transfer_conditions: Vec<String>,
    },
    /// Auction to highest bidder
    Auction {
        minimum_matl: f64,
    },
    /// Community election
    Election {
        eligible_voters: VoterEligibility,
    },
    /// Automatic to mentee
    Mentorship {
        mentee: AgentPubKey,
        verification_required: Vec<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum VoterEligibility {
    AllMatl { minimum: f64 },
    DomainExperts { domain: String, minimum_matl: f64 },
    PreviousPredictors { market_id: EntryHash },
}

// ============================================================================
// WISDOM SEEDS (Gifts to the Future)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WisdomSeed {
    /// If this prediction resolves correctly, what lesson?
    pub if_correct: WisdomLesson,
    /// If this prediction resolves incorrectly, what lesson?
    pub if_incorrect: WisdomLesson,
    /// Regardless of outcome, what's worth knowing?
    pub meta_lesson: Option<String>,
    /// Message to future predictors in this domain
    pub letter_to_future: Option<String>,
    /// Activation: when should this wisdom become prominent?
    pub activation_conditions: Vec<ActivationCondition>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WisdomLesson {
    pub lesson: String,
    pub confidence: f64,
    pub applicability: LessonApplicability,
    pub caveats: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LessonApplicability {
    /// Domains this lesson applies to
    pub domains: Vec<String>,
    /// Time horizons this lesson applies to
    pub time_horizons: Vec<String>,
    /// Conditions under which this lesson applies
    pub conditions: Vec<String>,
    /// Known exceptions
    pub exceptions: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ActivationCondition {
    /// Activate after resolution
    AfterResolution,
    /// Activate at specific time
    AtTime(u64),
    /// Activate when similar prediction made
    SimilarPrediction { similarity_threshold: f64 },
    /// Activate when domain becomes active
    DomainActivity { domain: String, threshold: f64 },
    /// Activate on specific event
    OnEvent { event_type: String },
}

// ============================================================================
// PREDICTION UPDATES AND RESOLUTION
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictionUpdate {
    pub timestamp: u64,
    pub old_confidence: f64,
    pub new_confidence: f64,
    pub reason: UpdateReason,
    pub reasoning_delta: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UpdateReason {
    /// New evidence
    NewEvidence { description: String, source: String },
    /// Dependency resolved
    DependencyResolved { dependency: EntryHash, outcome: bool },
    /// Reasoning error corrected
    ReasoningCorrection { error: String, correction: String },
    /// Time passage (base rate update)
    TimePassage,
    /// External event
    ExternalEvent { event: String },
    /// Crux resolved
    CruxResolved { crux: String, outcome: bool },
    /// Peer feedback
    PeerFeedback { from: AgentPubKey, feedback: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictionResolution {
    pub resolved_at: u64,
    pub outcome: String,
    pub was_correct: bool,
    pub brier_contribution: f64,

    /// Post-resolution reflection
    pub reflection: Option<PostResolutionReflection>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PostResolutionReflection {
    /// What did I get right?
    pub correct_reasoning: Vec<String>,
    /// What did I get wrong?
    pub incorrect_reasoning: Vec<String>,
    /// What would I do differently?
    pub lessons_learned: Vec<String>,
    /// Advice for future similar predictions
    pub advice_for_future: String,
    /// Which assumptions held? Which failed?
    pub assumption_outcomes: HashMap<String, bool>,
}

// ============================================================================
// MULTI-DIMENSIONAL STAKES (From markets zome, repeated for completeness)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct MultiDimensionalStake {
    pub monetary: Option<MonetaryStake>,
    pub reputation: Option<ReputationStake>,
    pub social: Option<SocialStake>,
    pub commitment: Option<CommitmentStake>,
    pub time: Option<TimeStake>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MonetaryStake {
    pub amount: u64,
    pub currency: String,
    pub escrow_id: Option<EntryHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReputationStake {
    pub domains: Vec<String>,
    pub stake_percentage: f64,
    pub confidence_multiplier: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SocialStake {
    pub visibility: Visibility,
    pub identity_link: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Visibility {
    Private,
    Limited(Vec<AgentPubKey>),
    Community,
    Public,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommitmentStake {
    pub if_correct: Vec<Commitment>,
    pub if_wrong: Vec<Commitment>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Commitment {
    BeliefUpdate { topic: String },
    Investigation { hours: u32, topic: String },
    Donation { amount: u64, recipient: String },
    Mentorship { hours: u32, domain: String },
    Custom { description: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimeStake {
    pub research_hours: u32,
    pub evidence_submitted: Vec<EntryHash>,
}

// ============================================================================
// ZOME FUNCTIONS
// ============================================================================

#[hdk_extern]
pub fn submit_prediction(input: SubmitPredictionInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let predictor = agent_info()?.agent_initial_pubkey;

    // Validate reasoning trace has minimum required fields
    validate_reasoning_trace(&input.reasoning)?;

    let prediction = Prediction {
        id: EntryHash::from_raw_36(vec![0; 36]),
        market_id: input.market_id.clone(),
        predictor: predictor.clone(),
        outcome: input.outcome,
        confidence: input.confidence,
        stake: input.stake,
        reasoning: input.reasoning,
        depends_on: input.depends_on,
        temporal_commitment: input.temporal_commitment,
        wisdom_seed: input.wisdom_seed,
        epistemic_lineage: input.epistemic_lineage,
        created_at: now,
        updated_at: now,
        updates: vec![],
        resolved: None,
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let hash = hash_entry(&prediction)?;
    let _action_hash = create_entry(&EntryTypes::Prediction(prediction.clone()))?;

    // Link to market
    create_link(
        input.market_id.clone(),
        hash.clone(),
        LinkTypes::MarketToPrediction,
        (),
    )?;

    // Link to predictor
    create_link(
        predictor.clone(),
        hash.clone(),
        LinkTypes::PredictorToPrediction,
        (),
    )?;

    // Link to dependencies (for belief graph)
    for dep in &prediction.depends_on {
        if let DependencyTarget::Prediction(dep_hash) = &dep.depends_on {
            create_link(
                dep_hash.clone(),
                hash.clone(),
                LinkTypes::DependencyLink,
                (),
            )?;
        }
    }

    // Link to epistemic lineage
    for ancestor in &prediction.epistemic_lineage {
        create_link(
            ancestor.clone(),
            hash.clone(),
            LinkTypes::LineageLink,
            (),
        )?;
    }

    // Emit signal for real-time updates
    emit_signal(serde_json::json!({
        "type": "prediction_submitted",
        "market_id": input.market_id,
        "prediction_id": hash,
        "outcome": prediction.outcome,
        "confidence": prediction.confidence,
        "has_reasoning": true,
        "has_wisdom_seed": prediction.wisdom_seed.is_some(),
    }))?;

    Ok(hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubmitPredictionInput {
    pub market_id: EntryHash,
    pub outcome: String,
    pub confidence: f64,
    pub stake: MultiDimensionalStake,
    pub reasoning: ReasoningTrace,
    pub depends_on: Vec<BeliefDependency>,
    pub temporal_commitment: TemporalCommitment,
    pub wisdom_seed: Option<WisdomSeed>,
    pub epistemic_lineage: Vec<EntryHash>,
}

fn validate_reasoning_trace(trace: &ReasoningTrace) -> ExternResult<()> {
    // Require minimum reasoning
    if trace.summary.len() < 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reasoning summary must be at least 50 characters".into()
        )));
    }

    if trace.steps.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one reasoning step required".into()
        )));
    }

    if trace.assumptions.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one assumption must be made explicit".into()
        )));
    }

    Ok(())
}

#[hdk_extern]
pub fn update_prediction(input: UpdatePredictionInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let caller = agent_info()?.agent_initial_pubkey;

    // HDK 0.6: Get record to extract action hash for update
    let record = get(input.prediction_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;

    let mut prediction: Prediction = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction entry not found".into())))?;

    // Only predictor can update
    if caller != prediction.predictor {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only predictor can update".into()
        )));
    }

    // Check temporal commitment allows update
    match &prediction.temporal_commitment {
        TemporalCommitment::Locked { lock_until, .. } => {
            if now < *lock_until {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Prediction is locked".into()
                )));
            }
        }
        _ => {}
    }

    // Record update
    let update = PredictionUpdate {
        timestamp: now,
        old_confidence: prediction.confidence,
        new_confidence: input.new_confidence,
        reason: input.reason,
        reasoning_delta: input.reasoning_delta,
    };

    prediction.updates.push(update);
    prediction.confidence = input.new_confidence;
    prediction.updated_at = now;

    // HDK 0.6: update_entry takes ActionHash
    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &prediction)?;

    emit_signal(serde_json::json!({
        "type": "prediction_updated",
        "prediction_id": input.prediction_id,
        "old_confidence": prediction.updates.last().unwrap().old_confidence,
        "new_confidence": input.new_confidence,
    }))?;

    Ok(input.prediction_id)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdatePredictionInput {
    pub prediction_id: EntryHash,
    pub new_confidence: f64,
    pub reason: UpdateReason,
    pub reasoning_delta: String,
}

#[hdk_extern]
pub fn add_post_resolution_reflection(input: AddReflectionInput) -> ExternResult<EntryHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // HDK 0.6: Get record to extract action hash for update
    let record = get(input.prediction_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;

    let mut prediction: Prediction = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction entry not found".into())))?;

    if caller != prediction.predictor {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only predictor can add reflection".into()
        )));
    }

    if let Some(ref mut resolution) = prediction.resolved {
        resolution.reflection = Some(input.reflection);

        // HDK 0.6: update_entry takes ActionHash
        let action_hash = record.action_hashed().hash.clone();
        update_entry(action_hash, &prediction)?;

        // Trigger wisdom extraction
        emit_signal(serde_json::json!({
            "type": "reflection_added",
            "prediction_id": input.prediction_id,
            "wisdom_seed": prediction.wisdom_seed,
        }))?;

        Ok(input.prediction_id)
    } else {
        Err(wasm_error!(WasmErrorInner::Guest(
            "Prediction not yet resolved".into()
        )))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddReflectionInput {
    pub prediction_id: EntryHash,
    pub reflection: PostResolutionReflection,
}

#[hdk_extern]
pub fn get_prediction(hash: EntryHash) -> ExternResult<Option<Prediction>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let prediction: Prediction = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;
            Ok(Some(prediction))
        }
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn get_predictions_for_market(market_id: EntryHash) -> ExternResult<Vec<Prediction>> {
    let links = get_links(
        LinkQuery::try_new(market_id, LinkTypes::MarketToPrediction)?,
        GetStrategy::default(),
    )?;

    let mut predictions = Vec::new();
    for link in links {
        if let Some(prediction) = get_prediction(link.target.into_entry_hash().unwrap())? {
            predictions.push(prediction);
        }
    }

    predictions.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(predictions)
}

#[hdk_extern]
pub fn get_belief_dependents(prediction_id: EntryHash) -> ExternResult<Vec<Prediction>> {
    let links = get_links(
        LinkQuery::try_new(prediction_id, LinkTypes::DependencyLink)?,
        GetStrategy::default(),
    )?;

    let mut dependents = Vec::new();
    for link in links {
        if let Some(prediction) = get_prediction(link.target.into_entry_hash().unwrap())? {
            dependents.push(prediction);
        }
    }

    Ok(dependents)
}

#[hdk_extern]
pub fn get_epistemic_lineage(prediction_id: EntryHash) -> ExternResult<Vec<Prediction>> {
    let prediction = get_prediction(prediction_id)?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;

    let mut lineage = Vec::new();
    for ancestor_id in &prediction.epistemic_lineage {
        if let Some(ancestor) = get_prediction(ancestor_id.clone())? {
            lineage.push(ancestor);
        }
    }

    Ok(lineage)
}

#[hdk_extern]
pub fn get_wisdom_seeds_for_domain(domain: String) -> ExternResult<Vec<WisdomSeedRecord>> {
    // Get all predictions with wisdom seeds in this domain
    let anchor = anchor_hash(&format!("wisdom_domain:{}", domain))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::DomainToWisdom)?,
        GetStrategy::default(),
    )?;

    let mut seeds = Vec::new();
    for link in links {
        if let Some(prediction) = get_prediction(link.target.into_entry_hash().unwrap())? {
            if let Some(ref wisdom_seed) = prediction.wisdom_seed {
                seeds.push(WisdomSeedRecord {
                    prediction_id: prediction.id,
                    predictor: prediction.predictor,
                    market_question: prediction.market_id.to_string(), // Would fetch actual question
                    wisdom_seed: wisdom_seed.clone(),
                    resolved: prediction.resolved.is_some(),
                    was_correct: prediction.resolved.as_ref().map(|r| r.was_correct),
                });
            }
        }
    }

    Ok(seeds)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WisdomSeedRecord {
    pub prediction_id: EntryHash,
    pub predictor: AgentPubKey,
    pub market_question: String,
    pub wisdom_seed: WisdomSeed,
    pub resolved: bool,
    pub was_correct: Option<bool>,
}

// ============================================================================
// VESTING ENFORCEMENT SYSTEM
// ============================================================================

/// Status of vesting schedule
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VestingStatus {
    /// Prediction this vesting applies to
    pub prediction_id: EntryHash,
    /// Original stake amount (monetary)
    pub total_stake: u64,
    /// Amount currently unlocked
    pub unlocked_amount: u64,
    /// Amount still locked
    pub locked_amount: u64,
    /// Percentage unlocked (0.0-1.0)
    pub unlock_percentage: f64,
    /// Next milestone (if any)
    pub next_milestone: Option<VestingMilestone>,
    /// Completed milestones
    pub completed_milestones: Vec<CompletedMilestone>,
    /// Early withdrawal penalty if withdrawn now
    pub early_penalty_amount: u64,
    /// Penalty percentage
    pub early_penalty_rate: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompletedMilestone {
    pub milestone: VestingMilestone,
    pub completed_at: u64,
    pub amount_unlocked: u64,
}

/// Input for checking vesting status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CheckVestingInput {
    pub prediction_id: EntryHash,
}

/// Input for early withdrawal
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EarlyWithdrawInput {
    pub prediction_id: EntryHash,
    /// Whether to accept the penalty and proceed
    pub accept_penalty: bool,
}

/// Result of withdrawal
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WithdrawalResult {
    pub prediction_id: EntryHash,
    pub total_stake: u64,
    pub amount_withdrawn: u64,
    pub penalty_applied: u64,
    pub remaining_locked: u64,
}

#[hdk_extern]
pub fn check_vesting_status(input: CheckVestingInput) -> ExternResult<VestingStatus> {
    let now = sys_time()?.as_micros() as u64;

    let prediction = get_prediction(input.prediction_id.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;

    // Get total stake amount
    let total_stake = prediction.stake.monetary
        .as_ref()
        .map(|m| m.amount)
        .unwrap_or(0);

    match &prediction.temporal_commitment {
        TemporalCommitment::Vesting { schedule } => {
            calculate_vesting_status(&input.prediction_id, total_stake, schedule, now)
        }
        TemporalCommitment::Locked { lock_until, early_exit_penalty } => {
            // For locked commitments, either fully locked or fully unlocked
            if now >= *lock_until {
                Ok(VestingStatus {
                    prediction_id: input.prediction_id,
                    total_stake,
                    unlocked_amount: total_stake,
                    locked_amount: 0,
                    unlock_percentage: 1.0,
                    next_milestone: None,
                    completed_milestones: vec![],
                    early_penalty_amount: 0,
                    early_penalty_rate: 0.0,
                })
            } else {
                let penalty = (total_stake as f64 * early_exit_penalty) as u64;
                Ok(VestingStatus {
                    prediction_id: input.prediction_id,
                    total_stake,
                    unlocked_amount: 0,
                    locked_amount: total_stake,
                    unlock_percentage: 0.0,
                    next_milestone: Some(VestingMilestone {
                        timestamp: *lock_until,
                        percentage_unlocked: 1.0,
                        condition: None,
                    }),
                    completed_milestones: vec![],
                    early_penalty_amount: penalty,
                    early_penalty_rate: *early_exit_penalty,
                })
            }
        }
        TemporalCommitment::Standard => {
            // Standard commitments have no vesting
            Ok(VestingStatus {
                prediction_id: input.prediction_id,
                total_stake,
                unlocked_amount: total_stake,
                locked_amount: 0,
                unlock_percentage: 1.0,
                next_milestone: None,
                completed_milestones: vec![],
                early_penalty_amount: 0,
                early_penalty_rate: 0.0,
            })
        }
        TemporalCommitment::Legacy { .. } | TemporalCommitment::Generational { .. } => {
            // Legacy and generational are special cases
            Ok(VestingStatus {
                prediction_id: input.prediction_id,
                total_stake,
                unlocked_amount: 0,
                locked_amount: total_stake,
                unlock_percentage: 0.0,
                next_milestone: None,
                completed_milestones: vec![],
                early_penalty_amount: total_stake, // Full stake at risk
                early_penalty_rate: 1.0,
            })
        }
    }
}

fn calculate_vesting_status(
    prediction_id: &EntryHash,
    total_stake: u64,
    schedule: &[VestingMilestone],
    now: u64,
) -> ExternResult<VestingStatus> {
    if schedule.is_empty() {
        return Ok(VestingStatus {
            prediction_id: prediction_id.clone(),
            total_stake,
            unlocked_amount: total_stake,
            locked_amount: 0,
            unlock_percentage: 1.0,
            next_milestone: None,
            completed_milestones: vec![],
            early_penalty_amount: 0,
            early_penalty_rate: 0.0,
        });
    }

    let mut completed = Vec::new();
    let mut current_unlock_pct = 0.0;
    let mut next_milestone: Option<VestingMilestone> = None;

    // Sort milestones by timestamp
    let mut sorted_schedule = schedule.to_vec();
    sorted_schedule.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    for milestone in &sorted_schedule {
        if now >= milestone.timestamp {
            // Milestone completed
            let amount_unlocked = (total_stake as f64 *
                (milestone.percentage_unlocked - current_unlock_pct)) as u64;

            completed.push(CompletedMilestone {
                milestone: milestone.clone(),
                completed_at: milestone.timestamp,
                amount_unlocked,
            });

            current_unlock_pct = milestone.percentage_unlocked;
        } else if next_milestone.is_none() {
            // This is the next upcoming milestone
            next_milestone = Some(milestone.clone());
        }
    }

    let unlocked_amount = (total_stake as f64 * current_unlock_pct) as u64;
    let locked_amount = total_stake - unlocked_amount;

    // Calculate early withdrawal penalty
    // Penalty is based on how much is still locked and time remaining
    let early_penalty_rate = if locked_amount > 0 {
        // Base penalty: 20% of remaining locked amount
        // Plus time-based: decreases as we get closer to next milestone
        let base_penalty = 0.20;
        let time_factor = if let Some(ref next) = next_milestone {
            let time_remaining = next.timestamp.saturating_sub(now);
            let total_period = next.timestamp.saturating_sub(
                completed.last().map(|c| c.completed_at).unwrap_or(0)
            );
            if total_period > 0 {
                (time_remaining as f64 / total_period as f64).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
        base_penalty + (0.10 * time_factor) // Up to 30% total
    } else {
        0.0
    };

    let early_penalty_amount = (locked_amount as f64 * early_penalty_rate) as u64;

    Ok(VestingStatus {
        prediction_id: prediction_id.clone(),
        total_stake,
        unlocked_amount,
        locked_amount,
        unlock_percentage: current_unlock_pct,
        next_milestone,
        completed_milestones: completed,
        early_penalty_amount,
        early_penalty_rate,
    })
}

#[hdk_extern]
pub fn withdraw_vested(input: CheckVestingInput) -> ExternResult<WithdrawalResult> {
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?.as_micros() as u64;

    let prediction = get_prediction(input.prediction_id.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;

    // Verify caller is the predictor
    if caller != prediction.predictor {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only predictor can withdraw".into()
        )));
    }

    // Get vesting status
    let status = check_vesting_status(CheckVestingInput {
        prediction_id: input.prediction_id.clone(),
    })?;

    // Only withdraw unlocked amount (no penalty)
    if status.unlocked_amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No vested amount available for withdrawal".into()
        )));
    }

    // Execute withdrawal via Finance hApp
    execute_vested_withdrawal(&caller, status.unlocked_amount, &input.prediction_id)?;

    // Emit signal
    emit_signal(serde_json::json!({
        "type": "vested_withdrawal",
        "prediction_id": input.prediction_id,
        "amount": status.unlocked_amount,
        "remaining_locked": status.locked_amount,
    }))?;

    Ok(WithdrawalResult {
        prediction_id: input.prediction_id,
        total_stake: status.total_stake,
        amount_withdrawn: status.unlocked_amount,
        penalty_applied: 0,
        remaining_locked: status.locked_amount,
    })
}

#[hdk_extern]
pub fn early_withdraw(input: EarlyWithdrawInput) -> ExternResult<WithdrawalResult> {
    let caller = agent_info()?.agent_initial_pubkey;

    let prediction = get_prediction(input.prediction_id.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;

    // Verify caller is the predictor
    if caller != prediction.predictor {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only predictor can withdraw".into()
        )));
    }

    // Get vesting status
    let status = check_vesting_status(CheckVestingInput {
        prediction_id: input.prediction_id.clone(),
    })?;

    if status.locked_amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No locked amount - use withdraw_vested instead".into()
        )));
    }

    if !input.accept_penalty {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Early withdrawal requires accepting penalty of {} ({:.1}%)",
            status.early_penalty_amount,
            status.early_penalty_rate * 100.0
        ))));
    }

    // Calculate withdrawal after penalty
    let total_available = status.unlocked_amount + status.locked_amount;
    let penalty = status.early_penalty_amount;
    let withdrawal_amount = total_available.saturating_sub(penalty);

    // Execute withdrawal and penalty via Finance hApp
    execute_early_withdrawal(
        &caller,
        withdrawal_amount,
        penalty,
        &input.prediction_id,
    )?;

    // Update reputation (early withdrawal impacts reputation)
    apply_early_withdrawal_reputation_penalty(&caller, status.early_penalty_rate)?;

    // Emit signal
    emit_signal(serde_json::json!({
        "type": "early_withdrawal",
        "prediction_id": input.prediction_id,
        "amount_withdrawn": withdrawal_amount,
        "penalty_applied": penalty,
        "penalty_rate": status.early_penalty_rate,
    }))?;

    Ok(WithdrawalResult {
        prediction_id: input.prediction_id,
        total_stake: status.total_stake,
        amount_withdrawn: withdrawal_amount,
        penalty_applied: penalty,
        remaining_locked: 0,
    })
}

/// Get all predictions with active vesting for an agent
#[hdk_extern]
pub fn get_vesting_predictions(agent: AgentPubKey) -> ExternResult<Vec<VestingStatus>> {
    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::PredictorToPrediction)?,
        GetStrategy::default(),
    )?;

    let mut vesting_statuses = Vec::new();

    for link in links {
        let prediction_id = link.target.into_entry_hash().unwrap();
        if let Some(prediction) = get_prediction(prediction_id.clone())? {
            // Only include predictions with vesting
            match &prediction.temporal_commitment {
                TemporalCommitment::Vesting { .. } | TemporalCommitment::Locked { .. } => {
                    let status = check_vesting_status(CheckVestingInput { prediction_id })?;
                    if status.locked_amount > 0 {
                        vesting_statuses.push(status);
                    }
                }
                _ => {}
            }
        }
    }

    // Sort by next milestone timestamp
    vesting_statuses.sort_by(|a, b| {
        let a_time = a.next_milestone.as_ref().map(|m| m.timestamp).unwrap_or(u64::MAX);
        let b_time = b.next_milestone.as_ref().map(|m| m.timestamp).unwrap_or(u64::MAX);
        a_time.cmp(&b_time)
    });

    Ok(vesting_statuses)
}

// ============================================================================
// WISDOM ACTIVATION ENGINE
// ============================================================================

/// Status of a wisdom seed activation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WisdomActivation {
    /// The prediction containing this wisdom
    pub prediction_id: EntryHash,
    /// The wisdom seed that was activated
    pub wisdom_seed: WisdomSeed,
    /// Which condition triggered activation
    pub activated_by: ActivationCondition,
    /// When the activation occurred
    pub activated_at: u64,
    /// Relevance score (0.0-1.0) for the triggering context
    pub relevance_score: f64,
    /// Whether the original prediction resolved correctly
    pub was_prediction_correct: Option<bool>,
}

/// Record of an activated wisdom seed (persisted)
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ActivatedWisdom {
    pub prediction_id: EntryHash,
    pub wisdom_seed: WisdomSeed,
    pub activation: WisdomActivation,
    pub created_at: u64,
    /// Number of times this wisdom has been surfaced
    pub surface_count: u64,
    /// Average usefulness rating (from users who acted on it)
    pub usefulness_rating: f64,
    pub rating_count: u64,
}

/// Input for checking wisdom activations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CheckActivationsInput {
    /// Domain to check (optional, checks all if None)
    pub domain: Option<String>,
    /// Specific event type to check
    pub event_type: Option<String>,
    /// New prediction being made (for similarity check)
    pub new_prediction_context: Option<PredictionContext>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PredictionContext {
    pub market_id: EntryHash,
    pub question: String,
    pub domain: String,
    pub outcomes: Vec<String>,
}

/// Input for surfacing relevant wisdom
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SurfaceWisdomInput {
    pub domain: String,
    pub context: Option<PredictionContext>,
    pub limit: Option<usize>,
}

/// Surfaced wisdom with relevance info
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SurfacedWisdom {
    pub wisdom: WisdomSeed,
    pub source_prediction_id: EntryHash,
    pub source_predictor: AgentPubKey,
    pub was_correct: Option<bool>,
    pub relevance_score: f64,
    pub relevance_reasons: Vec<String>,
    pub usefulness_rating: f64,
}

#[hdk_extern]
pub fn check_wisdom_activations(input: CheckActivationsInput) -> ExternResult<Vec<WisdomActivation>> {
    let now = sys_time()?.as_micros() as u64;
    let mut activations = Vec::new();

    // Get all predictions with wisdom seeds
    let wisdom_anchor = anchor_hash("all_wisdom_seeds")?;
    let links = get_links(
        LinkQuery::try_new(wisdom_anchor, LinkTypes::DomainToWisdom)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let prediction_id = link.target.into_entry_hash().unwrap();
        if let Some(prediction) = get_prediction(prediction_id.clone())? {
            if let Some(ref wisdom_seed) = prediction.wisdom_seed {
                // Check if wisdom matches domain filter
                if let Some(ref domain) = input.domain {
                    let matches_domain = wisdom_seed.if_correct.applicability.domains.contains(domain)
                        || wisdom_seed.if_incorrect.applicability.domains.contains(domain);
                    if !matches_domain {
                        continue;
                    }
                }

                // Check each activation condition
                for condition in &wisdom_seed.activation_conditions {
                    if let Some(activation) = check_condition_met(
                        condition,
                        &prediction,
                        wisdom_seed,
                        now,
                        &input,
                    )? {
                        activations.push(activation);
                    }
                }
            }
        }
    }

    // Sort by relevance
    activations.sort_by(|a, b| {
        b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(activations)
}

fn check_condition_met(
    condition: &ActivationCondition,
    prediction: &Prediction,
    wisdom_seed: &WisdomSeed,
    now: u64,
    input: &CheckActivationsInput,
) -> ExternResult<Option<WisdomActivation>> {
    let relevance = match condition {
        ActivationCondition::AfterResolution => {
            // Activate if prediction has been resolved
            if prediction.resolved.is_some() {
                Some(1.0) // Full relevance
            } else {
                None
            }
        }

        ActivationCondition::AtTime(timestamp) => {
            // Activate if we've passed the specified time
            if now >= *timestamp {
                Some(0.9) // High relevance
            } else {
                None
            }
        }

        ActivationCondition::SimilarPrediction { similarity_threshold } => {
            // Check if new prediction is similar
            if let Some(ref context) = input.new_prediction_context {
                let similarity = calculate_prediction_similarity(prediction, context);
                if similarity >= *similarity_threshold {
                    Some(similarity)
                } else {
                    None
                }
            } else {
                None
            }
        }

        ActivationCondition::DomainActivity { domain, threshold } => {
            // Check domain activity level
            if let Some(ref check_domain) = input.domain {
                if check_domain == domain {
                    // Would check actual domain activity here
                    // For now, assume any domain query indicates activity
                    Some(*threshold)
                } else {
                    None
                }
            } else {
                None
            }
        }

        ActivationCondition::OnEvent { event_type } => {
            // Check if matching event occurred
            if let Some(ref check_event) = input.event_type {
                if check_event == event_type {
                    Some(0.95)
                } else {
                    None
                }
            } else {
                None
            }
        }
    };

    if let Some(relevance_score) = relevance {
        Ok(Some(WisdomActivation {
            prediction_id: prediction.id.clone(),
            wisdom_seed: wisdom_seed.clone(),
            activated_by: condition.clone(),
            activated_at: now,
            relevance_score,
            was_prediction_correct: prediction.resolved.as_ref().map(|r| r.was_correct),
        }))
    } else {
        Ok(None)
    }
}

fn calculate_prediction_similarity(prediction: &Prediction, context: &PredictionContext) -> f64 {
    let mut score = 0.0;
    let mut factors = 0;

    // Domain match
    if prediction.reasoning.sources.first().is_some() {
        // Simplified: just check if domains might overlap
        factors += 1;
        score += 0.5; // Base similarity for being in system
    }

    // Outcome similarity (check if similar outcomes exist)
    let pred_outcome_lower = prediction.outcome.to_lowercase();
    for outcome in &context.outcomes {
        let outcome_lower = outcome.to_lowercase();
        if pred_outcome_lower.contains(&outcome_lower) ||
           outcome_lower.contains(&pred_outcome_lower) {
            score += 0.3;
            break;
        }
    }
    factors += 1;

    // Question keyword overlap (simplified)
    let pred_summary_lower = prediction.reasoning.summary.to_lowercase();
    let pred_words: std::collections::HashSet<&str> = pred_summary_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    let context_question_lower = context.question.to_lowercase();
    let context_words: std::collections::HashSet<&str> = context_question_lower
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    let overlap = pred_words.intersection(&context_words).count();
    if overlap > 0 {
        score += (overlap as f64 / pred_words.len().max(1) as f64).min(0.5);
    }
    factors += 1;

    if factors > 0 {
        (score / factors as f64).min(1.0)
    } else {
        0.0
    }
}

#[hdk_extern]
pub fn surface_relevant_wisdom(input: SurfaceWisdomInput) -> ExternResult<Vec<SurfacedWisdom>> {
    let limit = input.limit.unwrap_or(10);

    // First, check for new activations
    let activations = check_wisdom_activations(CheckActivationsInput {
        domain: Some(input.domain.clone()),
        event_type: None,
        new_prediction_context: input.context.clone(),
    })?;

    // Convert activations to surfaced wisdom
    let mut surfaced: Vec<SurfacedWisdom> = Vec::new();

    for activation in activations.into_iter().take(limit) {
        if let Some(prediction) = get_prediction(activation.prediction_id.clone())? {
            let mut relevance_reasons = Vec::new();

            // Add reasons based on what triggered activation
            match &activation.activated_by {
                ActivationCondition::AfterResolution => {
                    relevance_reasons.push("Prediction has been resolved".to_string());
                }
                ActivationCondition::SimilarPrediction { .. } => {
                    relevance_reasons.push("Similar to your prediction context".to_string());
                }
                ActivationCondition::DomainActivity { domain, .. } => {
                    relevance_reasons.push(format!("Relevant to active domain: {}", domain));
                }
                ActivationCondition::AtTime(_) => {
                    relevance_reasons.push("Scheduled wisdom surfacing".to_string());
                }
                ActivationCondition::OnEvent { event_type } => {
                    relevance_reasons.push(format!("Triggered by event: {}", event_type));
                }
            }

            // Add relevance based on prediction correctness
            if let Some(was_correct) = activation.was_prediction_correct {
                if was_correct {
                    relevance_reasons.push("Based on a correct prediction".to_string());
                } else {
                    relevance_reasons.push("Lesson from an incorrect prediction".to_string());
                }
            }

            surfaced.push(SurfacedWisdom {
                wisdom: activation.wisdom_seed,
                source_prediction_id: activation.prediction_id,
                source_predictor: prediction.predictor,
                was_correct: activation.was_prediction_correct,
                relevance_score: activation.relevance_score,
                relevance_reasons,
                usefulness_rating: 0.0, // Would fetch from ActivatedWisdom if it exists
            });
        }
    }

    // Sort by relevance score
    surfaced.sort_by(|a, b| {
        b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(surfaced)
}

#[hdk_extern]
pub fn activate_wisdom_seed(prediction_id: EntryHash) -> ExternResult<ActivatedWisdom> {
    let now = sys_time()?.as_micros() as u64;

    let prediction = get_prediction(prediction_id.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction not found".into())))?;

    let wisdom_seed = prediction.wisdom_seed.clone()
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Prediction has no wisdom seed".into())))?;

    // Determine which wisdom to use based on resolution
    let activation_condition = if prediction.resolved.is_some() {
        ActivationCondition::AfterResolution
    } else {
        ActivationCondition::AtTime(now)
    };

    let activation = WisdomActivation {
        prediction_id: prediction_id.clone(),
        wisdom_seed: wisdom_seed.clone(),
        activated_by: activation_condition,
        activated_at: now,
        relevance_score: 1.0,
        was_prediction_correct: prediction.resolved.as_ref().map(|r| r.was_correct),
    };

    let activated = ActivatedWisdom {
        prediction_id: prediction_id.clone(),
        wisdom_seed,
        activation,
        created_at: now,
        surface_count: 0,
        usefulness_rating: 0.0,
        rating_count: 0,
    };

    // Store activated wisdom
    let hash = hash_entry(&activated)?;
    let _action_hash = create_entry(&EntryTypes::ActivatedWisdom(activated.clone()))?;

    // Link to domain anchors for discovery
    for domain in &activated.wisdom_seed.if_correct.applicability.domains {
        let domain_anchor = anchor_hash(&format!("wisdom_domain:{}", domain))?;
        create_link(domain_anchor, hash.clone(), LinkTypes::DomainToWisdom, ())?;
    }

    // Emit signal
    emit_signal(serde_json::json!({
        "type": "wisdom_activated",
        "prediction_id": prediction_id,
        "activated_wisdom_id": hash,
        "was_correct": activated.activation.was_prediction_correct,
    }))?;

    Ok(activated)
}

#[hdk_extern]
pub fn rate_wisdom_usefulness(input: RateWisdomInput) -> ExternResult<()> {
    let record = get(input.activated_wisdom_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Activated wisdom not found".into())))?;

    let mut activated: ActivatedWisdom = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Wisdom entry not found".into())))?;

    // Update running average
    let old_total = activated.usefulness_rating * activated.rating_count as f64;
    activated.rating_count += 1;
    activated.usefulness_rating = (old_total + input.rating) / activated.rating_count as f64;

    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &activated)?;

    Ok(())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RateWisdomInput {
    pub activated_wisdom_id: EntryHash,
    pub rating: f64, // 0.0 to 1.0
}

/// Get top-ranked wisdom for a domain
#[hdk_extern]
pub fn get_top_wisdom(input: GetTopWisdomInput) -> ExternResult<Vec<RankedWisdom>> {
    let domain_anchor = anchor_hash(&format!("wisdom_domain:{}", input.domain))?;
    let links = get_links(
        LinkQuery::try_new(domain_anchor, LinkTypes::DomainToWisdom)?,
        GetStrategy::default(),
    )?;

    let mut ranked = Vec::new();

    for link in links {
        if let Some(record) = get(link.target.into_entry_hash().unwrap(), GetOptions::default())? {
            // Try to get as ActivatedWisdom first (has ratings)
            if let Some(activated) = record.entry().to_app_option::<ActivatedWisdom>().ok().flatten() {
                let rank_score = calculate_wisdom_rank(&activated);
                ranked.push(RankedWisdom {
                    wisdom: activated.wisdom_seed,
                    source_prediction_id: activated.prediction_id,
                    was_correct: activated.activation.was_prediction_correct,
                    usefulness_rating: activated.usefulness_rating,
                    rating_count: activated.rating_count,
                    surface_count: activated.surface_count,
                    rank_score,
                });
            }
        }
    }

    // Sort by rank score
    ranked.sort_by(|a, b| {
        b.rank_score.partial_cmp(&a.rank_score).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ranked.into_iter().take(input.limit.unwrap_or(20)).collect())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetTopWisdomInput {
    pub domain: String,
    pub limit: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RankedWisdom {
    pub wisdom: WisdomSeed,
    pub source_prediction_id: EntryHash,
    pub was_correct: Option<bool>,
    pub usefulness_rating: f64,
    pub rating_count: u64,
    pub surface_count: u64,
    pub rank_score: f64,
}

fn calculate_wisdom_rank(activated: &ActivatedWisdom) -> f64 {
    // Factors:
    // 1. Usefulness rating (0-1, weight: 40%)
    // 2. Prediction correctness (adds 0.2 if correct)
    // 3. Rating confidence (more ratings = more confident)
    // 4. Recency (newer = slightly higher)

    let rating_factor = activated.usefulness_rating * 0.4;

    let correctness_factor = match activated.activation.was_prediction_correct {
        Some(true) => 0.2,
        Some(false) => 0.1, // Still valuable - lessons from failures
        None => 0.05,
    };

    // Confidence: asymptotic function approaching 0.3 as ratings increase
    let confidence_factor = 0.3 * (1.0 - 1.0 / (1.0 + activated.rating_count as f64 / 10.0));

    // Surface count indicates engagement
    let engagement_factor = 0.1 * (activated.surface_count as f64 / 100.0).min(1.0);

    rating_factor + correctness_factor + confidence_factor + engagement_factor
}

// ============================================================================
// VESTING HELPER FUNCTIONS
// ============================================================================

fn execute_vested_withdrawal(
    recipient: &AgentPubKey,
    amount: u64,
    prediction_id: &EntryHash,
) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct VestedWithdrawalInput {
        pub to: AgentPubKey,
        pub amount: u64,
        pub prediction_id: EntryHash,
        pub withdrawal_type: String,
    }

    let withdrawal = VestedWithdrawalInput {
        to: recipient.clone(),
        amount,
        prediction_id: prediction_id.clone(),
        withdrawal_type: "vested".to_string(),
    };

    let response = call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::new("kredit"),
        FunctionName::new("release_vested_funds"),
        None,
        withdrawal,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Withdrawal failed: {:?}", e))))?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        _ => Err(wasm_error!(WasmErrorInner::Guest("Withdrawal failed".into()))),
    }
}

fn execute_early_withdrawal(
    recipient: &AgentPubKey,
    amount: u64,
    penalty: u64,
    prediction_id: &EntryHash,
) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct EarlyWithdrawalInput {
        pub to: AgentPubKey,
        pub amount: u64,
        pub penalty: u64,
        pub prediction_id: EntryHash,
    }

    let withdrawal = EarlyWithdrawalInput {
        to: recipient.clone(),
        amount,
        penalty,
        prediction_id: prediction_id.clone(),
    };

    let response = call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::new("kredit"),
        FunctionName::new("process_early_withdrawal"),
        None,
        withdrawal,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Early withdrawal failed: {:?}", e))))?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        _ => Err(wasm_error!(WasmErrorInner::Guest("Early withdrawal failed".into()))),
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ReputationPenaltyInput {
    pub agent: AgentPubKey,
    pub domain: String,
    pub delta: f64,
    pub reason: String,
}

fn apply_early_withdrawal_reputation_penalty(
    agent: &AgentPubKey,
    penalty_rate: f64,
) -> ExternResult<()> {
    // Early withdrawal impacts reputation proportionally
    let reputation_impact = -0.02 * penalty_rate; // Up to -0.6% reputation

    let penalty = ReputationPenaltyInput {
        agent: agent.clone(),
        domain: "epistemic_markets".to_string(),
        delta: reputation_impact,
        reason: format!("Early vesting withdrawal with {:.1}% penalty", penalty_rate * 100.0),
    };

    let response = call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::new("matl"),
        FunctionName::new("update_reputation"),
        None,
        penalty,
    );

    // Non-critical - don't fail the withdrawal
    match response {
        Ok(_) => Ok(()),
        Err(_) => Ok(()),
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
    Prediction(Prediction),
    ActivatedWisdom(ActivatedWisdom),
}

#[hdk_link_types]
pub enum LinkTypes {
    MarketToPrediction,
    PredictorToPrediction,
    DependencyLink,
    LineageLink,
    DomainToWisdom,
}
