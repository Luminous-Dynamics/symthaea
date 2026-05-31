# Advanced Epistemic Market Mechanisms

## Beyond Prediction: Toward Collective Intelligence Infrastructure

The initial implementation handles prediction well. But true collective intelligence requires mechanisms that:

1. **Surface disagreement** rather than just consensus
2. **Reward information discovery** not just accurate prediction
3. **Enable long-term thinking** against short-term incentives
4. **Prevent manipulation** beyond Byzantine tolerance
5. **Model belief dependencies** not isolated predictions
6. **Integrate human-AI collaboration** as first-class citizens

---

## I. Belief Graphs: Modeling Epistemic Dependencies

### The Problem with Isolated Predictions

Current prediction markets treat each question independently. But beliefs are interconnected:
- "Will AI achieve AGI by 2030?" depends on "Will compute costs continue declining?"
- "Will democracy survive?" depends on "Will social media be regulated?"

### Solution: Directed Acyclic Belief Graphs (DABGs)

```rust
/// A node in the belief graph
pub struct BeliefNode {
    pub id: EntryHash,
    pub claim: String,
    pub current_probability: f64,

    /// Incoming edges: beliefs this depends on
    pub dependencies: Vec<BeliefDependency>,

    /// Outgoing edges: beliefs that depend on this
    pub dependents: Vec<EntryHash>,

    /// Conditional probability table
    pub cpt: ConditionalProbabilityTable,
}

pub struct BeliefDependency {
    pub source_belief: EntryHash,
    pub influence_type: InfluenceType,
    pub strength: f64,  // -1.0 to 1.0
}

pub enum InfluenceType {
    /// If A is true, B is more likely
    Supports,
    /// If A is true, B is less likely
    Contradicts,
    /// A is necessary for B
    Prerequisite,
    /// A causes B
    Causal,
    /// A and B share common cause
    Correlated,
}

/// Conditional probability table for Bayesian updates
pub struct ConditionalProbabilityTable {
    /// P(this | parent_states)
    pub entries: HashMap<Vec<bool>, f64>,
}

impl BeliefGraph {
    /// When a belief updates, propagate through the graph
    pub fn propagate_update(&mut self, updated_node: EntryHash, new_prob: f64) {
        // Use belief propagation algorithm (Pearl's message passing)
        let mut messages: HashMap<(EntryHash, EntryHash), f64> = HashMap::new();

        // Forward pass: update dependents
        for dependent in &self.get_node(updated_node).dependents {
            let message = self.compute_message(updated_node, *dependent, new_prob);
            messages.insert((updated_node, *dependent), message);

            // Recursively update
            let new_dependent_prob = self.compute_belief(*dependent, &messages);
            self.propagate_update(*dependent, new_dependent_prob);
        }
    }

    /// Find beliefs that would most change if we learned X
    pub fn information_value(&self, query_node: EntryHash) -> Vec<(EntryHash, f64)> {
        let mut impacts = Vec::new();

        // Calculate expected entropy reduction for each dependent
        for node_id in self.all_nodes() {
            let current_entropy = self.entropy(node_id);
            let expected_entropy_if_true = self.conditional_entropy(node_id, query_node, true);
            let expected_entropy_if_false = self.conditional_entropy(node_id, query_node, false);

            let prob_true = self.get_node(query_node).current_probability;
            let expected_reduction = current_entropy
                - (prob_true * expected_entropy_if_true
                   + (1.0 - prob_true) * expected_entropy_if_false);

            impacts.push((node_id, expected_reduction));
        }

        impacts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        impacts
    }
}
```

### Trading on Belief Graph Edges

Markets can trade on the *strength of connections*, not just individual beliefs:

```rust
pub struct EdgeMarket {
    pub source_belief: EntryHash,
    pub target_belief: EntryHash,

    /// Current market estimate of influence strength
    pub influence_strength: f64,  // -1.0 to 1.0

    /// How this resolves: correlation after both beliefs resolve
    pub resolution_method: EdgeResolutionMethod,
}

pub enum EdgeResolutionMethod {
    /// Compute empirical correlation from resolved beliefs
    EmpiricalCorrelation,

    /// Expert panel assesses causal relationship
    ExpertAssessment,

    /// Granger causality test if time series
    GrangerCausality,

    /// Intervention study
    Intervention { study_hash: EntryHash },
}
```

**Why this matters**: Understanding *why* beliefs connect is often more valuable than the beliefs themselves. This creates a tradeable map of collective epistemology.

---

## II. Disagreement Mining: Rewarding Productive Conflict

### The Problem with Consensus

Markets naturally converge. But premature consensus destroys information:
- Minority views get drowned out
- Contrarians are punished even when they have signal
- Groupthink emerges

### Solution: Explicit Disagreement Markets

```rust
/// A market specifically for surfacing and exploring disagreement
pub struct DisagreementMarket {
    pub base_market: EntryHash,
    pub thesis: String,
    pub antithesis: String,

    /// Agents who have taken each side
    pub thesis_holders: Vec<DisagreementPosition>,
    pub antithesis_holders: Vec<DisagreementPosition>,

    /// The crux: what would change minds?
    pub identified_cruxes: Vec<Crux>,

    /// Resolution: which side provided better epistemic contribution?
    pub epistemic_contribution_scores: HashMap<AgentPubKey, f64>,
}

pub struct DisagreementPosition {
    pub agent: AgentPubKey,
    pub position: Position,
    pub confidence: f64,
    pub reasoning: String,

    /// What evidence would change your mind?
    pub cruxes: Vec<Crux>,

    /// Acknowledgments of opposing side's valid points
    pub steel_mans: Vec<SteelMan>,
}

pub struct Crux {
    pub description: String,
    pub operationalization: String,  // How to test this
    pub threshold: f64,  // What result would change mind
    pub importance: f64,  // How much this matters to position
}

pub struct SteelMan {
    pub opposing_point: String,
    pub acknowledgment: String,
    pub how_it_affects_confidence: f64,
}

impl DisagreementMarket {
    /// Calculate epistemic contribution score
    pub fn score_contribution(&self, agent: &AgentPubKey) -> f64 {
        let position = self.get_position(agent)?;

        let mut score = 0.0;

        // Reward identifying cruxes that proved decisive
        for crux in &position.cruxes {
            if self.crux_was_decisive(crux) {
                score += 100.0 * crux.importance;
            }
        }

        // Reward steel-manning (acknowledging valid opposing points)
        score += position.steel_mans.len() as f64 * 20.0;

        // Reward changing mind when evidence warranted
        if self.agent_updated_appropriately(agent) {
            score += 50.0;
        }

        // Penalize overconfidence (calibration)
        let calibration_penalty = self.calculate_calibration_error(agent);
        score -= calibration_penalty * 30.0;

        // Reward bringing new information
        for evidence in self.evidence_submitted_by(agent) {
            if self.evidence_was_novel(evidence) {
                score += 40.0 * evidence.impact;
            }
        }

        score
    }

    /// Find the most productive disagreements to resolve
    pub fn prioritize_cruxes(&self) -> Vec<(Crux, f64)> {
        let mut crux_values = Vec::new();

        for crux in &self.identified_cruxes {
            // Value = (people who'd change mind) × (average confidence change) × (testability)
            let minds_changed = self.count_would_change_mind(crux);
            let avg_confidence_shift = self.average_confidence_shift(crux);
            let testability = crux.estimate_testability();

            let value = minds_changed as f64 * avg_confidence_shift * testability;
            crux_values.push((crux.clone(), value));
        }

        crux_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        crux_values
    }
}
```

### Double Crux Protocol

Implement the CFAR "Double Crux" technique as a market mechanism:

```rust
pub struct DoubleCruxSession {
    pub participants: (AgentPubKey, AgentPubKey),
    pub topic: String,
    pub initial_positions: (f64, f64),  // Initial probabilities

    /// The search for shared cruxes
    pub rounds: Vec<DoubleCruxRound>,

    /// Found double crux (if any)
    pub double_crux: Option<DoubleCrux>,

    /// Final positions after deliberation
    pub final_positions: Option<(f64, f64)>,
}

pub struct DoubleCruxRound {
    pub round_number: u32,
    pub a_proposed_crux: Crux,
    pub b_proposed_crux: Crux,
    pub a_response_to_b: CruxResponse,
    pub b_response_to_a: CruxResponse,
}

pub struct DoubleCrux {
    /// A proposition that BOTH would change minds about
    pub proposition: String,
    pub a_if_true_shift: f64,
    pub a_if_false_shift: f64,
    pub b_if_true_shift: f64,
    pub b_if_false_shift: f64,
}

impl DoubleCruxSession {
    /// Reward participants for finding double crux
    pub fn calculate_rewards(&self) -> (f64, f64) {
        if let Some(ref dc) = self.double_crux {
            // Both participants rewarded for finding shared crux
            let base_reward = 100.0;

            // Bonus for larger potential belief updates
            let a_update_magnitude = (dc.a_if_true_shift - dc.a_if_false_shift).abs();
            let b_update_magnitude = (dc.b_if_true_shift - dc.b_if_false_shift).abs();

            let a_reward = base_reward * (1.0 + a_update_magnitude);
            let b_reward = base_reward * (1.0 + b_update_magnitude);

            (a_reward, b_reward)
        } else {
            // Smaller reward for good-faith participation
            (20.0, 20.0)
        }
    }
}
```

**Why this matters**: Most valuable information comes from *resolving* disagreements, not *suppressing* them. This creates explicit incentives to find the cruxes.

---

## III. Attention Markets: What Deserves Collective Focus?

### The Problem with Implicit Attention

Current systems let attention flow to whatever is viral/engaging, not what's important.

### Solution: Explicit Markets for Attention Allocation

```rust
/// A market for collective attention allocation
pub struct AttentionMarket {
    pub id: EntryHash,
    pub domain: String,

    /// Topics competing for attention
    pub topics: Vec<AttentionTopic>,

    /// Total attention budget for this domain (in attention-hours)
    pub total_budget: f64,

    /// Current allocation
    pub current_allocation: HashMap<EntryHash, f64>,

    /// Resolution: was attention well-spent?
    pub retrospective: Option<AttentionRetrospective>,
}

pub struct AttentionTopic {
    pub id: EntryHash,
    pub title: String,
    pub description: String,
    pub proposer: AgentPubKey,

    /// Expected value of paying attention
    pub expected_value: AttentionValue,

    /// Current shares (determines attention allocation)
    pub shares: u64,
}

pub struct AttentionValue {
    /// Will attending to this change important decisions?
    pub decision_relevance: f64,

    /// How much uncertainty can be reduced?
    pub uncertainty_reduction: f64,

    /// Time-sensitivity: does this need attention NOW?
    pub urgency: f64,

    /// Will insights here transfer to other domains?
    pub transferability: f64,

    /// Is this being neglected despite importance?
    pub neglectedness: f64,
}

impl AttentionValue {
    /// Calculate composite attention worthiness
    pub fn composite(&self) -> f64 {
        // ITN framework inspired (Importance, Tractability, Neglectedness)
        let importance = self.decision_relevance * self.uncertainty_reduction;
        let tractability = 1.0 / (1.0 + (-self.transferability).exp());
        let neglectedness = self.neglectedness;

        importance * tractability * neglectedness * (1.0 + self.urgency)
    }
}

pub struct AttentionRetrospective {
    pub period_end: u64,

    /// Did attention lead to valuable insights?
    pub insights_generated: Vec<InsightRecord>,

    /// Did attention lead to better decisions?
    pub decisions_informed: Vec<DecisionRecord>,

    /// Hindsight: should we have allocated differently?
    pub optimal_allocation_estimate: HashMap<EntryHash, f64>,

    /// Reward distribution based on retrospective
    pub rewards: HashMap<AgentPubKey, f64>,
}

impl AttentionMarket {
    /// Score attention allocators based on retrospective
    pub fn score_allocation(&self, agent: &AgentPubKey) -> f64 {
        let retro = self.retrospective.as_ref()?;

        // Get agent's predicted allocation
        let agent_allocation = self.agent_allocation(agent);

        // Compare to optimal (hindsight)
        let mut score = 0.0;
        for (topic, agent_weight) in &agent_allocation {
            let optimal_weight = retro.optimal_allocation_estimate.get(topic).unwrap_or(&0.0);
            let accuracy = 1.0 - (agent_weight - optimal_weight).abs();
            score += accuracy;
        }

        // Bonus for topics that were neglected but agent identified
        for (topic, agent_weight) in &agent_allocation {
            let was_neglected = self.topics.iter()
                .find(|t| &t.id == topic)
                .map(|t| t.expected_value.neglectedness > 0.7)
                .unwrap_or(false);

            let turned_out_important = retro.optimal_allocation_estimate
                .get(topic)
                .map(|w| *w > 0.2)
                .unwrap_or(false);

            if was_neglected && turned_out_important && *agent_weight > 0.1 {
                score += 50.0;  // Big bonus for identifying neglected important topics
            }
        }

        score
    }
}
```

**Why this matters**: Collective attention is our scarcest resource. Markets that explicitly allocate it can dramatically improve what humanity focuses on.

---

## IV. Long-Horizon Mechanisms: Defeating Temporal Myopia

### The Problem with Short-Term Incentives

Prediction markets favor short-term predictions:
- Faster resolution = faster payoff
- Discount rates make distant predictions worth less
- Predictors exit markets before resolution

### Solution: Temporal Commitment Mechanisms

```rust
/// Long-horizon prediction with commitment mechanisms
pub struct LongHorizonPrediction {
    pub market_id: EntryHash,
    pub predictor: AgentPubKey,
    pub prediction: Prediction,

    /// Commitment tier
    pub commitment: TemporalCommitment,

    /// Rolling updates (public accountability)
    pub updates: Vec<PredictionUpdate>,

    /// Legacy stake: what happens after predictor's participation ends
    pub legacy_stake: LegacyStake,
}

pub enum TemporalCommitment {
    /// Standard: can exit anytime
    Standard,

    /// Locked: stake locked until resolution
    Locked { unlock_date: u64 },

    /// Vesting: stake unlocks gradually
    Vesting {
        schedule: Vec<(u64, f64)>,  // (timestamp, percentage)
    },

    /// Legacy: stake transfers to successors
    Legacy {
        successor_policy: SuccessorPolicy,
    },

    /// Perpetual: rolling predictions that never fully resolve
    Perpetual {
        update_frequency: Duration,
        scoring_window: Duration,
    },
}

pub struct PredictionUpdate {
    pub timestamp: u64,
    pub new_probability: f64,
    pub reasoning: String,

    /// What changed since last update?
    pub delta_explanation: String,

    /// Information sources consulted
    pub sources: Vec<String>,
}

pub struct LegacyStake {
    /// If predictor becomes inactive, who inherits?
    pub successor: SuccessorPolicy,

    /// Reputation continues to accumulate/decay
    pub reputation_continuity: bool,

    /// Can designate beneficiary for payouts
    pub beneficiary: Option<AgentPubKey>,
}

pub enum SuccessorPolicy {
    /// Stake returns to pool
    ReturnToPool,

    /// Designated successor inherits position
    Designated(AgentPubKey),

    /// Auction position to highest bidder
    Auction,

    /// Community votes on successor
    CommunityElection,
}

/// Perpetual prediction market (never fully resolves)
pub struct PerpetualMarket {
    pub question: String,  // e.g., "What is the probability of human extinction by 2100?"

    /// Rolling window scoring
    pub scoring_windows: Vec<ScoringWindow>,

    /// Current aggregate prediction
    pub current_prediction: f64,

    /// Prediction stability over time
    pub stability_metrics: StabilityMetrics,
}

pub struct ScoringWindow {
    pub start: u64,
    pub end: u64,

    /// Partial resolution signals during window
    pub signals: Vec<PartialSignal>,

    /// Score based on signals
    pub scores: HashMap<AgentPubKey, f64>,
}

pub struct PartialSignal {
    pub timestamp: u64,
    pub signal_type: String,
    pub value: f64,
    pub weight: f64,  // How much this signal matters
}

impl PerpetualMarket {
    /// Score predictions based on partial signals
    pub fn score_window(&self, window: &ScoringWindow) -> HashMap<AgentPubKey, f64> {
        let mut scores = HashMap::new();

        for (agent, prediction) in self.predictions_during(window) {
            let mut agent_score = 0.0;

            for signal in &window.signals {
                // Brier-like scoring against partial signals
                let error = (prediction.probability - signal.value).powi(2);
                agent_score -= error * signal.weight;
            }

            // Bonus for prediction stability (not just chasing signals)
            let stability_bonus = self.stability_score(&agent, window);
            agent_score += stability_bonus * 10.0;

            scores.insert(agent, agent_score);
        }

        scores
    }
}
```

### Intergenerational Prediction Markets

```rust
/// Markets that span generations
pub struct IntergenerationalMarket {
    pub question: String,  // e.g., "Will fusion power be commercially viable by 2100?"
    pub created_at: u64,
    pub target_resolution: u64,  // Could be 50+ years

    /// Generational cohorts
    pub cohorts: Vec<Cohort>,

    /// How knowledge transfers between cohorts
    pub knowledge_transfer: KnowledgeTransferProtocol,

    /// Institutional backing (survives individual participation)
    pub institutional_sponsors: Vec<Institution>,
}

pub struct Cohort {
    pub generation: u32,
    pub active_period: (u64, u64),
    pub members: Vec<AgentPubKey>,

    /// Aggregate prediction from this cohort
    pub cohort_prediction: f64,

    /// Documentation for future cohorts
    pub reasoning_archive: Vec<ReasoningDocument>,
}

pub struct KnowledgeTransferProtocol {
    /// How predictions pass to next generation
    pub prediction_inheritance: InheritanceMethod,

    /// Required documentation for successors
    pub documentation_requirements: Vec<String>,

    /// Mentorship requirements
    pub mentorship: MentorshipRequirements,
}

pub enum InheritanceMethod {
    /// Direct transfer: next gen inherits positions
    Direct,

    /// Fresh start: next gen makes own predictions informed by archive
    FreshWithArchive,

    /// Bayesian: next gen's prior is previous gen's posterior
    BayesianUpdate,
}
```

**Why this matters**: Humanity's biggest challenges (climate, AI safety, existential risk) require long-horizon thinking that current markets don't support.

---

## V. Anti-Manipulation Beyond Byzantine Tolerance

### Attack Vectors Not Covered by MATL

1. **Wealth attacks**: Rich actors can afford to lose to manipulate
2. **Information cascades**: Everyone copies early predictors
3. **Sybil armies**: Create many identities with legitimate MATL
4. **Oracle collusion**: Oracles coordinate outside the system
5. **Self-fulfilling prophecies**: Predictions that cause their outcomes

### Comprehensive Anti-Manipulation Framework

```rust
pub struct ManipulationDetector {
    /// Multiple detection strategies
    pub detectors: Vec<Box<dyn DetectionStrategy>>,

    /// Anomaly threshold for intervention
    pub intervention_threshold: f64,

    /// Actions to take when manipulation detected
    pub interventions: Vec<Intervention>,
}

pub trait DetectionStrategy {
    fn analyze(&self, market: &Market, predictions: &[Prediction]) -> ManipulationScore;
}

pub struct ManipulationScore {
    pub overall: f64,
    pub breakdown: HashMap<String, f64>,
    pub evidence: Vec<ManipulationEvidence>,
}

/// Detect wealth-based manipulation
pub struct WealthConcentrationDetector;

impl DetectionStrategy for WealthConcentrationDetector {
    fn analyze(&self, market: &Market, predictions: &[Prediction]) -> ManipulationScore {
        // Calculate Gini coefficient of stakes
        let stakes: Vec<f64> = predictions.iter()
            .map(|p| p.stake.total_value())
            .collect();

        let gini = calculate_gini(&stakes);

        // High concentration + large price movement = suspicious
        let price_movement = market.price_volatility();
        let score = gini * price_movement;

        ManipulationScore {
            overall: score,
            breakdown: hashmap! {
                "gini".to_string() => gini,
                "price_movement".to_string() => price_movement,
            },
            evidence: vec![],
        }
    }
}

/// Detect information cascades
pub struct CascadeDetector;

impl DetectionStrategy for CascadeDetector {
    fn analyze(&self, market: &Market, predictions: &[Prediction]) -> ManipulationScore {
        // Look for herding behavior
        let mut cascade_score = 0.0;

        // Sort predictions by time
        let mut sorted = predictions.to_vec();
        sorted.sort_by_key(|p| p.timestamp);

        // Check if later predictions just copy earlier ones
        for window in sorted.windows(10) {
            let early_avg: f64 = window[..3].iter().map(|p| p.confidence).sum::<f64>() / 3.0;
            let late_avg: f64 = window[7..].iter().map(|p| p.confidence).sum::<f64>() / 3.0;

            // High correlation between early and late = possible cascade
            let correlation = 1.0 - (early_avg - late_avg).abs();

            // But check if information actually changed
            let new_info = market.information_events_in_window(window);

            if correlation > 0.9 && new_info.is_empty() {
                cascade_score += 0.1;
            }
        }

        ManipulationScore {
            overall: cascade_score,
            breakdown: hashmap! { "cascade_score".to_string() => cascade_score },
            evidence: vec![],
        }
    }
}

/// Detect Sybil attacks via behavioral analysis
pub struct SybilDetector;

impl DetectionStrategy for SybilDetector {
    fn analyze(&self, market: &Market, predictions: &[Prediction]) -> ManipulationScore {
        let mut sybil_clusters: Vec<Vec<AgentPubKey>> = vec![];

        // Cluster agents by behavioral similarity
        for i in 0..predictions.len() {
            for j in (i+1)..predictions.len() {
                let sim = behavioral_similarity(&predictions[i], &predictions[j]);

                if sim > 0.95 {
                    // Suspiciously similar behavior
                    // Timing, amounts, confidence levels all match
                    add_to_cluster(&mut sybil_clusters,
                        &predictions[i].predictor,
                        &predictions[j].predictor);
                }
            }
        }

        // Score based on largest cluster
        let largest_cluster = sybil_clusters.iter().map(|c| c.len()).max().unwrap_or(0);
        let score = (largest_cluster as f64 / predictions.len() as f64).min(1.0);

        ManipulationScore {
            overall: score,
            breakdown: hashmap! {
                "largest_cluster".to_string() => largest_cluster as f64,
                "cluster_count".to_string() => sybil_clusters.len() as f64,
            },
            evidence: sybil_clusters.iter()
                .map(|c| ManipulationEvidence::SybilCluster(c.clone()))
                .collect(),
        }
    }
}

/// Intervention strategies
pub enum Intervention {
    /// Increase required MATL for participation
    IncreaseMatl { new_minimum: f64 },

    /// Add delay between prediction and effect
    AddDelay { duration: Duration },

    /// Require diverse stake types
    RequireDiverseStakes { min_types: u32 },

    /// Cap individual influence
    CapInfluence { max_percentage: f64 },

    /// Require reasoning disclosure
    RequireReasoning { min_length: u32 },

    /// Pause market for review
    PauseMarket { duration: Duration },

    /// Escalate to governance
    Escalate,
}
```

### Self-Fulfilling Prophecy Detector

```rust
pub struct SelfFulfillingDetector {
    /// Track markets where predictions might cause outcomes
    pub causal_markets: HashSet<EntryHash>,
}

impl SelfFulfillingDetector {
    /// Identify markets at risk of self-fulfilling dynamics
    pub fn assess_causal_risk(&self, market: &Market) -> CausalRisk {
        let mut risk = CausalRisk::default();

        // Check if market outcome could be influenced by market participants
        let participant_influence = self.estimate_participant_influence(market);
        risk.participant_influence = participant_influence;

        // Check if prediction publication could affect outcome
        let publication_effect = self.estimate_publication_effect(market);
        risk.publication_effect = publication_effect;

        // Check for coordination games
        let coordination_risk = self.is_coordination_game(market);
        risk.coordination_game = coordination_risk;

        risk
    }

    /// Mitigations for self-fulfilling prophecies
    pub fn suggest_mitigations(&self, risk: &CausalRisk) -> Vec<Mitigation> {
        let mut mitigations = vec![];

        if risk.publication_effect > 0.5 {
            // Delay publication until close to resolution
            mitigations.push(Mitigation::DelayedPublication {
                delay_until: risk.suggested_publication_time,
            });
        }

        if risk.participant_influence > 0.3 {
            // Exclude influential parties
            mitigations.push(Mitigation::ExcludeInfluential {
                influence_threshold: 0.1,
            });
        }

        if risk.coordination_game {
            // Use sealed-bid prediction
            mitigations.push(Mitigation::SealedBid {
                reveal_after: risk.resolution_time,
            });
        }

        mitigations
    }
}
```

---

## VI. Human-AI Collaborative Prediction

### The Problem with Siloed Intelligence

Humans have intuition, context, and values. AIs have compute, consistency, and breadth. Keeping them separate wastes potential.

### Solution: Structured Human-AI Collaboration

```rust
pub struct HumanAITeam {
    pub human: AgentPubKey,
    pub ai: AIOracle,

    pub collaboration_protocol: CollaborationProtocol,
    pub shared_reasoning: SharedReasoningSpace,

    /// Performance tracking
    pub team_performance: TeamPerformance,
}

pub enum CollaborationProtocol {
    /// AI provides base rate, human adjusts
    AIBaseHumanAdjust {
        max_adjustment: f64,
        requires_justification: bool,
    },

    /// Human provides intuition, AI stress-tests
    HumanIntuitionAIChallenge {
        challenge_types: Vec<ChallengeType>,
    },

    /// Independent predictions, then synthesis
    IndependentSynthesis {
        synthesis_method: SynthesisMethod,
    },

    /// Deliberative: multiple rounds of exchange
    Deliberative {
        max_rounds: u32,
        convergence_threshold: f64,
    },

    /// Division of labor by question type
    DomainSpecific {
        ai_domains: Vec<String>,
        human_domains: Vec<String>,
        collaborative_domains: Vec<String>,
    },
}

pub enum ChallengeType {
    /// Find counter-examples
    CounterExample,
    /// Identify base rate neglect
    BaseRateCheck,
    /// Check for availability bias
    AvailabilityBiasCheck,
    /// Stress-test edge cases
    EdgeCases,
    /// Find contradictions in reasoning
    ConsistencyCheck,
}

pub struct SharedReasoningSpace {
    /// Shared working memory
    pub facts: Vec<Fact>,
    pub hypotheses: Vec<Hypothesis>,
    pub evidence: Vec<Evidence>,

    /// Points of agreement
    pub agreements: Vec<Agreement>,

    /// Points of disagreement
    pub disagreements: Vec<Disagreement>,

    /// Open questions
    pub open_questions: Vec<String>,
}

impl HumanAITeam {
    /// Run deliberative protocol
    pub async fn deliberate(&mut self, question: &str) -> TeamPrediction {
        let protocol = match &self.collaboration_protocol {
            CollaborationProtocol::Deliberative { max_rounds, convergence_threshold } => {
                (*max_rounds, *convergence_threshold)
            },
            _ => (5, 0.05),  // Default
        };

        let mut human_pred = self.human.initial_prediction(question).await?;
        let mut ai_pred = self.ai.initial_prediction(question).await?;

        for round in 0..protocol.0 {
            // AI challenges human reasoning
            let ai_challenges = self.ai.challenge(&human_pred, &self.shared_reasoning).await?;

            // Human updates based on challenges
            let human_response = self.human.respond_to_challenges(ai_challenges).await?;
            human_pred = human_response.updated_prediction;

            // Human challenges AI reasoning
            let human_challenges = self.human.challenge(&ai_pred, &self.shared_reasoning).await?;

            // AI updates based on challenges
            let ai_response = self.ai.respond_to_challenges(human_challenges).await?;
            ai_pred = ai_response.updated_prediction;

            // Update shared reasoning space
            self.shared_reasoning.incorporate(&human_response);
            self.shared_reasoning.incorporate(&ai_response);

            // Check convergence
            if (human_pred.probability - ai_pred.probability).abs() < protocol.1 {
                break;
            }
        }

        // Synthesize final prediction
        self.synthesize(human_pred, ai_pred)
    }

    fn synthesize(&self, human: Prediction, ai: Prediction) -> TeamPrediction {
        // Weight by historical accuracy in this domain
        let human_weight = self.team_performance.human_accuracy_in_domain(&human.domain);
        let ai_weight = self.team_performance.ai_accuracy_in_domain(&ai.domain);

        let total = human_weight + ai_weight;
        let combined_prob = (human.probability * human_weight + ai.probability * ai_weight) / total;

        TeamPrediction {
            probability: combined_prob,
            human_contribution: human,
            ai_contribution: ai,
            reasoning: self.shared_reasoning.summarize(),
            confidence_in_synthesis: self.estimate_synthesis_quality(),
        }
    }
}

/// Track when human-AI teams outperform individuals
pub struct TeamPerformance {
    pub predictions: Vec<ResolvedTeamPrediction>,

    /// Brier scores by configuration
    pub human_alone: f64,
    pub ai_alone: f64,
    pub team_combined: f64,

    /// Domains where collaboration helps most
    pub collaboration_benefit_by_domain: HashMap<String, f64>,
}

impl TeamPerformance {
    /// Calculate collaboration benefit
    pub fn collaboration_benefit(&self) -> f64 {
        let best_individual = self.human_alone.min(self.ai_alone);
        (best_individual - self.team_combined) / best_individual
    }

    /// Identify domains where collaboration helps most
    pub fn best_collaboration_domains(&self) -> Vec<(String, f64)> {
        let mut domains: Vec<_> = self.collaboration_benefit_by_domain.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        domains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        domains
    }
}
```

---

## VII. Meta-Markets: Markets About Markets

### Trading on System Performance

```rust
/// Meta-prediction market about prediction market performance
pub struct MetaMarket {
    pub target_market: EntryHash,
    pub meta_question: MetaQuestion,

    /// Resolution based on target market's performance
    pub resolution_criteria: MetaResolutionCriteria,
}

pub enum MetaQuestion {
    /// What will the final price be?
    FinalPrice {
        target_market: EntryHash,
        price_ranges: Vec<(f64, f64)>,  // Outcome buckets
    },

    /// Will this market be manipulated?
    ManipulationDetection {
        target_market: EntryHash,
    },

    /// How accurate will the market's prediction be?
    MarketAccuracy {
        target_market: EntryHash,
        accuracy_metric: AccuracyMetric,
    },

    /// Will the resolution be disputed?
    ResolutionDispute {
        target_market: EntryHash,
    },

    /// How much liquidity will the market attract?
    LiquidityPrediction {
        target_market: EntryHash,
        liquidity_ranges: Vec<(u64, u64)>,
    },

    /// System-wide: what will overall calibration be?
    SystemCalibration {
        time_period: (u64, u64),
        domain: Option<String>,
    },
}

pub enum AccuracyMetric {
    BrierScore,
    LogScore,
    CalibrationError,
    Discrimination,
}

/// Meta-incentives: rewards for improving the system itself
pub struct SystemImprovement {
    pub improvement_type: ImprovementType,
    pub proposer: AgentPubKey,
    pub evidence: Vec<EvidenceHash>,
    pub verified: bool,
    pub reward: u64,
}

pub enum ImprovementType {
    /// Found and reported a bug
    BugReport { severity: Severity },

    /// Proposed mechanism improvement
    MechanismImprovement {
        description: String,
        simulation_results: Option<SimulationResults>,
    },

    /// Identified manipulation attempt
    ManipulationReport {
        market_id: EntryHash,
        evidence: ManipulationEvidence,
    },

    /// Improved calibration methodology
    CalibrationImprovement {
        before_metrics: CalibrationMetrics,
        after_metrics: CalibrationMetrics,
    },

    /// Created valuable educational content
    Education {
        content_hash: EntryHash,
        engagement_metrics: EngagementMetrics,
    },
}
```

---

## VIII. Implementation Priorities

Based on impact and feasibility:

### Immediate (Phase 2)

1. **Disagreement Mining** - Explicit crux identification in existing markets
2. **Cascade Detection** - Add to manipulation detector
3. **Long-horizon Commitments** - Vesting and locking options

### Medium-term (Phase 3)

4. **Belief Graphs** - Start with simple dependency tracking
5. **Human-AI Teams** - Structured collaboration protocols
6. **Attention Markets** - For Question Markets prioritization

### Long-term (Phase 4+)

7. **Intergenerational Markets** - Requires institutional infrastructure
8. **Meta-Markets** - After base system is stable
9. **Full Bayesian Belief Propagation** - Complex but powerful

---

## Summary: From Prediction to Collective Intelligence

| Current State | Enhanced State |
|---------------|----------------|
| Aggregate beliefs | **Generate new knowledge** |
| Reward accuracy | **Reward information discovery** |
| Consensus-seeking | **Disagreement-mining** |
| Short-term focus | **Long-horizon commitment** |
| Individual predictions | **Human-AI collaboration** |
| Isolated markets | **Belief graph networks** |
| Implicit attention | **Explicit attention markets** |
| Post-hoc analysis | **Meta-markets for system improvement** |

These mechanisms transform Mycelix Epistemic Markets from a prediction system into genuine **collective intelligence infrastructure**—a system that makes humanity smarter, not just better at betting.
