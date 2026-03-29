// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Simulation Framework for Epistemic Markets Research
//!
//! This module provides infrastructure for running controlled experiments:
//! - Agent simulation with configurable behaviors
//! - Market dynamics simulation
//! - A/B testing framework
//! - Data collection and export

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// SIMULATION CONFIGURATION
// ============================================================================

/// Configuration for a simulation experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Unique identifier for this simulation run
    pub simulation_id: String,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of simulated agents
    pub num_agents: usize,
    /// Number of markets to simulate
    pub num_markets: usize,
    /// Number of time steps
    pub num_steps: usize,
    /// Agent configuration
    pub agent_config: AgentPopulationConfig,
    /// Market configuration
    pub market_config: MarketSimConfig,
    /// Data collection configuration
    pub collection_config: DataCollectionConfig,
    /// Experiment groups (for A/B testing)
    pub experiment_groups: Vec<ExperimentGroup>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            simulation_id: "default".to_string(),
            seed: 42,
            num_agents: 100,
            num_markets: 10,
            num_steps: 1000,
            agent_config: AgentPopulationConfig::default(),
            market_config: MarketSimConfig::default(),
            collection_config: DataCollectionConfig::default(),
            experiment_groups: vec![],
        }
    }
}

/// Configuration for agent population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPopulationConfig {
    /// Distribution of agent types
    pub type_distribution: Vec<(AgentType, f64)>,
    /// Calibration distribution (mean, std)
    pub calibration_distribution: (f64, f64),
    /// Information access distribution
    pub information_distribution: (f64, f64),
    /// Risk aversion distribution
    pub risk_aversion_distribution: (f64, f64),
    /// Domain expertise distribution
    pub domain_expertise: HashMap<String, f64>,
}

impl Default for AgentPopulationConfig {
    fn default() -> Self {
        Self {
            type_distribution: vec![
                (AgentType::Informed, 0.3),
                (AgentType::Noise, 0.4),
                (AgentType::Contrarian, 0.1),
                (AgentType::Herder, 0.2),
            ],
            calibration_distribution: (0.15, 0.1), // Mean ECE, std
            information_distribution: (0.5, 0.2),
            risk_aversion_distribution: (0.5, 0.2),
            domain_expertise: HashMap::new(),
        }
    }
}

/// Types of simulated agents
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentType {
    /// Has access to genuine information
    Informed,
    /// Random noise trader
    Noise,
    /// Contrarian: bets against the crowd
    Contrarian,
    /// Herder: follows the crowd
    Herder,
    /// Manipulator: attempts to move prices
    Manipulator,
    /// Expert: highly calibrated in specific domains
    Expert,
    /// Novice: learning calibration
    Novice,
}

/// Market simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSimConfig {
    /// Market mechanism type
    pub mechanism: SimulatedMechanism,
    /// Base rate distribution for outcomes
    pub base_rate_distribution: (f64, f64),
    /// Average market duration in steps
    pub market_duration: usize,
    /// Information arrival rate
    pub information_rate: f64,
    /// Liquidity parameter (for LMSR)
    pub liquidity_parameter: f64,
}

impl Default for MarketSimConfig {
    fn default() -> Self {
        Self {
            mechanism: SimulatedMechanism::LMSR,
            base_rate_distribution: (0.5, 0.2),
            market_duration: 100,
            information_rate: 0.1,
            liquidity_parameter: 100.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimulatedMechanism {
    LMSR,
    OrderBook,
    Parimutuel,
}

/// Data collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionConfig {
    /// Collect price time series
    pub collect_prices: bool,
    /// Collect individual predictions
    pub collect_predictions: bool,
    /// Collect agent state over time
    pub collect_agent_state: bool,
    /// Snapshot interval (steps between full state snapshots)
    pub snapshot_interval: usize,
    /// Export format
    pub export_format: ExportFormat,
}

impl Default for DataCollectionConfig {
    fn default() -> Self {
        Self {
            collect_prices: true,
            collect_predictions: true,
            collect_agent_state: true,
            snapshot_interval: 10,
            export_format: ExportFormat::Json,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Csv,
    Parquet,
}

/// Experiment group for A/B testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentGroup {
    pub name: String,
    pub description: String,
    /// Percentage of agents assigned to this group
    pub assignment_ratio: f64,
    /// Treatment parameters
    pub treatment: Treatment,
}

/// Treatment conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Treatment {
    /// Training intervention
    pub training: Option<TrainingIntervention>,
    /// Information display modification
    pub information_display: Option<InformationDisplay>,
    /// Incentive structure
    pub incentive_structure: Option<IncentiveStructure>,
    /// Wisdom seed format
    pub wisdom_seed_format: Option<WisdomSeedFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingIntervention {
    CalibrationFeedback { frequency: usize },
    BaseRateTraining,
    OutsideView,
    DebiasigExercises,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InformationDisplay {
    ShowPricesFirst,
    HidePricesUntilPrediction,
    ShowConfidenceDistribution,
    ShowReasoningFirst,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IncentiveStructure {
    BrierScoring,
    LogScoring,
    Parimutuel,
    ReputationOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WisdomSeedFormat {
    NaturalLanguage,
    Structured,
    Minimal,
    Detailed,
}

// ============================================================================
// SIMULATED AGENTS
// ============================================================================

/// A simulated agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedAgent {
    pub id: String,
    pub agent_type: AgentType,
    /// Group assignment for A/B testing
    pub group: Option<String>,
    /// Calibration error (lower is better)
    pub calibration_error: f64,
    /// Information quality (0-1)
    pub information_quality: f64,
    /// Risk aversion (0-1)
    pub risk_aversion: f64,
    /// Domain expertise
    pub domain_expertise: HashMap<String, f64>,
    /// Prediction history
    pub prediction_history: Vec<SimulatedPrediction>,
    /// Current wealth/reputation
    pub wealth: f64,
    pub reputation: f64,
    /// Learning state
    pub learning_state: LearningState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningState {
    /// Cumulative feedback received
    pub feedback_count: usize,
    /// Running calibration estimate
    pub estimated_calibration: f64,
    /// Calibration improvement trajectory
    pub calibration_history: Vec<f64>,
}

impl SimulatedAgent {
    /// Create a new agent with given parameters
    pub fn new(
        id: String,
        agent_type: AgentType,
        calibration_error: f64,
        information_quality: f64,
    ) -> Self {
        Self {
            id,
            agent_type,
            group: None,
            calibration_error,
            information_quality,
            risk_aversion: 0.5,
            domain_expertise: HashMap::new(),
            prediction_history: vec![],
            wealth: 1000.0,
            reputation: 0.5,
            learning_state: LearningState {
                feedback_count: 0,
                estimated_calibration: calibration_error,
                calibration_history: vec![calibration_error],
            },
        }
    }

    /// Generate a prediction for a market
    pub fn make_prediction(
        &self,
        market: &SimulatedMarket,
        current_price: f64,
        rng: &mut impl Rng,
    ) -> SimulatedPrediction {
        let true_prob = market.true_probability;

        // Base estimate depends on information quality
        let private_signal = if rng.gen_f64() < self.information_quality {
            // Received informative signal
            true_prob + rng.gen_range_f64(-0.1, 0.1)
        } else {
            // Noisy signal
            rng.gen_f64()
        };

        // Combine with current price based on agent type
        let estimate = match self.agent_type {
            AgentType::Informed => {
                // Rely more on private signal
                0.7 * private_signal + 0.3 * current_price
            }
            AgentType::Noise => {
                // Random with slight price anchoring
                0.3 * rng.gen_f64() + 0.7 * current_price
            }
            AgentType::Contrarian => {
                // Bet against the price
                1.0 - current_price + rng.gen_range_f64(-0.1, 0.1)
            }
            AgentType::Herder => {
                // Follow the price closely
                current_price + rng.gen_range_f64(-0.05, 0.05)
            }
            AgentType::Expert => {
                // Highly weighted on information
                0.9 * private_signal + 0.1 * current_price
            }
            AgentType::Novice | AgentType::Manipulator => {
                0.5 * private_signal + 0.5 * current_price
            }
        };

        // Apply calibration error (push toward extremes or center)
        let miscalibrated = if self.calibration_error > 0.0 {
            // Overconfidence: push away from 0.5
            let direction = if estimate > 0.5 { 1.0 } else { -1.0 };
            estimate + direction * self.calibration_error * (estimate - 0.5).abs()
        } else {
            estimate
        };

        let final_prob = miscalibrated.clamp(0.01, 0.99);

        SimulatedPrediction {
            agent_id: self.id.clone(),
            market_id: market.id.clone(),
            predicted_probability: final_prob,
            timestamp: 0, // Will be set by simulator
            confidence: (final_prob - 0.5).abs() * 2.0,
            reasoning_quality: self.information_quality,
        }
    }

    /// Update agent after receiving feedback
    pub fn receive_feedback(&mut self, outcome: bool, predicted: f64) {
        let error = if outcome {
            (1.0 - predicted).powi(2)
        } else {
            predicted.powi(2)
        };

        self.learning_state.feedback_count += 1;

        // Update calibration estimate (exponential moving average)
        self.learning_state.estimated_calibration =
            0.95 * self.learning_state.estimated_calibration + 0.05 * error.sqrt();

        // Learning effect: slight improvement over time
        let learning_rate = 0.001;
        self.calibration_error =
            (self.calibration_error - learning_rate * self.calibration_error).max(0.0);

        self.learning_state
            .calibration_history
            .push(self.calibration_error);
    }
}

/// Simple RNG trait for testability
pub trait Rng {
    /// Generate a random f64 in [0, 1)
    fn gen_f64(&mut self) -> f64;

    /// Generate a random f64 in [low, high)
    fn gen_range_f64(&mut self, low: f64, high: f64) -> f64;
}

/// Simple pseudo-random number generator
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // Linear congruential generator
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
}

impl Rng for SimpleRng {
    fn gen_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }

    fn gen_range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + self.gen_f64() * (high - low)
    }
}

// ============================================================================
// SIMULATED MARKETS
// ============================================================================

/// A simulated market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedMarket {
    pub id: String,
    pub domain: String,
    /// True probability of the positive outcome (hidden from agents)
    pub true_probability: f64,
    /// Current market probability
    pub current_probability: f64,
    /// LMSR quantities
    pub yes_quantity: f64,
    pub no_quantity: f64,
    /// Market history
    pub price_history: Vec<PricePoint>,
    /// Predictions made in this market
    pub predictions: Vec<SimulatedPrediction>,
    /// Market state
    pub state: MarketState,
    /// Resolution
    pub outcome: Option<bool>,
    /// Information events
    pub information_events: Vec<InformationEvent>,
    /// Liquidity parameter
    pub liquidity_parameter: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MarketState {
    Open,
    Closed,
    Resolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub step: usize,
    pub probability: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedPrediction {
    pub agent_id: String,
    pub market_id: String,
    pub predicted_probability: f64,
    pub timestamp: usize,
    pub confidence: f64,
    pub reasoning_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationEvent {
    pub step: usize,
    pub impact: f64,
    pub direction: f64,
    pub public: bool,
}

impl SimulatedMarket {
    /// Create a new simulated market
    pub fn new(id: String, domain: String, true_prob: f64, liquidity: f64) -> Self {
        Self {
            id,
            domain,
            true_probability: true_prob,
            current_probability: 0.5, // Start at even odds
            yes_quantity: 0.0,
            no_quantity: 0.0,
            price_history: vec![PricePoint {
                step: 0,
                probability: 0.5,
                volume: 0.0,
            }],
            predictions: vec![],
            state: MarketState::Open,
            outcome: None,
            information_events: vec![],
            liquidity_parameter: liquidity,
        }
    }

    /// Process a trade and update market price
    pub fn process_trade(&mut self, prediction: SimulatedPrediction, step: usize) {
        let mut pred = prediction;
        pred.timestamp = step;

        // Update LMSR quantities based on prediction direction
        let bet_size = 10.0; // Fixed bet size for simplicity
        if pred.predicted_probability > self.current_probability {
            self.yes_quantity += bet_size;
        } else {
            self.no_quantity += bet_size;
        }

        // Calculate new LMSR price
        let b = self.liquidity_parameter;
        let exp_yes = (self.yes_quantity / b).exp();
        let exp_no = (self.no_quantity / b).exp();
        self.current_probability = exp_yes / (exp_yes + exp_no);

        self.price_history.push(PricePoint {
            step,
            probability: self.current_probability,
            volume: bet_size,
        });

        self.predictions.push(pred);
    }

    /// Add an information event
    pub fn add_information_event(&mut self, step: usize, impact: f64, direction: f64, public: bool) {
        self.information_events.push(InformationEvent {
            step,
            impact,
            direction,
            public,
        });

        // Shift true probability slightly
        self.true_probability =
            (self.true_probability + impact * direction).clamp(0.01, 0.99);
    }

    /// Resolve the market
    pub fn resolve(&mut self, rng: &mut SimpleRng) {
        // Outcome determined by true probability
        self.outcome = Some((rng.next_u64() as f64 / u64::MAX as f64) < self.true_probability);
        self.state = MarketState::Resolved;
    }
}

// ============================================================================
// SIMULATION ENGINE
// ============================================================================

/// Main simulation engine
#[derive(Debug)]
pub struct SimulationEngine {
    pub config: SimulationConfig,
    pub agents: Vec<SimulatedAgent>,
    pub markets: Vec<SimulatedMarket>,
    pub current_step: usize,
    pub rng: SimpleRng,
    pub results: SimulationResults,
}

impl SimulationEngine {
    /// Create a new simulation engine
    pub fn new(config: SimulationConfig) -> Self {
        let rng = SimpleRng::new(config.seed);

        Self {
            config,
            agents: vec![],
            markets: vec![],
            current_step: 0,
            rng,
            results: SimulationResults::new(),
        }
    }

    /// Initialize agents based on configuration
    pub fn initialize_agents(&mut self) {
        for i in 0..self.config.num_agents {
            let agent_type = self.sample_agent_type();
            let calibration = self.sample_calibration();
            let info_quality = self.sample_information_quality();

            let mut agent = SimulatedAgent::new(
                format!("agent_{}", i),
                agent_type,
                calibration,
                info_quality,
            );

            // Assign to experiment group
            if !self.config.experiment_groups.is_empty() {
                agent.group = Some(self.assign_group());
            }

            self.agents.push(agent);
        }
    }

    fn sample_agent_type(&mut self) -> AgentType {
        let roll = self.rng.next_u64() as f64 / u64::MAX as f64;
        let mut cumulative = 0.0;

        for (agent_type, prob) in &self.config.agent_config.type_distribution {
            cumulative += prob;
            if roll < cumulative {
                return *agent_type;
            }
        }

        AgentType::Noise
    }

    fn sample_calibration(&mut self) -> f64 {
        let (mean, std) = self.config.agent_config.calibration_distribution;
        // Simple Box-Muller for normal distribution
        let u1 = self.rng.next_u64() as f64 / u64::MAX as f64;
        let u2 = self.rng.next_u64() as f64 / u64::MAX as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (mean + std * z).max(0.0)
    }

    fn sample_information_quality(&mut self) -> f64 {
        let (mean, std) = self.config.agent_config.information_distribution;
        let u1 = self.rng.next_u64() as f64 / u64::MAX as f64;
        let u2 = self.rng.next_u64() as f64 / u64::MAX as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (mean + std * z).clamp(0.0, 1.0)
    }

    fn assign_group(&mut self) -> String {
        let roll = self.rng.next_u64() as f64 / u64::MAX as f64;
        let mut cumulative = 0.0;

        for group in &self.config.experiment_groups {
            cumulative += group.assignment_ratio;
            if roll < cumulative {
                return group.name.clone();
            }
        }

        self.config
            .experiment_groups
            .last()
            .map(|g| g.name.clone())
            .unwrap_or_else(|| "control".to_string())
    }

    /// Initialize markets
    pub fn initialize_markets(&mut self) {
        for i in 0..self.config.num_markets {
            let (mean, std) = self.config.market_config.base_rate_distribution;
            let u1 = self.rng.next_u64() as f64 / u64::MAX as f64;
            let u2 = self.rng.next_u64() as f64 / u64::MAX as f64;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let true_prob = (mean + std * z).clamp(0.1, 0.9);

            let market = SimulatedMarket::new(
                format!("market_{}", i),
                "general".to_string(),
                true_prob,
                self.config.market_config.liquidity_parameter,
            );

            self.markets.push(market);
        }
    }

    /// Run one simulation step
    pub fn step(&mut self) {
        self.current_step += 1;

        // Add information events
        self.generate_information_events();

        // Each agent may make a prediction
        for agent_idx in 0..self.agents.len() {
            // Select a random open market
            let open_markets: Vec<usize> = self
                .markets
                .iter()
                .enumerate()
                .filter(|(_, m)| m.state == MarketState::Open)
                .map(|(i, _)| i)
                .collect();

            if open_markets.is_empty() {
                continue;
            }

            let market_idx = open_markets[self.rng.next_u64() as usize % open_markets.len()];

            // Decide whether to participate (based on agent type and market state)
            let participate_prob = 0.1; // 10% chance per step
            if self.rng.next_u64() as f64 / u64::MAX as f64 > participate_prob {
                continue;
            }

            let current_price = self.markets[market_idx].current_probability;
            let prediction = self.agents[agent_idx].make_prediction(
                &self.markets[market_idx],
                current_price,
                &mut self.rng,
            );

            self.markets[market_idx].process_trade(prediction, self.current_step);
        }

        // Check for market resolution
        self.check_market_resolution();

        // Collect data
        if self.current_step % self.config.collection_config.snapshot_interval == 0 {
            self.collect_snapshot();
        }
    }

    fn generate_information_events(&mut self) {
        for market in &mut self.markets {
            if market.state != MarketState::Open {
                continue;
            }

            if (self.rng.next_u64() as f64 / u64::MAX as f64)
                < self.config.market_config.information_rate
            {
                let impact = 0.1;
                let direction = if self.rng.next_u64() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                market.add_information_event(self.current_step, impact, direction, true);
            }
        }
    }

    fn check_market_resolution(&mut self) {
        for market in &mut self.markets {
            if market.state != MarketState::Open {
                continue;
            }

            // Resolve after duration
            if market.price_history.len() >= self.config.market_config.market_duration {
                market.resolve(&mut self.rng);

                // Provide feedback to agents
                if let Some(outcome) = market.outcome {
                    for pred in &market.predictions {
                        if let Some(agent) = self.agents.iter_mut().find(|a| a.id == pred.agent_id)
                        {
                            agent.receive_feedback(outcome, pred.predicted_probability);
                        }
                    }
                }
            }
        }
    }

    fn collect_snapshot(&mut self) {
        let snapshot = SimulationSnapshot {
            step: self.current_step,
            market_prices: self
                .markets
                .iter()
                .map(|m| (m.id.clone(), m.current_probability))
                .collect(),
            agent_calibrations: self
                .agents
                .iter()
                .map(|a| (a.id.clone(), a.calibration_error))
                .collect(),
            total_predictions: self.markets.iter().map(|m| m.predictions.len()).sum(),
        };

        self.results.snapshots.push(snapshot);
    }

    /// Run the full simulation
    pub fn run(&mut self) -> &SimulationResults {
        self.initialize_agents();
        self.initialize_markets();

        for _ in 0..self.config.num_steps {
            self.step();
        }

        self.finalize_results();
        &self.results
    }

    fn finalize_results(&mut self) {
        // Calculate final metrics
        self.results.simulation_id = self.config.simulation_id.clone();
        self.results.total_steps = self.current_step;
        self.results.total_agents = self.agents.len();
        self.results.total_markets = self.markets.len();

        // Collect all predictions for analysis
        for market in &self.markets {
            if let Some(outcome) = market.outcome {
                for pred in &market.predictions {
                    self.results.all_predictions.push((
                        pred.predicted_probability,
                        if outcome { 1.0 } else { 0.0 },
                    ));
                }
            }
        }

        // Group-specific results
        for group in &self.config.experiment_groups {
            let group_preds: Vec<(f64, f64)> = self
                .agents
                .iter()
                .filter(|a| a.group.as_ref() == Some(&group.name))
                .flat_map(|a| &a.prediction_history)
                .filter_map(|p| {
                    let market = self.markets.iter().find(|m| m.id == p.market_id)?;
                    let outcome = market.outcome?;
                    Some((p.predicted_probability, if outcome { 1.0 } else { 0.0 }))
                })
                .collect();

            self.results
                .group_predictions
                .insert(group.name.clone(), group_preds);
        }
    }
}

// ============================================================================
// SIMULATION RESULTS
// ============================================================================

/// Results from a simulation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResults {
    pub simulation_id: String,
    pub total_steps: usize,
    pub total_agents: usize,
    pub total_markets: usize,

    /// All predictions with outcomes
    pub all_predictions: Vec<(f64, f64)>,

    /// Per-group predictions
    pub group_predictions: HashMap<String, Vec<(f64, f64)>>,

    /// Time series snapshots
    pub snapshots: Vec<SimulationSnapshot>,

    /// Final agent states
    pub final_agent_calibrations: HashMap<String, f64>,
}

impl SimulationResults {
    pub fn new() -> Self {
        Self {
            simulation_id: String::new(),
            total_steps: 0,
            total_agents: 0,
            total_markets: 0,
            all_predictions: vec![],
            group_predictions: HashMap::new(),
            snapshots: vec![],
            final_agent_calibrations: HashMap::new(),
        }
    }

    /// Export results to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

impl Default for SimulationResults {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSnapshot {
    pub step: usize,
    pub market_prices: HashMap<String, f64>,
    pub agent_calibrations: HashMap<String, f64>,
    pub total_predictions: usize,
}

// ============================================================================
// A/B TEST ANALYSIS
// ============================================================================

/// Analysis of A/B test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestAnalysis {
    pub experiment_id: String,
    pub groups: Vec<GroupAnalysis>,
    pub comparisons: Vec<GroupComparison>,
    pub winner: Option<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupAnalysis {
    pub name: String,
    pub sample_size: usize,
    pub mean_brier: f64,
    pub std_brier: f64,
    pub mean_calibration_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupComparison {
    pub group_a: String,
    pub group_b: String,
    pub brier_difference: f64,
    pub p_value: f64,
    pub significant: bool,
    pub effect_size: f64,
}

impl ABTestAnalysis {
    /// Analyze A/B test results
    pub fn analyze(results: &SimulationResults) -> Self {
        let mut groups = Vec::new();
        let mut comparisons = Vec::new();

        for (name, predictions) in &results.group_predictions {
            if predictions.is_empty() {
                continue;
            }

            let brier_scores: Vec<f64> = predictions
                .iter()
                .map(|(p, o)| (p - o).powi(2))
                .collect();

            let mean_brier = brier_scores.iter().sum::<f64>() / brier_scores.len() as f64;
            let variance: f64 = brier_scores
                .iter()
                .map(|b| (b - mean_brier).powi(2))
                .sum::<f64>()
                / (brier_scores.len() - 1).max(1) as f64;

            groups.push(GroupAnalysis {
                name: name.clone(),
                sample_size: predictions.len(),
                mean_brier,
                std_brier: variance.sqrt(),
                mean_calibration_error: mean_brier.sqrt(), // Simplified
            });
        }

        // Pairwise comparisons
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                let g_a = &groups[i];
                let g_b = &groups[j];

                let diff = g_a.mean_brier - g_b.mean_brier;
                let pooled_std = ((g_a.std_brier.powi(2) / g_a.sample_size as f64)
                    + (g_b.std_brier.powi(2) / g_b.sample_size as f64))
                .sqrt();

                let t_stat = if pooled_std > 0.0 {
                    diff / pooled_std
                } else {
                    0.0
                };

                // Simplified p-value estimation
                let p_value = 2.0 * (1.0 - 0.5 * (1.0 + (t_stat.abs() / 1.5).tanh()));

                comparisons.push(GroupComparison {
                    group_a: g_a.name.clone(),
                    group_b: g_b.name.clone(),
                    brier_difference: diff,
                    p_value,
                    significant: p_value < 0.05,
                    effect_size: diff / g_a.std_brier.max(0.001),
                });
            }
        }

        // Determine winner
        let winner = groups
            .iter()
            .min_by(|a, b| {
                a.mean_brier
                    .partial_cmp(&b.mean_brier)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|g| g.name.clone());

        let confidence = if let Some(ref w) = winner {
            comparisons
                .iter()
                .filter(|c| c.group_a == *w || c.group_b == *w)
                .map(|c| 1.0 - c.p_value)
                .fold(1.0, f64::min)
        } else {
            0.0
        };

        Self {
            experiment_id: results.simulation_id.clone(),
            groups,
            comparisons,
            winner,
            confidence,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_engine_creation() {
        let config = SimulationConfig::default();
        let engine = SimulationEngine::new(config);
        assert_eq!(engine.current_step, 0);
    }

    #[test]
    fn test_agent_creation() {
        let agent = SimulatedAgent::new(
            "test_agent".to_string(),
            AgentType::Informed,
            0.1,
            0.8,
        );
        assert_eq!(agent.agent_type, AgentType::Informed);
        assert_eq!(agent.calibration_error, 0.1);
    }

    #[test]
    fn test_market_creation() {
        let market = SimulatedMarket::new(
            "test_market".to_string(),
            "politics".to_string(),
            0.7,
            100.0,
        );
        assert_eq!(market.true_probability, 0.7);
        assert_eq!(market.current_probability, 0.5);
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_simulation_short_run() {
        let config = SimulationConfig {
            num_agents: 10,
            num_markets: 2,
            num_steps: 20,
            ..Default::default()
        };

        let mut engine = SimulationEngine::new(config);
        let results = engine.run();

        assert!(results.total_steps > 0);
        assert_eq!(results.total_agents, 10);
    }
}
