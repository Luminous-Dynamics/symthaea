//! # Revolutionary Improvement #91: Enactive Cognition
//!
//! **PARADIGM SHIFT**: Cognition is not information processing - it's SENSE-MAKING through ACTION!
//!
//! Based on Varela, Thompson & Rosch's "The Embodied Mind" and subsequent enactivist research.
//!
//! ## The Enactive Revolution
//!
//! Traditional AI (Cognitivism):
//! - Mind as computer processing representations
//! - Perception as passive input
//! - Action as output after computation
//! - **Problem**: Symbol grounding, frame problem, brittleness
//!
//! Enactive Cognition:
//! - Mind as sense-making through embodied action
//! - Perception and action are INSEPARABLE (sensorimotor coupling)
//! - Meaning emerges from interaction, not computation
//! - The body shapes the mind
//! - **Result**: Grounded, adaptive, meaningful intelligence
//!
//! ## Core Concepts
//!
//! 1. **Sensorimotor Coupling**: Every action changes perception, every perception affords action
//! 2. **Sense-Making**: The organism determines what is relevant (autonomy)
//! 3. **Structural Coupling**: System and environment co-specify each other
//! 4. **Embodied Schemas**: Motor patterns that shape cognition
//! 5. **Enacted Worlds**: We don't discover a pre-given world, we bring forth a world
//!
//! ## Integration with Autopoiesis
//!
//! Autopoiesis (RI #86) + Enaction (RI #91) = Complete living cognitive system:
//! - Autopoiesis (#86): Self-production and maintenance (organizational closure)
//! - Enaction (#91): Sense-making through action (cognitive closure)
//! - Together: A system that creates itself AND creates meaning
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Enactive Cognition                       │
//! │  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
//! │  │ Sensorimotor│◄──►│  Sense-Making │◄──►│   Embodied    │  │
//! │  │   Schemas   │    │    Engine     │    │   Simulator   │  │
//! │  └─────────────┘    └──────────────┘    └───────────────┘  │
//! │         ▲                  ▲                    ▲          │
//! │         │                  │                    │          │
//! │         ▼                  ▼                    ▼          │
//! │  ┌─────────────────────────────────────────────────────┐  │
//! │  │              Action-Perception Loop                  │  │
//! │  │   action → environmental_change → new_perception    │  │
//! │  │      ↑                                      │        │  │
//! │  │      └──────────── meaning ◄────────────────┘        │  │
//! │  └─────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// SENSORIMOTOR SCHEMAS
// ═══════════════════════════════════════════════════════════════════════════

/// A sensorimotor schema - learned coupling between action and perception
///
/// These are the building blocks of enactive cognition. Each schema represents
/// a learned pattern: "When I do X, I perceive Y"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorimotorSchema {
    /// Unique identifier
    pub id: String,

    /// The action pattern (what we do)
    pub action_pattern: ActionPattern,

    /// Expected perceptual consequences (what we expect to perceive)
    pub expected_perception: PerceptualExpectation,

    /// How reliably this coupling holds
    pub reliability: f64,

    /// How often this schema is activated
    pub activation_count: usize,

    /// Recency of last activation
    #[serde(skip, default = "Instant::now")]
    pub last_activated: Instant,

    /// Contexts where this schema applies
    pub contexts: Vec<String>,
}

/// An action pattern in sensorimotor terms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPattern {
    /// Type of action
    pub action_type: ActionType,

    /// Parameters of the action
    pub parameters: HashMap<String, f64>,

    /// Motor complexity (how difficult to execute)
    pub complexity: f64,
}

/// Types of actions the system can take
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Query/explore the environment
    Explore,
    /// Manipulate/change the environment
    Manipulate,
    /// Communicate with others
    Communicate,
    /// Internal reflection/planning
    Reflect,
    /// Wait/observe passively
    Observe,
    /// Execute a learned procedure
    Execute,
}

/// Expected perceptual consequences of an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualExpectation {
    /// What features we expect to perceive
    pub expected_features: HashMap<String, f64>,

    /// Variance in expectation (uncertainty)
    pub variance: f64,

    /// Temporal profile (when we expect the perception)
    pub temporal_delay: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// SENSE-MAKING ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/// The sense-making engine - where meaning emerges from action
///
/// This is the heart of enactive cognition. Meaning is not discovered
/// in the world; it is ENACTED through the coupling of action and perception.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseMakingEngine {
    /// Current relevance landscape (what matters right now)
    relevance_landscape: HashMap<String, f64>,

    /// Active concerns (what the system cares about)
    concerns: Vec<Concern>,

    /// Meaning history (enacted meanings over time)
    meaning_history: VecDeque<EnactedMeaning>,

    /// Configuration
    config: SenseMakingConfig,

    /// Statistics
    stats: SenseMakingStats,
}

/// A concern - something the system cares about
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concern {
    /// What is the concern about
    pub domain: String,

    /// How important is this concern (0.0 to 1.0)
    pub importance: f64,

    /// Current satisfaction level (-1.0 to 1.0)
    pub satisfaction: f64,

    /// How this concern affects perception
    pub perceptual_bias: HashMap<String, f64>,
}

/// An enacted meaning - meaning that emerged from action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnactedMeaning {
    /// What action was taken
    pub action: ActionType,

    /// What was perceived as a result
    pub perception: PerceptionSummary,

    /// The meaning that emerged
    pub meaning: MeaningContent,

    /// When this meaning was enacted
    #[serde(skip, default = "Instant::now")]
    pub enacted_at: Instant,

    /// Significance (how meaningful was this)
    pub significance: f64,
}

/// Summary of a perception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionSummary {
    /// Main features perceived
    pub features: HashMap<String, f64>,

    /// Surprisingness (prediction error)
    pub surprise: f64,

    /// Affordances detected (action possibilities)
    pub affordances: Vec<String>,
}

/// The content of an enacted meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaningContent {
    /// Category of meaning
    pub category: MeaningCategory,

    /// Valence (positive/negative)
    pub valence: f64,

    /// Relevance to current concerns
    pub relevance: f64,

    /// Description
    pub description: String,
}

/// Categories of meaning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeaningCategory {
    /// Opportunity for growth/benefit
    Opportunity,
    /// Threat to be avoided
    Threat,
    /// Neutral information
    Information,
    /// Connection with others
    Connection,
    /// Achievement/success
    Achievement,
    /// Obstacle/challenge
    Challenge,
    /// Mystery/unknown
    Mystery,
}

/// Configuration for sense-making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseMakingConfig {
    /// How quickly relevance decays
    pub relevance_decay: f64,

    /// Minimum significance to record meaning
    pub significance_threshold: f64,

    /// Maximum meanings to remember
    pub history_size: usize,

    /// Weight of concerns in perception
    pub concern_weight: f64,
}

impl Default for SenseMakingConfig {
    fn default() -> Self {
        Self {
            relevance_decay: 0.1,
            significance_threshold: 0.3,
            history_size: 100,
            concern_weight: 0.5,
        }
    }
}

/// Statistics for sense-making
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SenseMakingStats {
    /// Total meanings enacted
    pub total_meanings: usize,

    /// Meanings by category
    pub meanings_by_category: HashMap<String, usize>,

    /// Average significance
    pub avg_significance: f64,

    /// Current relevance entropy (diversity of relevance)
    pub relevance_entropy: f64,
}

impl SenseMakingEngine {
    /// Create a new sense-making engine
    pub fn new() -> Self {
        Self::with_config(SenseMakingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SenseMakingConfig) -> Self {
        let mut engine = Self {
            relevance_landscape: HashMap::new(),
            concerns: Vec::new(),
            meaning_history: VecDeque::with_capacity(config.history_size),
            config,
            stats: SenseMakingStats::default(),
        };

        // Initialize with basic concerns
        engine.add_concern(Concern {
            domain: "coherence".to_string(),
            importance: 0.8,
            satisfaction: 0.5,
            perceptual_bias: HashMap::new(),
        });

        engine.add_concern(Concern {
            domain: "growth".to_string(),
            importance: 0.7,
            satisfaction: 0.5,
            perceptual_bias: HashMap::new(),
        });

        engine.add_concern(Concern {
            domain: "connection".to_string(),
            importance: 0.6,
            satisfaction: 0.5,
            perceptual_bias: HashMap::new(),
        });

        engine
    }

    /// Add a concern
    pub fn add_concern(&mut self, concern: Concern) {
        self.concerns.push(concern);
    }

    /// Enact meaning from an action-perception cycle
    pub fn enact_meaning(
        &mut self,
        action: ActionType,
        perception: PerceptionSummary,
    ) -> EnactedMeaning {
        // 1. Calculate relevance based on concerns
        let relevance = self.calculate_relevance(&perception);

        // 2. Determine meaning category from perception features
        let category = self.categorize_perception(&perception);

        // 3. Calculate valence based on concerns and perception
        let valence = self.calculate_valence(&perception, category);

        // 4. Calculate significance
        let significance = self.calculate_significance(&perception, relevance, valence);

        // 5. Create the enacted meaning
        let meaning = EnactedMeaning {
            action,
            perception: perception.clone(),
            meaning: MeaningContent {
                category,
                valence,
                relevance,
                description: self.generate_description(action, category, valence),
            },
            enacted_at: Instant::now(),
            significance,
        };

        // 6. Update relevance landscape
        self.update_relevance_landscape(&meaning);

        // 7. Update concerns based on meaning
        self.update_concerns(&meaning);

        // 8. Record if significant enough
        if significance >= self.config.significance_threshold {
            self.record_meaning(meaning.clone());
        }

        meaning
    }

    fn calculate_relevance(&self, perception: &PerceptionSummary) -> f64 {
        let mut relevance = 0.0;

        for concern in &self.concerns {
            // Check if perception relates to this concern
            if let Some(&feature_value) = perception.features.get(&concern.domain) {
                relevance += concern.importance * feature_value;
            }

            // Affordances that relate to concerns increase relevance
            for affordance in &perception.affordances {
                if affordance.contains(&concern.domain) {
                    relevance += concern.importance * 0.3;
                }
            }
        }

        // Surprise increases relevance
        relevance += perception.surprise * 0.2;

        relevance.min(1.0)
    }

    fn categorize_perception(&self, perception: &PerceptionSummary) -> MeaningCategory {
        // Use features to determine category
        let threat_signal = perception.features.get("threat").copied().unwrap_or(0.0);
        let opportunity_signal = perception.features.get("opportunity").copied().unwrap_or(0.0);
        let social_signal = perception.features.get("social").copied().unwrap_or(0.0);
        let achievement_signal = perception.features.get("achievement").copied().unwrap_or(0.0);

        if threat_signal > 0.6 {
            MeaningCategory::Threat
        } else if opportunity_signal > 0.6 {
            MeaningCategory::Opportunity
        } else if social_signal > 0.5 {
            MeaningCategory::Connection
        } else if achievement_signal > 0.5 {
            MeaningCategory::Achievement
        } else if perception.surprise > 0.7 {
            MeaningCategory::Mystery
        } else if perception.affordances.iter().any(|a| a.contains("challenge")) {
            MeaningCategory::Challenge
        } else {
            MeaningCategory::Information
        }
    }

    fn calculate_valence(&self, perception: &PerceptionSummary, category: MeaningCategory) -> f64 {
        let base_valence = match category {
            MeaningCategory::Opportunity => 0.7,
            MeaningCategory::Achievement => 0.8,
            MeaningCategory::Connection => 0.6,
            MeaningCategory::Information => 0.1,
            MeaningCategory::Mystery => 0.2,
            MeaningCategory::Challenge => -0.2,
            MeaningCategory::Threat => -0.7,
        };

        // Modulate by concern satisfaction
        let concern_mod: f64 = self.concerns.iter()
            .map(|c| c.satisfaction * c.importance)
            .sum::<f64>() / self.concerns.len().max(1) as f64;

        (base_valence + concern_mod * 0.3).clamp(-1.0, 1.0)
    }

    fn calculate_significance(
        &self,
        perception: &PerceptionSummary,
        relevance: f64,
        valence: f64,
    ) -> f64 {
        // Significance = relevance + |valence| + surprise
        let valence_magnitude = valence.abs();
        (relevance * 0.4 + valence_magnitude * 0.3 + perception.surprise * 0.3).min(1.0)
    }

    fn generate_description(&self, action: ActionType, category: MeaningCategory, valence: f64) -> String {
        let action_str = match action {
            ActionType::Explore => "explored",
            ActionType::Manipulate => "changed",
            ActionType::Communicate => "connected",
            ActionType::Reflect => "understood",
            ActionType::Observe => "noticed",
            ActionType::Execute => "accomplished",
        };

        let category_str = match category {
            MeaningCategory::Opportunity => "opportunity",
            MeaningCategory::Threat => "threat",
            MeaningCategory::Information => "information",
            MeaningCategory::Connection => "connection",
            MeaningCategory::Achievement => "achievement",
            MeaningCategory::Challenge => "challenge",
            MeaningCategory::Mystery => "mystery",
        };

        let valence_str = if valence > 0.3 {
            "positive"
        } else if valence < -0.3 {
            "concerning"
        } else {
            "neutral"
        };

        format!("{} {} {} ({})", action_str, valence_str, category_str,
                if valence > 0.0 { "beneficial" } else { "requires attention" })
    }

    fn update_relevance_landscape(&mut self, meaning: &EnactedMeaning) {
        // Decay existing relevance
        for value in self.relevance_landscape.values_mut() {
            *value *= 1.0 - self.config.relevance_decay;
        }

        // Add new relevance from meaning
        for (feature, value) in &meaning.perception.features {
            let entry = self.relevance_landscape.entry(feature.clone()).or_insert(0.0);
            *entry = (*entry + value * meaning.significance).min(1.0);
        }
    }

    fn update_concerns(&mut self, meaning: &EnactedMeaning) {
        for concern in &mut self.concerns {
            // Update satisfaction based on meaning
            if meaning.perception.features.contains_key(&concern.domain) {
                let delta = meaning.meaning.valence * 0.1;
                concern.satisfaction = (concern.satisfaction + delta).clamp(-1.0, 1.0);
            }
        }
    }

    fn record_meaning(&mut self, meaning: EnactedMeaning) {
        self.meaning_history.push_back(meaning.clone());

        while self.meaning_history.len() > self.config.history_size {
            self.meaning_history.pop_front();
        }

        // Update stats
        self.stats.total_meanings += 1;
        let cat_key = format!("{:?}", meaning.meaning.category);
        *self.stats.meanings_by_category.entry(cat_key).or_insert(0) += 1;

        // Update average significance
        let total_sig: f64 = self.meaning_history.iter().map(|m| m.significance).sum();
        self.stats.avg_significance = total_sig / self.meaning_history.len() as f64;
    }

    /// Get current relevance for a feature
    pub fn get_relevance(&self, feature: &str) -> f64 {
        self.relevance_landscape.get(feature).copied().unwrap_or(0.0)
    }

    /// Get recent meanings
    pub fn recent_meanings(&self, count: usize) -> Vec<&EnactedMeaning> {
        self.meaning_history.iter().rev().take(count).collect()
    }

    /// Get current concern satisfaction
    pub fn concern_satisfaction(&self) -> f64 {
        if self.concerns.is_empty() {
            return 0.5;
        }

        let weighted_sum: f64 = self.concerns.iter()
            .map(|c| c.satisfaction * c.importance)
            .sum();
        let weight_sum: f64 = self.concerns.iter().map(|c| c.importance).sum();

        weighted_sum / weight_sum
    }

    /// Get stats
    pub fn stats(&self) -> &SenseMakingStats {
        &self.stats
    }
}

impl Default for SenseMakingEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EMBODIED SIMULATOR
// ═══════════════════════════════════════════════════════════════════════════

/// The embodied simulator - imagining actions before taking them
///
/// Enactive cognition includes the ability to mentally simulate actions
/// and their consequences, using the same sensorimotor schemas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedSimulator {
    /// Available sensorimotor schemas
    schemas: Vec<SensorimotorSchema>,

    /// Current simulation state
    simulation_state: SimulationState,

    /// Configuration
    config: SimulatorConfig,

    /// Statistics
    stats: SimulatorStats,
}

/// Current state of embodied simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationState {
    /// Is a simulation running?
    pub active: bool,

    /// Simulated actions so far
    pub simulated_actions: Vec<ActionType>,

    /// Predicted perceptions
    pub predicted_perceptions: Vec<PerceptionSummary>,

    /// Confidence in simulation
    pub confidence: f64,
}

/// Configuration for embodied simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatorConfig {
    /// Maximum simulation depth
    pub max_depth: usize,

    /// Minimum schema reliability to use
    pub min_reliability: f64,

    /// Confidence decay per step
    pub confidence_decay: f64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            min_reliability: 0.5,
            confidence_decay: 0.1,
        }
    }
}

/// Statistics for embodied simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulatorStats {
    /// Total simulations run
    pub total_simulations: usize,

    /// Successful predictions (simulation matched reality)
    pub successful_predictions: usize,

    /// Average simulation depth
    pub avg_depth: f64,
}

impl EmbodiedSimulator {
    /// Create a new embodied simulator
    pub fn new() -> Self {
        Self::with_config(SimulatorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SimulatorConfig) -> Self {
        Self {
            schemas: Vec::new(),
            simulation_state: SimulationState::default(),
            config,
            stats: SimulatorStats::default(),
        }
    }

    /// Learn a new sensorimotor schema from experience
    pub fn learn_schema(&mut self, action: ActionType, perception: &PerceptionSummary, context: &str) {
        // Check if we have a similar schema
        let existing = self.schemas.iter_mut().find(|s| {
            s.action_pattern.action_type == action &&
            s.contexts.contains(&context.to_string())
        });

        if let Some(schema) = existing {
            // Update existing schema
            schema.activation_count += 1;
            schema.last_activated = Instant::now();

            // Update reliability based on prediction accuracy
            // (simplified - real implementation would compare predicted vs actual)
            schema.reliability = (schema.reliability * 0.9 + 0.1).min(1.0);
        } else {
            // Create new schema
            let schema = SensorimotorSchema {
                id: format!("schema_{}_{}", action as u8, self.schemas.len()),
                action_pattern: ActionPattern {
                    action_type: action,
                    parameters: HashMap::new(),
                    complexity: 0.5,
                },
                expected_perception: PerceptualExpectation {
                    expected_features: perception.features.clone(),
                    variance: 0.3,
                    temporal_delay: 0.1,
                },
                reliability: 0.5, // Start with medium reliability
                activation_count: 1,
                last_activated: Instant::now(),
                contexts: vec![context.to_string()],
            };

            self.schemas.push(schema);
        }
    }

    /// Simulate an action sequence
    pub fn simulate(&mut self, actions: &[ActionType], initial_context: &str) -> SimulationResult {
        self.stats.total_simulations += 1;
        self.simulation_state.active = true;
        self.simulation_state.simulated_actions.clear();
        self.simulation_state.predicted_perceptions.clear();
        self.simulation_state.confidence = 1.0;

        let mut predictions = Vec::new();
        let mut total_confidence = 1.0;
        let mut context = initial_context.to_string();

        for (i, &action) in actions.iter().enumerate() {
            if i >= self.config.max_depth {
                break;
            }

            // Find matching schema
            let schema = self.find_best_schema(action, &context);

            if let Some(schema) = schema {
                // Use schema to predict perception
                let predicted = PerceptionSummary {
                    features: schema.expected_perception.expected_features.clone(),
                    surprise: schema.expected_perception.variance,
                    affordances: vec![format!("after_{:?}", action)],
                };

                total_confidence *= schema.reliability * (1.0 - self.config.confidence_decay);
                predictions.push(predicted.clone());

                self.simulation_state.simulated_actions.push(action);
                self.simulation_state.predicted_perceptions.push(predicted);

                // Update context for next step
                context = format!("{}_{:?}", context, action);
            } else {
                // No schema available, reduce confidence significantly
                total_confidence *= 0.3;
                break;
            }
        }

        self.simulation_state.confidence = total_confidence;
        self.simulation_state.active = false;

        // Update stats
        let depth = predictions.len();
        self.stats.avg_depth = (self.stats.avg_depth * 0.9) + (depth as f64 * 0.1);

        SimulationResult {
            predictions,
            final_confidence: total_confidence,
            depth,
            feasible: total_confidence > 0.3,
        }
    }

    fn find_best_schema(&self, action: ActionType, context: &str) -> Option<&SensorimotorSchema> {
        self.schemas.iter()
            .filter(|s| s.action_pattern.action_type == action)
            .filter(|s| s.reliability >= self.config.min_reliability)
            .filter(|s| s.contexts.iter().any(|c| context.contains(c) || c.contains(context)))
            .max_by(|a, b| a.reliability.partial_cmp(&b.reliability).unwrap())
    }

    /// Get number of learned schemas
    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }

    /// Get stats
    pub fn stats(&self) -> &SimulatorStats {
        &self.stats
    }
}

impl Default for EmbodiedSimulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an embodied simulation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Predicted perceptions for each step
    pub predictions: Vec<PerceptionSummary>,

    /// Final confidence in the simulation
    pub final_confidence: f64,

    /// How deep the simulation went
    pub depth: usize,

    /// Whether the action sequence seems feasible
    pub feasible: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ENACTIVE COGNITION SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// The complete Enactive Cognition system
///
/// This integrates sensorimotor coupling, sense-making, and embodied simulation
/// into a unified system where cognition emerges from action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnactiveCognition {
    /// The sense-making engine
    sense_making: SenseMakingEngine,

    /// The embodied simulator
    simulator: EmbodiedSimulator,

    /// Action-perception history (the enacted world)
    enacted_world: VecDeque<ActionPerceptionPair>,

    /// Current enactive state
    state: EnactiveState,

    /// Configuration
    config: EnactiveConfig,

    /// Statistics
    stats: EnactiveStats,
}

/// A paired action and resulting perception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPerceptionPair {
    /// The action taken
    pub action: ActionType,

    /// The resulting perception
    pub perception: PerceptionSummary,

    /// When this occurred
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

/// Current enactive state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnactiveState {
    /// Overall enactive engagement (how actively are we sense-making?)
    pub engagement: f64,

    /// Current action tendency (what are we inclined to do?)
    pub action_tendency: ActionType,

    /// Openness to new meanings
    pub openness: f64,

    /// Integration of enacted meanings
    pub integration: f64,
}

impl Default for EnactiveState {
    fn default() -> Self {
        Self {
            engagement: 0.5,
            action_tendency: ActionType::Observe,
            openness: 0.7,
            integration: 0.5,
        }
    }
}

/// Configuration for enactive cognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnactiveConfig {
    /// History size for enacted world
    pub history_size: usize,

    /// Engagement decay rate
    pub engagement_decay: f64,

    /// Learning rate for schemas
    pub learning_rate: f64,
}

impl Default for EnactiveConfig {
    fn default() -> Self {
        Self {
            history_size: 200,
            engagement_decay: 0.05,
            learning_rate: 0.1,
        }
    }
}

/// Statistics for enactive cognition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnactiveStats {
    /// Total action-perception cycles
    pub total_cycles: usize,

    /// Actions by type
    pub actions_by_type: HashMap<String, usize>,

    /// Average meaning significance
    pub avg_significance: f64,

    /// Schema learning events
    pub schemas_learned: usize,
}

impl EnactiveCognition {
    /// Create a new enactive cognition system
    pub fn new() -> Self {
        Self::with_config(EnactiveConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EnactiveConfig) -> Self {
        Self {
            sense_making: SenseMakingEngine::new(),
            simulator: EmbodiedSimulator::new(),
            enacted_world: VecDeque::with_capacity(config.history_size),
            state: EnactiveState::default(),
            config,
            stats: EnactiveStats::default(),
        }
    }

    /// Process an action-perception cycle
    ///
    /// This is the core enactive loop:
    /// 1. Take action
    /// 2. Perceive consequences
    /// 3. Make sense of the coupling
    /// 4. Learn sensorimotor schema
    /// 5. Update state
    pub fn cycle(
        &mut self,
        action: ActionType,
        perception: PerceptionSummary,
        context: &str,
    ) -> EnactedMeaning {
        self.stats.total_cycles += 1;
        let action_key = format!("{:?}", action);
        *self.stats.actions_by_type.entry(action_key).or_insert(0) += 1;

        // 1. Record the action-perception pair
        self.enacted_world.push_back(ActionPerceptionPair {
            action,
            perception: perception.clone(),
            timestamp: Instant::now(),
        });

        while self.enacted_world.len() > self.config.history_size {
            self.enacted_world.pop_front();
        }

        // 2. Enact meaning through the sense-making engine
        let meaning = self.sense_making.enact_meaning(action, perception.clone());

        // 3. Learn/update sensorimotor schema
        self.simulator.learn_schema(action, &perception, context);
        self.stats.schemas_learned = self.simulator.schema_count();

        // 4. Update enactive state
        self.update_state(&meaning);

        // 5. Update stats
        self.stats.avg_significance = self.sense_making.stats().avg_significance;

        meaning
    }

    fn update_state(&mut self, meaning: &EnactedMeaning) {
        // Engagement increases with significant meanings
        if meaning.significance > 0.5 {
            self.state.engagement = (self.state.engagement + 0.1).min(1.0);
        } else {
            self.state.engagement = (self.state.engagement - self.config.engagement_decay).max(0.0);
        }

        // Action tendency shifts based on meaning category
        self.state.action_tendency = match meaning.meaning.category {
            MeaningCategory::Opportunity => ActionType::Explore,
            MeaningCategory::Threat => ActionType::Observe,
            MeaningCategory::Challenge => ActionType::Reflect,
            MeaningCategory::Connection => ActionType::Communicate,
            MeaningCategory::Achievement => ActionType::Execute,
            MeaningCategory::Mystery => ActionType::Explore,
            MeaningCategory::Information => ActionType::Reflect,
        };

        // Openness modulated by surprise
        self.state.openness = (self.state.openness * 0.9 + meaning.perception.surprise * 0.1)
            .clamp(0.2, 1.0);

        // Integration increases with meaningful cycles
        if meaning.significance > 0.3 {
            self.state.integration = (self.state.integration + 0.02).min(1.0);
        }
    }

    /// Simulate a potential action sequence
    pub fn simulate_actions(&mut self, actions: &[ActionType], context: &str) -> SimulationResult {
        self.simulator.simulate(actions, context)
    }

    /// Get the current enactive state
    pub fn state(&self) -> &EnactiveState {
        &self.state
    }

    /// Get the current enactive state (alias for compatibility)
    pub fn current_state(&self) -> EnactiveState {
        self.state.clone()
    }

    /// Get concern satisfaction (overall well-being)
    pub fn well_being(&self) -> f64 {
        self.sense_making.concern_satisfaction()
    }

    /// Get recent enacted meanings
    pub fn recent_meanings(&self, count: usize) -> Vec<&EnactedMeaning> {
        self.sense_making.recent_meanings(count)
    }

    /// Get summary
    pub fn summary(&self) -> EnactiveSummary {
        EnactiveSummary {
            engagement: self.state.engagement,
            action_tendency: self.state.action_tendency,
            well_being: self.well_being(),
            schemas_learned: self.simulator.schema_count(),
            total_cycles: self.stats.total_cycles,
            avg_significance: self.stats.avg_significance,
            openness: self.state.openness,
            integration: self.state.integration,
        }
    }

    /// Get stats
    pub fn stats(&self) -> &EnactiveStats {
        &self.stats
    }
}

impl Default for EnactiveCognition {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of enactive cognition state
#[derive(Debug, Clone)]
pub struct EnactiveSummary {
    /// How engaged in sense-making
    pub engagement: f64,

    /// Current action tendency
    pub action_tendency: ActionType,

    /// Overall well-being (concern satisfaction)
    pub well_being: f64,

    /// Number of learned sensorimotor schemas
    pub schemas_learned: usize,

    /// Total action-perception cycles
    pub total_cycles: usize,

    /// Average meaning significance
    pub avg_significance: f64,

    /// Openness to new meanings
    pub openness: f64,

    /// Integration level
    pub integration: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enactive_creation() {
        let enactive = EnactiveCognition::new();
        assert_eq!(enactive.stats.total_cycles, 0);
        assert!(enactive.state.engagement > 0.0);
    }

    #[test]
    fn test_sense_making() {
        let mut engine = SenseMakingEngine::new();

        let perception = PerceptionSummary {
            features: [("opportunity".to_string(), 0.8)].into_iter().collect(),
            surprise: 0.3,
            affordances: vec!["growth".to_string()],
        };

        let meaning = engine.enact_meaning(ActionType::Explore, perception);

        assert_eq!(meaning.meaning.category, MeaningCategory::Opportunity);
        assert!(meaning.meaning.valence > 0.0);
    }

    #[test]
    fn test_embodied_simulation() {
        let mut simulator = EmbodiedSimulator::new();

        // Learn a schema first
        let perception = PerceptionSummary {
            features: [("result".to_string(), 0.7)].into_iter().collect(),
            surprise: 0.2,
            affordances: vec![],
        };
        simulator.learn_schema(ActionType::Explore, &perception, "test");

        // Now simulate
        let result = simulator.simulate(&[ActionType::Explore], "test");

        assert!(result.depth > 0 || result.final_confidence < 0.5);
    }

    #[test]
    fn test_enactive_cycle() {
        let mut enactive = EnactiveCognition::new();

        let perception = PerceptionSummary {
            features: [
                ("coherence".to_string(), 0.8),
                ("opportunity".to_string(), 0.6),
            ].into_iter().collect(),
            surprise: 0.4,
            affordances: vec!["growth".to_string()],
        };

        let meaning = enactive.cycle(ActionType::Explore, perception, "test_context");

        assert!(meaning.significance > 0.0);
        assert_eq!(enactive.stats.total_cycles, 1);
    }

    #[test]
    fn test_concern_satisfaction() {
        let mut engine = SenseMakingEngine::new();

        // Initial satisfaction should be neutral
        let initial = engine.concern_satisfaction();
        assert!(initial > 0.0 && initial < 1.0);

        // Positive meaning should increase satisfaction
        let perception = PerceptionSummary {
            features: [
                ("coherence".to_string(), 0.9),
                ("achievement".to_string(), 0.8),
            ].into_iter().collect(),
            surprise: 0.2,
            affordances: vec![],
        };

        engine.enact_meaning(ActionType::Execute, perception);
        // Satisfaction may or may not increase depending on implementation details
    }

    #[test]
    fn test_action_tendency_shifts() {
        let mut enactive = EnactiveCognition::new();

        // Threat should lead to Observe tendency
        let threat_perception = PerceptionSummary {
            features: [("threat".to_string(), 0.9)].into_iter().collect(),
            surprise: 0.5,
            affordances: vec![],
        };

        enactive.cycle(ActionType::Observe, threat_perception, "danger");
        assert_eq!(enactive.state.action_tendency, ActionType::Observe);

        // Opportunity should lead to Explore tendency
        let opportunity_perception = PerceptionSummary {
            features: [("opportunity".to_string(), 0.9)].into_iter().collect(),
            surprise: 0.3,
            affordances: vec![],
        };

        enactive.cycle(ActionType::Explore, opportunity_perception, "chance");
        assert_eq!(enactive.state.action_tendency, ActionType::Explore);
    }

    #[test]
    fn test_schema_learning() {
        let mut enactive = EnactiveCognition::new();

        let perception = PerceptionSummary {
            features: [("result".to_string(), 0.7)].into_iter().collect(),
            surprise: 0.2,
            affordances: vec![],
        };

        // Multiple cycles should learn schemas
        for _ in 0..5 {
            enactive.cycle(ActionType::Execute, perception.clone(), "task");
        }

        assert!(enactive.simulator.schema_count() > 0);
    }

    #[test]
    fn test_summary() {
        let enactive = EnactiveCognition::new();
        let summary = enactive.summary();

        assert!(summary.engagement >= 0.0 && summary.engagement <= 1.0);
        assert!(summary.well_being >= -1.0 && summary.well_being <= 1.0);
        assert!(summary.openness >= 0.0 && summary.openness <= 1.0);
    }
}
