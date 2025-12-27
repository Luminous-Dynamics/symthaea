//! Live Learner Module (Phase B4)
//!
//! Online reinforcement learning from user feedback, concept acquisition,
//! and adaptive response strategies. Enables continuous improvement
//! through interaction.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::hdc::binary_hv::HV16;

// ============================================================================
// FEEDBACK TYPES
// ============================================================================

/// Types of user feedback
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeedbackType {
    /// Explicit positive (e.g., "thanks", "good")
    ExplicitPositive,
    /// Explicit negative (e.g., "no", "wrong")
    ExplicitNegative,
    /// Implicit positive (continued engagement)
    ImplicitPositive,
    /// Implicit negative (disengagement, topic change)
    ImplicitNegative,
    /// Clarification request
    ClarificationRequest,
    /// Correction provided
    Correction,
}

impl FeedbackType {
    /// Convert to reward signal
    pub fn to_reward(&self) -> f32 {
        match self {
            Self::ExplicitPositive => 1.0,
            Self::ImplicitPositive => 0.3,
            Self::ClarificationRequest => -0.1,
            Self::ImplicitNegative => -0.3,
            Self::ExplicitNegative => -0.7,
            Self::Correction => -0.5,
        }
    }
}

/// A feedback signal from user interaction
#[derive(Debug, Clone)]
pub struct Feedback {
    /// Type of feedback
    pub feedback_type: FeedbackType,
    /// Confidence in detection (0.0-1.0)
    pub confidence: f32,
    /// Context that triggered the feedback
    pub context: String,
    /// Timestamp
    pub timestamp: u64,
}

impl Feedback {
    pub fn new(feedback_type: FeedbackType, context: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            feedback_type,
            confidence: 0.8,
            context,
            timestamp: now,
        }
    }

    /// Get reward value
    pub fn reward(&self) -> f32 {
        self.feedback_type.to_reward() * self.confidence
    }
}

// ============================================================================
// FEEDBACK COLLECTOR
// ============================================================================

/// Collects and analyzes user feedback
#[derive(Debug)]
pub struct FeedbackCollector {
    /// Feedback history
    history: VecDeque<Feedback>,
    /// Maximum history size
    max_history: usize,
    /// Positive keywords
    positive_keywords: Vec<&'static str>,
    /// Negative keywords
    negative_keywords: Vec<&'static str>,
    /// Clarification keywords
    clarification_keywords: Vec<&'static str>,
}

impl FeedbackCollector {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 100,
            positive_keywords: vec![
                "thanks", "thank you", "great", "good", "perfect", "exactly",
                "yes", "correct", "right", "helpful", "awesome", "nice",
                "love", "excellent", "wonderful", "amazing",
            ],
            negative_keywords: vec![
                "no", "wrong", "incorrect", "bad", "not what", "that's not",
                "you don't understand", "misunderstood", "confused",
            ],
            clarification_keywords: vec![
                "what do you mean", "can you explain", "i don't understand",
                "what?", "huh?", "clarify", "elaborate", "more detail",
            ],
        }
    }

    /// Analyze text for feedback signals
    pub fn analyze(&self, text: &str, previous_response: Option<&str>) -> Feedback {
        let text_lower = text.to_lowercase();

        // Check for explicit feedback first
        if self.contains_any(&text_lower, &self.positive_keywords) {
            return Feedback::new(FeedbackType::ExplicitPositive, text.to_string());
        }

        if self.contains_any(&text_lower, &self.negative_keywords) {
            return Feedback::new(FeedbackType::ExplicitNegative, text.to_string());
        }

        if self.contains_any(&text_lower, &self.clarification_keywords) {
            return Feedback::new(FeedbackType::ClarificationRequest, text.to_string());
        }

        // Check for correction pattern
        if let Some(prev) = previous_response {
            if text_lower.starts_with("actually") || text_lower.starts_with("no,") {
                return Feedback::new(FeedbackType::Correction, text.to_string());
            }

            // Check for topic continuation (implicit positive)
            let prev_lower = prev.to_lowercase();
            let prev_words: Vec<_> = prev_lower.split_whitespace().collect();
            let curr_words: Vec<_> = text_lower.split_whitespace().collect();

            let common = prev_words.iter()
                .filter(|w| curr_words.contains(w) && w.len() > 3)
                .count();

            if common > 2 {
                return Feedback::new(FeedbackType::ImplicitPositive, text.to_string());
            }
        }

        // Default to implicit positive (engagement is positive)
        let mut feedback = Feedback::new(FeedbackType::ImplicitPositive, text.to_string());
        feedback.confidence = 0.5; // Lower confidence for implicit
        feedback
    }

    fn contains_any(&self, text: &str, keywords: &[&str]) -> bool {
        keywords.iter().any(|k| text.contains(k))
    }

    /// Record feedback
    pub fn record(&mut self, feedback: Feedback) {
        self.history.push_back(feedback);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get recent feedback summary
    pub fn recent_summary(&self, n: usize) -> FeedbackSummary {
        let recent: Vec<_> = self.history.iter().rev().take(n).collect();

        if recent.is_empty() {
            return FeedbackSummary::default();
        }

        let total_reward: f32 = recent.iter().map(|f| f.reward()).sum();
        let positive_count = recent.iter()
            .filter(|f| matches!(f.feedback_type, FeedbackType::ExplicitPositive | FeedbackType::ImplicitPositive))
            .count();
        let negative_count = recent.iter()
            .filter(|f| matches!(f.feedback_type, FeedbackType::ExplicitNegative | FeedbackType::ImplicitNegative | FeedbackType::Correction))
            .count();

        FeedbackSummary {
            total_feedback: recent.len(),
            average_reward: total_reward / recent.len() as f32,
            positive_ratio: positive_count as f32 / recent.len() as f32,
            negative_ratio: negative_count as f32 / recent.len() as f32,
        }
    }
}

impl Default for FeedbackCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of feedback statistics
#[derive(Debug, Clone, Default)]
pub struct FeedbackSummary {
    pub total_feedback: usize,
    pub average_reward: f32,
    pub positive_ratio: f32,
    pub negative_ratio: f32,
}

// ============================================================================
// REWARD SIGNAL
// ============================================================================

/// Reward signal for reinforcement learning
#[derive(Debug, Clone)]
pub struct RewardSignal {
    /// The reward value (-1.0 to 1.0)
    pub value: f32,
    /// Associated state (context encoding)
    pub state: HV16,
    /// Action taken (response encoding)
    pub action: HV16,
    /// Timestamp
    pub timestamp: u64,
}

impl RewardSignal {
    pub fn new(value: f32, state: HV16, action: HV16) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            value: value.clamp(-1.0, 1.0),
            state,
            action,
            timestamp: now,
        }
    }
}

/// Converts feedback to reward signals
#[derive(Debug)]
pub struct RewardProcessor {
    /// Discount factor for temporal difference
    gamma: f32,
    /// Learning rate
    alpha: f32,
    /// Recent rewards for averaging
    recent_rewards: VecDeque<f32>,
}

impl RewardProcessor {
    pub fn new() -> Self {
        Self {
            gamma: 0.9,
            alpha: 0.1,
            recent_rewards: VecDeque::new(),
        }
    }

    /// Process feedback into reward signal
    pub fn process(&mut self, feedback: &Feedback, context: &str, response: &str) -> RewardSignal {
        let value = feedback.reward();

        // Create state and action encodings
        let state_seed = context.bytes().fold(12345u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        let action_seed = response.bytes().fold(54321u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });

        let signal = RewardSignal::new(
            value,
            HV16::random(state_seed),
            HV16::random(action_seed),
        );

        // Track reward
        self.recent_rewards.push_back(value);
        if self.recent_rewards.len() > 50 {
            self.recent_rewards.pop_front();
        }

        signal
    }

    /// Get average recent reward
    pub fn average_reward(&self) -> f32 {
        if self.recent_rewards.is_empty() {
            return 0.0;
        }
        self.recent_rewards.iter().sum::<f32>() / self.recent_rewards.len() as f32
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f32 {
        self.alpha
    }

    /// Adapt learning rate based on performance
    pub fn adapt_learning_rate(&mut self) {
        let avg = self.average_reward();

        // Increase learning rate if doing poorly, decrease if doing well
        if avg < -0.2 {
            self.alpha = (self.alpha * 1.1).min(0.3);
        } else if avg > 0.5 {
            self.alpha = (self.alpha * 0.9).max(0.01);
        }
    }
}

impl Default for RewardProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CONCEPT LEARNER
// ============================================================================

/// A learned concept
#[derive(Debug, Clone)]
pub struct LearnedConcept {
    /// Concept name/identifier
    pub name: String,
    /// HDC encoding
    pub encoding: HV16,
    /// Confidence in understanding (0.0-1.0)
    pub confidence: f32,
    /// Number of times encountered
    pub encounter_count: u32,
    /// Last updated timestamp
    pub last_updated: u64,
    /// Context words associated
    pub context_words: Vec<String>,
}

impl LearnedConcept {
    pub fn new(name: String, context: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let seed = name.bytes().fold(98765u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });

        let context_words: Vec<String> = context.split_whitespace()
            .filter(|w| w.len() > 3)
            .map(String::from)
            .collect();

        Self {
            name,
            encoding: HV16::random(seed),
            confidence: 0.3, // Start with low confidence
            encounter_count: 1,
            last_updated: now,
            context_words,
        }
    }

    /// Reinforce the concept (increases confidence)
    pub fn reinforce(&mut self, context: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.encounter_count += 1;
        self.last_updated = now;

        // Confidence grows with encounters
        self.confidence = (self.confidence + 0.1).min(1.0);

        // Add new context words
        for word in context.split_whitespace() {
            if word.len() > 3 && !self.context_words.contains(&word.to_string()) {
                self.context_words.push(word.to_string());
            }
        }
    }
}

/// Learns new concepts from conversation
#[derive(Debug)]
pub struct ConceptLearner {
    /// Learned concepts by name
    concepts: HashMap<String, LearnedConcept>,
    /// Learning threshold (minimum encounters to consider learned)
    learning_threshold: u32,
    /// Maximum concepts to store
    max_concepts: usize,
}

impl ConceptLearner {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            learning_threshold: 3,
            max_concepts: 1000,
        }
    }

    /// Learn or reinforce a concept
    pub fn learn(&mut self, name: &str, context: &str) -> &LearnedConcept {
        let name_lower = name.to_lowercase();

        if let Some(concept) = self.concepts.get_mut(&name_lower) {
            concept.reinforce(context);
        } else {
            // Check capacity
            if self.concepts.len() >= self.max_concepts {
                self.prune_weak_concepts();
            }

            let concept = LearnedConcept::new(name_lower.clone(), context);
            self.concepts.insert(name_lower.clone(), concept);
        }

        self.concepts.get(&name_lower).unwrap()
    }

    /// Check if a concept is learned (above threshold)
    pub fn is_learned(&self, name: &str) -> bool {
        self.concepts.get(&name.to_lowercase())
            .map(|c| c.encounter_count >= self.learning_threshold)
            .unwrap_or(false)
    }

    /// Get concept confidence
    pub fn confidence(&self, name: &str) -> f32 {
        self.concepts.get(&name.to_lowercase())
            .map(|c| c.confidence)
            .unwrap_or(0.0)
    }

    /// Find similar concepts
    pub fn find_similar(&self, query: &str, limit: usize) -> Vec<(&LearnedConcept, f32)> {
        let query_seed = query.bytes().fold(11111u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        let query_encoding = HV16::random(query_seed);

        let mut results: Vec<_> = self.concepts.values()
            .map(|c| (c, c.encoding.similarity(&query_encoding)))
            .filter(|(_, sim)| *sim > 0.2)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(limit).collect()
    }

    fn prune_weak_concepts(&mut self) {
        // Remove concepts with low confidence and few encounters
        let to_remove: Vec<_> = self.concepts.iter()
            .filter(|(_, c)| c.confidence < 0.3 && c.encounter_count < 2)
            .map(|(k, _)| k.clone())
            .collect();

        for key in to_remove.iter().take(100) {
            self.concepts.remove(key);
        }
    }

    /// Get concept count
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Get learned concept count (above threshold)
    pub fn learned_count(&self) -> usize {
        self.concepts.values()
            .filter(|c| c.encounter_count >= self.learning_threshold)
            .count()
    }
}

impl Default for ConceptLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ADAPTIVE POLICY
// ============================================================================

/// Response strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseStrategy {
    /// Provide detailed explanations
    Detailed,
    /// Keep responses concise
    Concise,
    /// Ask clarifying questions
    Clarifying,
    /// Acknowledge and validate
    Supportive,
    /// Offer new perspectives
    Exploratory,
}

impl ResponseStrategy {
    pub fn all() -> &'static [ResponseStrategy] {
        &[
            Self::Detailed,
            Self::Concise,
            Self::Clarifying,
            Self::Supportive,
            Self::Exploratory,
        ]
    }
}

/// Adaptive policy for response selection
#[derive(Debug)]
pub struct AdaptivePolicy {
    /// Q-values for each strategy
    q_values: HashMap<ResponseStrategy, f32>,
    /// Strategy usage history
    history: VecDeque<(ResponseStrategy, f32)>,
    /// Exploration rate (epsilon)
    epsilon: f32,
    /// Learning rate
    alpha: f32,
    /// Current preferred strategy
    current_strategy: ResponseStrategy,
}

impl AdaptivePolicy {
    pub fn new() -> Self {
        let mut q_values = HashMap::new();
        for strategy in ResponseStrategy::all() {
            q_values.insert(*strategy, 0.5); // Initialize neutrally
        }

        Self {
            q_values,
            history: VecDeque::new(),
            epsilon: 0.2, // 20% exploration
            alpha: 0.1,
            current_strategy: ResponseStrategy::Supportive,
        }
    }

    /// Select strategy (epsilon-greedy)
    pub fn select_strategy(&mut self, context_hash: u64) -> ResponseStrategy {
        // Use context_hash for deterministic "randomness"
        let explore = (context_hash % 100) < (self.epsilon * 100.0) as u64;

        if explore {
            // Explore: select based on context hash
            let strategies = ResponseStrategy::all();
            let idx = (context_hash / 100) as usize % strategies.len();
            self.current_strategy = strategies[idx];
        } else {
            // Exploit: select best known strategy
            self.current_strategy = *self.q_values.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(s, _)| s)
                .unwrap_or(&ResponseStrategy::Supportive);
        }

        self.current_strategy
    }

    /// Update Q-value based on reward
    pub fn update(&mut self, strategy: ResponseStrategy, reward: f32) {
        let old_value = *self.q_values.get(&strategy).unwrap_or(&0.5);
        let new_value = old_value + self.alpha * (reward - old_value);
        self.q_values.insert(strategy, new_value);

        // Track history
        self.history.push_back((strategy, reward));
        if self.history.len() > 100 {
            self.history.pop_front();
        }

        // Adapt exploration rate
        self.adapt_epsilon();
    }

    fn adapt_epsilon(&mut self) {
        // Reduce exploration as we learn
        let avg_reward: f32 = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().map(|(_, r)| *r).sum::<f32>() / self.history.len() as f32
        };

        if avg_reward > 0.5 {
            self.epsilon = (self.epsilon * 0.95).max(0.05);
        } else if avg_reward < 0.0 {
            self.epsilon = (self.epsilon * 1.1).min(0.4);
        }
    }

    /// Get current best strategy
    pub fn best_strategy(&self) -> ResponseStrategy {
        *self.q_values.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| s)
            .unwrap_or(&ResponseStrategy::Supportive)
    }

    /// Get strategy Q-value
    pub fn q_value(&self, strategy: ResponseStrategy) -> f32 {
        *self.q_values.get(&strategy).unwrap_or(&0.5)
    }

    /// Get exploration rate
    pub fn exploration_rate(&self) -> f32 {
        self.epsilon
    }
}

impl Default for AdaptivePolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LIVE LEARNER (MAIN)
// ============================================================================

/// Complete live learning system
#[derive(Debug)]
pub struct LiveLearner {
    /// Feedback collection
    pub feedback: FeedbackCollector,
    /// Reward processing
    pub reward: RewardProcessor,
    /// Concept learning
    pub concepts: ConceptLearner,
    /// Adaptive policy
    pub policy: AdaptivePolicy,
    /// Total interactions processed
    interaction_count: u64,
}

impl LiveLearner {
    pub fn new() -> Self {
        Self {
            feedback: FeedbackCollector::new(),
            reward: RewardProcessor::new(),
            concepts: ConceptLearner::new(),
            policy: AdaptivePolicy::new(),
            interaction_count: 0,
        }
    }

    /// Process an interaction for learning
    pub fn learn_from_interaction(
        &mut self,
        user_input: &str,
        my_response: &str,
        previous_response: Option<&str>,
    ) -> LearningResult {
        self.interaction_count += 1;

        // Collect feedback
        let feedback = self.feedback.analyze(user_input, previous_response);
        self.feedback.record(feedback.clone());

        // Process reward
        let reward_signal = self.reward.process(&feedback, user_input, my_response);

        // Update policy
        let strategy = self.policy.current_strategy;
        self.policy.update(strategy, reward_signal.value);

        // Learn concepts from input
        let words: Vec<_> = user_input.split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();

        let mut concepts_learned = Vec::new();
        for word in words {
            let concept = self.concepts.learn(word, user_input);
            if concept.encounter_count == self.concepts.learning_threshold {
                concepts_learned.push(word.to_string());
            }
        }

        // Adapt learning rate
        if self.interaction_count % 10 == 0 {
            self.reward.adapt_learning_rate();
        }

        LearningResult {
            feedback_type: feedback.feedback_type,
            reward: reward_signal.value,
            strategy_used: strategy,
            concepts_learned,
            total_interactions: self.interaction_count,
        }
    }

    /// Select response strategy for next response
    pub fn select_strategy(&mut self, context: &str) -> ResponseStrategy {
        let context_hash = context.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });

        self.policy.select_strategy(context_hash)
    }

    /// Get learning statistics
    pub fn stats(&self) -> LearningStats {
        let summary = self.feedback.recent_summary(50);

        LearningStats {
            total_interactions: self.interaction_count,
            average_reward: self.reward.average_reward(),
            concepts_learned: self.concepts.learned_count(),
            total_concepts: self.concepts.concept_count(),
            best_strategy: self.policy.best_strategy(),
            exploration_rate: self.policy.exploration_rate(),
            positive_feedback_ratio: summary.positive_ratio,
        }
    }
}

impl Default for LiveLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of learning from an interaction
#[derive(Debug, Clone)]
pub struct LearningResult {
    pub feedback_type: FeedbackType,
    pub reward: f32,
    pub strategy_used: ResponseStrategy,
    pub concepts_learned: Vec<String>,
    pub total_interactions: u64,
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStats {
    pub total_interactions: u64,
    pub average_reward: f32,
    pub concepts_learned: usize,
    pub total_concepts: usize,
    pub best_strategy: ResponseStrategy,
    pub exploration_rate: f32,
    pub positive_feedback_ratio: f32,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Feedback Tests
    #[test]
    fn test_feedback_type_rewards() {
        assert!(FeedbackType::ExplicitPositive.to_reward() > 0.0);
        assert!(FeedbackType::ExplicitNegative.to_reward() < 0.0);
    }

    #[test]
    fn test_feedback_creation() {
        let feedback = Feedback::new(FeedbackType::ExplicitPositive, "great!".to_string());
        assert!(feedback.reward() > 0.0);
    }

    // FeedbackCollector Tests
    #[test]
    fn test_collector_creation() {
        let collector = FeedbackCollector::new();
        assert!(!collector.positive_keywords.is_empty());
    }

    #[test]
    fn test_positive_feedback_detection() {
        let collector = FeedbackCollector::new();
        let feedback = collector.analyze("Thanks, that was helpful!", None);
        assert_eq!(feedback.feedback_type, FeedbackType::ExplicitPositive);
    }

    #[test]
    fn test_negative_feedback_detection() {
        let collector = FeedbackCollector::new();
        let feedback = collector.analyze("No, that's wrong", None);
        assert_eq!(feedback.feedback_type, FeedbackType::ExplicitNegative);
    }

    // RewardProcessor Tests
    #[test]
    fn test_reward_processor_creation() {
        let processor = RewardProcessor::new();
        assert_eq!(processor.average_reward(), 0.0);
    }

    #[test]
    fn test_reward_processing() {
        let mut processor = RewardProcessor::new();
        let feedback = Feedback::new(FeedbackType::ExplicitPositive, "great".to_string());
        let signal = processor.process(&feedback, "context", "response");

        assert!(signal.value > 0.0);
    }

    // ConceptLearner Tests
    #[test]
    fn test_concept_learner_creation() {
        let learner = ConceptLearner::new();
        assert_eq!(learner.concept_count(), 0);
    }

    #[test]
    fn test_concept_learning() {
        let mut learner = ConceptLearner::new();

        // Learn a concept multiple times
        learner.learn("consciousness", "exploring consciousness deeply");
        learner.learn("consciousness", "what is consciousness");
        learner.learn("consciousness", "consciousness matters");

        assert!(learner.is_learned("consciousness"));
        assert!(learner.confidence("consciousness") > 0.3);
    }

    // AdaptivePolicy Tests
    #[test]
    fn test_policy_creation() {
        let policy = AdaptivePolicy::new();
        assert!(policy.exploration_rate() > 0.0);
    }

    #[test]
    fn test_policy_update() {
        let mut policy = AdaptivePolicy::new();

        // Give positive feedback to Detailed strategy
        policy.update(ResponseStrategy::Detailed, 1.0);
        policy.update(ResponseStrategy::Detailed, 0.8);
        policy.update(ResponseStrategy::Detailed, 0.9);

        // Should now prefer Detailed
        assert!(policy.q_value(ResponseStrategy::Detailed) > 0.5);
    }

    // LiveLearner Tests
    #[test]
    fn test_live_learner_creation() {
        let learner = LiveLearner::new();
        assert_eq!(learner.interaction_count, 0);
    }

    #[test]
    fn test_learning_from_interaction() {
        let mut learner = LiveLearner::new();

        let result = learner.learn_from_interaction(
            "Thanks for explaining consciousness!",
            "You're welcome! Consciousness is fascinating.",
            None,
        );

        assert_eq!(result.feedback_type, FeedbackType::ExplicitPositive);
        assert!(result.reward > 0.0);
    }

    #[test]
    fn test_strategy_selection() {
        let mut learner = LiveLearner::new();

        // Process some interactions with positive feedback
        for _ in 0..10 {
            learner.learn_from_interaction(
                "Thanks!",
                "Glad to help!",
                None,
            );
        }

        let strategy = learner.select_strategy("new context");
        assert!(ResponseStrategy::all().contains(&strategy));
    }

    #[test]
    fn test_learning_stats() {
        let mut learner = LiveLearner::new();

        learner.learn_from_interaction("Thanks!", "Welcome!", None);
        learner.learn_from_interaction("Great!", "Thanks!", None);

        let stats = learner.stats();
        assert_eq!(stats.total_interactions, 2);
        assert!(stats.positive_feedback_ratio > 0.0);
    }
}
