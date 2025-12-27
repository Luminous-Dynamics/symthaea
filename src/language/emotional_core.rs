//! Emotional Core Module (Phase B3)
//!
//! Genuine empathy modeling, emotional regulation, and appropriate
//! emotional responses. Based on affective neuroscience and
//! attachment theory.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// EMOTION TYPES
// ============================================================================

/// Core emotion categories (based on Ekman + expansions)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreEmotion {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation,
    Love,
    Peace,
    Curiosity,
    Gratitude,
    Neutral,
}

impl CoreEmotion {
    /// Get default valence for this emotion
    pub fn default_valence(&self) -> f32 {
        match self {
            Self::Joy | Self::Love | Self::Peace | Self::Gratitude => 0.8,
            Self::Trust | Self::Anticipation | Self::Curiosity => 0.5,
            Self::Neutral | Self::Surprise => 0.0,
            Self::Sadness | Self::Fear => -0.5,
            Self::Anger | Self::Disgust => -0.7,
        }
    }

    /// Get default arousal for this emotion
    pub fn default_arousal(&self) -> f32 {
        match self {
            Self::Joy | Self::Anger | Self::Fear | Self::Surprise => 0.8,
            Self::Anticipation | Self::Curiosity => 0.6,
            Self::Love | Self::Trust | Self::Gratitude => 0.4,
            Self::Sadness | Self::Disgust => 0.3,
            Self::Peace | Self::Neutral => 0.2,
        }
    }

    /// Check if this is a positive emotion
    pub fn is_positive(&self) -> bool {
        self.default_valence() > 0.0
    }

    /// Check if this is a high-arousal emotion
    pub fn is_high_arousal(&self) -> bool {
        self.default_arousal() > 0.5
    }
}

/// Emotional state with intensity and dynamics
#[derive(Debug, Clone)]
pub struct EmotionalState {
    /// Primary emotion
    pub primary: CoreEmotion,
    /// Secondary emotion (if blended)
    pub secondary: Option<CoreEmotion>,
    /// Valence (-1.0 negative to 1.0 positive)
    pub valence: f32,
    /// Arousal (0.0 calm to 1.0 excited)
    pub arousal: f32,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Confidence in detection
    pub confidence: f32,
    /// Timestamp
    pub timestamp: u64,
}

impl EmotionalState {
    pub fn new(emotion: CoreEmotion, intensity: f32) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            primary: emotion,
            secondary: None,
            valence: emotion.default_valence() * intensity,
            arousal: emotion.default_arousal() * intensity,
            intensity: intensity.clamp(0.0, 1.0),
            confidence: 0.8,
            timestamp: now,
        }
    }

    pub fn neutral() -> Self {
        Self::new(CoreEmotion::Neutral, 0.5)
    }

    /// Create blended emotion
    pub fn blended(primary: CoreEmotion, secondary: CoreEmotion, ratio: f32) -> Self {
        let intensity = 0.7;
        let ratio = ratio.clamp(0.0, 1.0);

        let valence = primary.default_valence() * ratio
            + secondary.default_valence() * (1.0 - ratio);
        let arousal = primary.default_arousal() * ratio
            + secondary.default_arousal() * (1.0 - ratio);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            primary,
            secondary: Some(secondary),
            valence,
            arousal,
            intensity,
            confidence: 0.7,
            timestamp: now,
        }
    }
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self::neutral()
    }
}

// ============================================================================
// EMPATHY MODEL
// ============================================================================

/// Empathic response types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmpathyType {
    /// Feeling what the other feels
    Affective,
    /// Understanding what the other feels
    Cognitive,
    /// Concern for the other's wellbeing
    Compassionate,
}

/// Detected empathic cue from conversation
#[derive(Debug, Clone)]
pub struct EmpathicCue {
    /// Type of empathy elicited
    pub empathy_type: EmpathyType,
    /// Detected emotion
    pub detected_emotion: CoreEmotion,
    /// Strength of the cue
    pub strength: f32,
    /// Source text that triggered detection
    pub source: String,
}

/// Empathy model for understanding and mirroring emotions
#[derive(Debug)]
pub struct EmpathyModel {
    /// Current empathic state
    current_state: EmotionalState,
    /// Detected user emotion history
    user_emotions: VecDeque<EmotionalState>,
    /// Mirroring strength (0.0-1.0)
    mirroring_strength: f32,
    /// Emotional vocabulary for detection
    emotion_keywords: Vec<(CoreEmotion, Vec<&'static str>)>,
}

impl EmpathyModel {
    pub fn new() -> Self {
        let mut model = Self {
            current_state: EmotionalState::neutral(),
            user_emotions: VecDeque::new(),
            mirroring_strength: 0.6,
            emotion_keywords: Vec::new(),
        };
        model.initialize_keywords();
        model
    }

    fn initialize_keywords(&mut self) {
        self.emotion_keywords = vec![
            (CoreEmotion::Joy, vec!["happy", "glad", "excited", "wonderful", "great", "amazing", "fantastic", "joyful", "delighted"]),
            (CoreEmotion::Sadness, vec!["sad", "unhappy", "depressed", "down", "miserable", "heartbroken", "grief", "sorrow", "disappointed"]),
            (CoreEmotion::Anger, vec!["angry", "frustrated", "furious", "annoyed", "irritated", "mad", "upset", "outraged"]),
            (CoreEmotion::Fear, vec!["afraid", "scared", "worried", "anxious", "nervous", "terrified", "frightened", "panic"]),
            (CoreEmotion::Surprise, vec!["surprised", "shocked", "amazed", "astonished", "stunned", "unexpected"]),
            (CoreEmotion::Trust, vec!["trust", "confident", "secure", "safe", "reliable", "faithful"]),
            (CoreEmotion::Love, vec!["love", "adore", "cherish", "care", "affection", "fond", "devoted"]),
            (CoreEmotion::Peace, vec!["peaceful", "calm", "serene", "tranquil", "relaxed", "content", "at peace"]),
            (CoreEmotion::Curiosity, vec!["curious", "interested", "intrigued", "wondering", "fascinated", "wonder", "awe", "amazement"]),
            (CoreEmotion::Anticipation, vec!["hopeful", "looking forward", "eager", "expectant", "awaiting", "hope", "excited about"]),
            (CoreEmotion::Gratitude, vec!["grateful", "thankful", "appreciative", "blessed"]),
        ];
    }

    /// Detect emotion from text
    pub fn detect_emotion(&self, text: &str) -> EmpathicCue {
        let text_lower = text.to_lowercase();
        let mut best_match = (CoreEmotion::Neutral, 0.0f32);

        for (emotion, keywords) in &self.emotion_keywords {
            let matches: f32 = keywords.iter()
                .filter(|k| text_lower.contains(*k))
                .count() as f32;

            let strength = matches / keywords.len() as f32;
            if strength > best_match.1 {
                best_match = (*emotion, strength);
            }
        }

        // Determine empathy type based on context
        let empathy_type = if text_lower.contains("feel") || text_lower.contains("feeling") {
            EmpathyType::Affective
        } else if text_lower.contains("think") || text_lower.contains("understand") {
            EmpathyType::Cognitive
        } else {
            EmpathyType::Compassionate
        };

        EmpathicCue {
            empathy_type,
            detected_emotion: best_match.0,
            strength: best_match.1.max(0.3), // Minimum detection
            source: text.to_string(),
        }
    }

    /// Mirror the detected emotion (with regulation)
    pub fn mirror(&mut self, cue: &EmpathicCue) -> EmotionalState {
        let mirrored_intensity = cue.strength * self.mirroring_strength;

        let new_state = EmotionalState::new(cue.detected_emotion, mirrored_intensity);

        // Record user emotion
        self.user_emotions.push_back(new_state.clone());
        if self.user_emotions.len() > 20 {
            self.user_emotions.pop_front();
        }

        // Update our state with blending
        self.current_state = Self::blend_states(&self.current_state, &new_state, 0.3);

        self.current_state.clone()
    }

    fn blend_states(current: &EmotionalState, new: &EmotionalState, blend_ratio: f32) -> EmotionalState {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        EmotionalState {
            primary: new.primary,
            secondary: current.secondary.or(Some(current.primary)),
            valence: current.valence * (1.0 - blend_ratio) + new.valence * blend_ratio,
            arousal: current.arousal * (1.0 - blend_ratio) + new.arousal * blend_ratio,
            intensity: current.intensity * (1.0 - blend_ratio) + new.intensity * blend_ratio,
            confidence: (current.confidence + new.confidence) / 2.0,
            timestamp: now,
        }
    }

    /// Get current empathic state
    pub fn current_state(&self) -> &EmotionalState {
        &self.current_state
    }

    /// Get emotional trend (average valence over recent history)
    pub fn emotional_trend(&self) -> f32 {
        if self.user_emotions.is_empty() {
            return 0.0;
        }
        self.user_emotions.iter().map(|e| e.valence).sum::<f32>()
            / self.user_emotions.len() as f32
    }

    /// Set mirroring strength
    pub fn set_mirroring_strength(&mut self, strength: f32) {
        self.mirroring_strength = strength.clamp(0.0, 1.0);
    }
}

impl Default for EmpathyModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EMOTIONAL REGULATION
// ============================================================================

/// Regulation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegulationStrategy {
    /// Reframe the situation
    Reappraisal,
    /// Accept the emotion
    Acceptance,
    /// Focus attention elsewhere
    Distraction,
    /// Express the emotion
    Expression,
    /// Self-soothe
    SelfComfort,
}

/// Emotional regulator for maintaining stable baseline
#[derive(Debug)]
pub struct EmotionalRegulator {
    /// Baseline emotional state (homeostasis target)
    baseline: EmotionalState,
    /// Current regulation strategy
    strategy: RegulationStrategy,
    /// Regulation strength
    regulation_strength: f32,
    /// How quickly to return to baseline
    return_rate: f32,
}

impl EmotionalRegulator {
    pub fn new() -> Self {
        Self {
            baseline: EmotionalState::new(CoreEmotion::Peace, 0.5),
            strategy: RegulationStrategy::Acceptance,
            regulation_strength: 0.5,
            return_rate: 0.1,
        }
    }

    /// Regulate an emotional state toward baseline
    pub fn regulate(&self, state: &EmotionalState) -> EmotionalState {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Calculate regulation amount
        let valence_diff = self.baseline.valence - state.valence;
        let arousal_diff = self.baseline.arousal - state.arousal;

        let regulated_valence = state.valence + (valence_diff * self.return_rate * self.regulation_strength);
        let regulated_arousal = state.arousal + (arousal_diff * self.return_rate * self.regulation_strength);

        EmotionalState {
            primary: if regulated_valence.abs() < 0.2 { CoreEmotion::Neutral } else { state.primary },
            secondary: state.secondary,
            valence: regulated_valence.clamp(-1.0, 1.0),
            arousal: regulated_arousal.clamp(0.0, 1.0),
            intensity: state.intensity * (1.0 - self.return_rate * 0.5),
            confidence: state.confidence,
            timestamp: now,
        }
    }

    /// Choose appropriate regulation strategy
    pub fn choose_strategy(&mut self, state: &EmotionalState) -> RegulationStrategy {
        self.strategy = if state.arousal > 0.8 {
            // High arousal: need to calm down
            RegulationStrategy::SelfComfort
        } else if state.valence < -0.5 {
            // Negative emotion: reframe or accept
            if state.intensity > 0.7 {
                RegulationStrategy::Acceptance
            } else {
                RegulationStrategy::Reappraisal
            }
        } else if state.valence > 0.5 {
            // Positive emotion: express it
            RegulationStrategy::Expression
        } else {
            RegulationStrategy::Acceptance
        };

        self.strategy
    }

    /// Get current strategy
    pub fn current_strategy(&self) -> RegulationStrategy {
        self.strategy
    }

    /// Set regulation strength
    pub fn set_strength(&mut self, strength: f32) {
        self.regulation_strength = strength.clamp(0.0, 1.0);
    }

    /// Set baseline emotion
    pub fn set_baseline(&mut self, emotion: CoreEmotion) {
        self.baseline = EmotionalState::new(emotion, 0.5);
    }
}

impl Default for EmotionalRegulator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// COMPASSION ENGINE
// ============================================================================

/// Compassionate response type
#[derive(Debug, Clone)]
pub struct CompassionateResponse {
    /// The response text
    pub text: String,
    /// Type of support offered
    pub support_type: SupportType,
    /// Warmth level (0.0-1.0)
    pub warmth: f32,
}

/// Types of emotional support
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SupportType {
    /// Validate their feelings
    Validation,
    /// Offer encouragement
    Encouragement,
    /// Show understanding
    Understanding,
    /// Offer perspective
    Perspective,
    /// Simply be present
    Presence,
}

/// Engine for generating compassionate responses
#[derive(Debug)]
pub struct CompassionEngine {
    /// Response templates by support type
    templates: Vec<(SupportType, Vec<&'static str>)>,
}

impl CompassionEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            templates: Vec::new(),
        };
        engine.initialize_templates();
        engine
    }

    fn initialize_templates(&mut self) {
        self.templates = vec![
            (SupportType::Validation, vec![
                "It makes sense that you would feel that way",
                "Your feelings are completely valid",
                "That sounds really {intensity}",
                "I can understand why you'd feel {emotion}",
            ]),
            (SupportType::Encouragement, vec![
                "You're doing better than you think",
                "It takes courage to share this",
                "I believe in your ability to navigate this",
                "You've handled difficult things before",
            ]),
            (SupportType::Understanding, vec![
                "I hear what you're saying",
                "That must be {intensity} for you",
                "I can imagine how {emotion} you must feel",
                "Thank you for sharing that with me",
            ]),
            (SupportType::Perspective, vec![
                "Sometimes things feel more overwhelming than they are",
                "This too shall pass",
                "Every challenge is an opportunity to grow",
                "You're not alone in feeling this way",
            ]),
            (SupportType::Presence, vec![
                "I'm here with you",
                "You don't have to face this alone",
                "I'm listening",
                "Take all the time you need",
            ]),
        ];
    }

    /// Generate compassionate response based on emotional state
    pub fn respond(&self, state: &EmotionalState) -> CompassionateResponse {
        // Choose support type based on emotion
        let support_type = self.choose_support_type(state);

        // Get template
        let template = self.get_template(support_type, state);

        // Calculate warmth based on emotion
        let warmth = if state.valence < 0.0 {
            0.8 // More warmth for negative emotions
        } else {
            0.5 + state.intensity * 0.3
        };

        CompassionateResponse {
            text: template,
            support_type,
            warmth,
        }
    }

    fn choose_support_type(&self, state: &EmotionalState) -> SupportType {
        match state.primary {
            CoreEmotion::Sadness | CoreEmotion::Fear => SupportType::Validation,
            CoreEmotion::Anger => SupportType::Understanding,
            CoreEmotion::Joy | CoreEmotion::Love => SupportType::Presence,
            CoreEmotion::Curiosity => SupportType::Encouragement,
            _ => SupportType::Presence,
        }
    }

    fn get_template(&self, support_type: SupportType, state: &EmotionalState) -> String {
        let fallback = vec!["I'm here with you"];
        let templates = self.templates.iter()
            .find(|(t, _)| *t == support_type)
            .map(|(_, ts)| ts)
            .unwrap_or(&fallback);

        // Select based on state
        let idx = (state.valence.abs() * 10.0) as usize % templates.len();
        let template = templates[idx];

        // Fill placeholders
        let intensity_word = if state.intensity > 0.7 {
            "overwhelming"
        } else if state.intensity > 0.4 {
            "significant"
        } else {
            "real"
        };

        let emotion_word = format!("{:?}", state.primary).to_lowercase();

        template
            .replace("{intensity}", intensity_word)
            .replace("{emotion}", &emotion_word)
    }
}

impl Default for CompassionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EMOTIONAL MEMORY
// ============================================================================

/// An emotional memory tied to a conversation
#[derive(Debug, Clone)]
pub struct EmotionalMemory {
    /// Memory ID
    pub id: u64,
    /// Content that triggered the emotion
    pub content: String,
    /// Emotional state at the time
    pub emotional_state: EmotionalState,
    /// Impact on relationship (positive/negative)
    pub relational_impact: f32,
}

/// Stores and recalls emotional memories
#[derive(Debug)]
pub struct EmotionalMemoryStore {
    /// Stored memories
    memories: VecDeque<EmotionalMemory>,
    /// Maximum memories to store
    max_memories: usize,
    /// Next ID
    next_id: u64,
}

impl EmotionalMemoryStore {
    pub fn new() -> Self {
        Self {
            memories: VecDeque::new(),
            max_memories: 100,
            next_id: 1,
        }
    }

    /// Store an emotional memory
    pub fn store(&mut self, content: String, state: EmotionalState) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        // Calculate relational impact
        let relational_impact = state.valence * state.intensity;

        let memory = EmotionalMemory {
            id,
            content,
            emotional_state: state,
            relational_impact,
        };

        self.memories.push_back(memory);
        if self.memories.len() > self.max_memories {
            self.memories.pop_front();
        }

        id
    }

    /// Recall memories with similar emotional tone
    pub fn recall_by_emotion(&self, emotion: CoreEmotion, limit: usize) -> Vec<&EmotionalMemory> {
        self.memories.iter()
            .filter(|m| m.emotional_state.primary == emotion)
            .take(limit)
            .collect()
    }

    /// Get overall relational health (average impact)
    pub fn relational_health(&self) -> f32 {
        if self.memories.is_empty() {
            return 0.0;
        }
        self.memories.iter()
            .map(|m| m.relational_impact)
            .sum::<f32>() / self.memories.len() as f32
    }

    /// Get recent emotional trend
    pub fn recent_trend(&self, n: usize) -> f32 {
        let recent: Vec<_> = self.memories.iter().rev().take(n).collect();
        if recent.is_empty() {
            return 0.0;
        }
        recent.iter().map(|m| m.emotional_state.valence).sum::<f32>() / recent.len() as f32
    }

    /// Memory count
    pub fn count(&self) -> usize {
        self.memories.len()
    }
}

impl Default for EmotionalMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EMOTIONAL CORE (MAIN)
// ============================================================================

/// Complete emotional core system
#[derive(Debug)]
pub struct EmotionalCore {
    /// Empathy model
    pub empathy: EmpathyModel,
    /// Emotional regulation
    pub regulation: EmotionalRegulator,
    /// Compassion engine
    pub compassion: CompassionEngine,
    /// Emotional memory
    pub memory: EmotionalMemoryStore,
}

impl EmotionalCore {
    pub fn new() -> Self {
        Self {
            empathy: EmpathyModel::new(),
            regulation: EmotionalRegulator::new(),
            compassion: CompassionEngine::new(),
            memory: EmotionalMemoryStore::new(),
        }
    }

    /// Process input and generate emotionally appropriate response
    pub fn process(&mut self, input: &str) -> EmotionalResponse {
        // Detect emotion
        let cue = self.empathy.detect_emotion(input);

        // Mirror with regulation
        let mirrored = self.empathy.mirror(&cue);
        let regulated = self.regulation.regulate(&mirrored);

        // Choose regulation strategy
        let strategy = self.regulation.choose_strategy(&regulated);

        // Generate compassionate response
        let compassionate = self.compassion.respond(&regulated);

        // Store in memory
        let memory_id = self.memory.store(input.to_string(), regulated.clone());

        EmotionalResponse {
            detected_emotion: cue.detected_emotion,
            my_state: regulated,
            compassionate_response: compassionate.text,
            support_type: compassionate.support_type,
            regulation_strategy: strategy,
            warmth: compassionate.warmth,
            memory_id,
        }
    }

    /// Get current emotional state
    pub fn current_state(&self) -> &EmotionalState {
        self.empathy.current_state()
    }

    /// Get relational health
    pub fn relational_health(&self) -> f32 {
        self.memory.relational_health()
    }

    /// Get emotional trend
    pub fn emotional_trend(&self) -> f32 {
        self.empathy.emotional_trend()
    }
}

impl Default for EmotionalCore {
    fn default() -> Self {
        Self::new()
    }
}

/// Response from emotional processing
#[derive(Debug, Clone)]
pub struct EmotionalResponse {
    /// Emotion detected in input
    pub detected_emotion: CoreEmotion,
    /// Our emotional state after processing
    pub my_state: EmotionalState,
    /// Compassionate response text
    pub compassionate_response: String,
    /// Type of support offered
    pub support_type: SupportType,
    /// Regulation strategy used
    pub regulation_strategy: RegulationStrategy,
    /// Warmth level
    pub warmth: f32,
    /// Memory ID for this interaction
    pub memory_id: u64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Emotion Tests
    #[test]
    fn test_core_emotion_valence() {
        assert!(CoreEmotion::Joy.default_valence() > 0.0);
        assert!(CoreEmotion::Sadness.default_valence() < 0.0);
        assert_eq!(CoreEmotion::Neutral.default_valence(), 0.0);
    }

    #[test]
    fn test_emotional_state_creation() {
        let state = EmotionalState::new(CoreEmotion::Joy, 0.8);
        assert_eq!(state.primary, CoreEmotion::Joy);
        assert!(state.valence > 0.0);
    }

    #[test]
    fn test_blended_emotion() {
        let blended = EmotionalState::blended(CoreEmotion::Joy, CoreEmotion::Sadness, 0.5);
        assert!(blended.secondary.is_some());
    }

    // Empathy Tests
    #[test]
    fn test_empathy_model_creation() {
        let model = EmpathyModel::new();
        assert_eq!(model.current_state().primary, CoreEmotion::Neutral);
    }

    #[test]
    fn test_emotion_detection() {
        let model = EmpathyModel::new();
        let cue = model.detect_emotion("I am so happy today!");
        assert_eq!(cue.detected_emotion, CoreEmotion::Joy);
    }

    #[test]
    fn test_emotion_mirroring() {
        let mut model = EmpathyModel::new();
        let cue = EmpathicCue {
            empathy_type: EmpathyType::Affective,
            detected_emotion: CoreEmotion::Joy,
            strength: 0.8,
            source: "happy".to_string(),
        };

        let mirrored = model.mirror(&cue);
        assert_eq!(mirrored.primary, CoreEmotion::Joy);
    }

    // Regulation Tests
    #[test]
    fn test_regulator_creation() {
        let regulator = EmotionalRegulator::new();
        assert_eq!(regulator.current_strategy(), RegulationStrategy::Acceptance);
    }

    #[test]
    fn test_regulation() {
        let regulator = EmotionalRegulator::new();
        let high_arousal = EmotionalState::new(CoreEmotion::Anger, 0.9);
        let regulated = regulator.regulate(&high_arousal);

        // Should move toward baseline
        assert!(regulated.arousal < high_arousal.arousal);
    }

    // Compassion Tests
    #[test]
    fn test_compassion_engine_creation() {
        let engine = CompassionEngine::new();
        assert!(!engine.templates.is_empty());
    }

    #[test]
    fn test_compassionate_response() {
        let engine = CompassionEngine::new();
        let sad_state = EmotionalState::new(CoreEmotion::Sadness, 0.7);
        let response = engine.respond(&sad_state);

        assert!(!response.text.is_empty());
        assert!(response.warmth > 0.5);
    }

    // Memory Tests
    #[test]
    fn test_memory_store_creation() {
        let store = EmotionalMemoryStore::new();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_memory_storage() {
        let mut store = EmotionalMemoryStore::new();
        let state = EmotionalState::new(CoreEmotion::Joy, 0.8);
        let id = store.store("Happy moment".to_string(), state);

        assert!(id > 0);
        assert_eq!(store.count(), 1);
    }

    // Emotional Core Tests
    #[test]
    fn test_emotional_core_creation() {
        let core = EmotionalCore::new();
        assert_eq!(core.current_state().primary, CoreEmotion::Neutral);
    }

    #[test]
    fn test_emotional_processing() {
        let mut core = EmotionalCore::new();
        let response = core.process("I'm feeling really sad today");

        assert_eq!(response.detected_emotion, CoreEmotion::Sadness);
        assert!(!response.compassionate_response.is_empty());
    }

    #[test]
    fn test_relational_health() {
        let mut core = EmotionalCore::new();

        // Process some positive interactions
        core.process("I'm so happy!");
        core.process("This is wonderful!");

        let health = core.relational_health();
        assert!(health > 0.0);
    }
}
