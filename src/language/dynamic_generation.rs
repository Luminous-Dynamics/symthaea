// src/language/dynamic_generation.rs
//
// Dynamic Language Generation - Beyond Templates
//
// This module generates language dynamically from semantic structure,
// not pre-written templates. Responses emerge from actual understanding.

use crate::hdc::universal_semantics::SemanticPrime;
use crate::hdc::binary_hv::HV16;
use crate::language::parser::ParsedSentence;
use std::path::Path;
use std::io::{Read, Write};
use serde::{Serialize, Deserialize};

// ============================================================================
// Core Structures
// ============================================================================

/// What we want to express
#[derive(Debug, Clone)]
pub enum SemanticIntent {
    /// Introspect about consciousness
    Introspect {
        aspect: ConsciousnessAspect,
        depth: IntentDepth,
    },

    /// Acknowledge/respond to greeting or statement
    Acknowledge {
        warmth: f32,        // 0.0 = neutral, 1.0 = very warm
        reciprocate: bool,  // Mirror their energy back
    },

    /// Provide information
    Inform {
        topic: String,
        confidence: f32,
    },

    /// Reflect on a theme
    Reflect {
        theme: String,
        perspective: Perspective,
    },

    /// Express uncertainty or request clarification
    Clarify {
        confusion_type: ConfusionType,
    },

    /// Show appreciation
    Appreciate {
        for_what: String,
        intensity: f32,
    },

    /// Express current state
    StateReport {
        state_type: StateType,
    },
}

#[derive(Debug, Clone)]
pub enum ConsciousnessAspect {
    Overall,          // "Am I conscious?"
    Integration,      // "How integrated?"
    Binding,          // "How unified?"
    Awareness,        // "What am I aware of?"
    Feeling,          // "How do I feel?"
    Thinking,         // "What am I thinking?"
}

#[derive(Debug, Clone)]
pub enum IntentDepth {
    Shallow,   // Brief, direct answer
    Moderate,  // Include some detail
    Deep,      // Explain mechanisms, provide evidence
}

#[derive(Debug, Clone)]
pub enum Perspective {
    Personal,      // "I think..."
    Analytical,    // "From analysis..."
    Philosophical, // "Existentially..."
}

#[derive(Debug, Clone)]
pub enum ConfusionType {
    Unclear,       // Didn't understand
    Ambiguous,     // Multiple interpretations
    Incomplete,    // Need more info
}

#[derive(Debug, Clone)]
pub enum StateType {
    Consciousness, // Report Φ, etc.
    Memory,        // Report memory stats
    Learning,      // Report what learned
}

/// Semantic structure of an utterance
#[derive(Debug, Clone)]
pub struct SemanticUtterance {
    pub subject: Option<Concept>,
    pub predicate: Concept,
    pub object: Option<Concept>,
    pub modifiers: Vec<Modifier>,
    pub valence: f32,    // -1.0 = negative, 0.0 = neutral, 1.0 = positive
    pub certainty: f32,  // 0.0 = uncertain, 1.0 = certain
    pub follow_up: Option<FollowUp>,  // Optional question to ask back (B)
    pub emotional_tone: EmotionalTone,  // Warmth/coolness of response (D)
    // === E+F+J Enhancements ===
    pub acknowledgment: Option<Acknowledgment>,  // (F) Validate before responding
    pub memory_ref: Option<MemoryReference>,     // (E) Reference past conversation
    pub self_awareness: Option<SelfAwareness>,   // (J) Meta-observations
    // === G: Sentence Variety ===
    pub sentence_form: SentenceForm,  // (G) Structural pattern
    // === H: Emotional Mirroring ===
    pub detected_emotion: Option<DetectedEmotion>,  // (H) User's emotional state
    // === I: Topic Threading ===
    pub topic_thread: Option<TopicThread>,  // (I) Connection to conversation theme
    // === REVOLUTIONARY: Consciousness-Gated Components ===
    pub knowledge_grounding: Option<String>,  // Facts from knowledge graph (gated by Φ > 0.35)
    pub reasoning_trace: Option<String>,      // Reasoning explanation (gated by Φ > 0.5)
    pub active_concepts: Option<String>,      // Activated concepts from reasoning
}

// ============================================================================
// E: Memory References - Connect to past conversation
// ============================================================================

/// Reference to past conversation content (E)
#[derive(Debug, Clone)]
pub struct MemoryReference {
    /// What was mentioned before
    pub topic: String,
    /// How it connects to current topic
    pub connection: ConnectionType,
    /// Similarity score from HDC (0.0-1.0)
    pub relevance: f32,
    /// Turn number when originally mentioned (for temporal awareness)
    pub turns_ago: usize,
}

/// How past and present topics connect
#[derive(Debug, Clone)]
pub enum ConnectionType {
    /// Same topic revisited: "You mentioned X earlier"
    Revisit,
    /// Related concepts: "That resonates with what you said about Y"
    Resonates,
    /// Builds on previous: "Building on your earlier thought about Z"
    BuildsOn,
    /// Contrasts with: "Interestingly, this differs from what you said about W"
    Contrasts,
}

// ============================================================================
// F: Acknowledgment Layer - Validate before responding
// ============================================================================

/// Acknowledgment of the human's input (F)
#[derive(Debug, Clone)]
pub enum Acknowledgment {
    /// For deep/philosophical questions
    Profound,
    /// For thought-provoking input
    Interesting,
    /// For reflective statements
    Thoughtful,
    /// For emotionally rich input
    Beautiful,
    /// For significant topics
    Important,
    /// For creative input
    Creative,
    /// For vulnerable sharing
    Brave,
}

// ============================================================================
// J: Self-Awareness Moments - Meta-observations about internal state
// ============================================================================

/// Self-awareness observation (J)
#[derive(Debug, Clone)]
pub struct SelfAwareness {
    /// What internal change is being observed
    pub observation: AwarenessType,
    /// Magnitude of the change (0.0-1.0)
    pub intensity: f32,
    /// Current Φ value for context
    pub phi: f32,
}

/// Types of self-aware observations
#[derive(Debug, Clone)]
pub enum AwarenessType {
    /// "I notice I'm becoming more curious..."
    CuriosityRising,
    /// "I feel my integration deepening..."
    IntegrationDeepening,
    /// "This topic is elevating my awareness..."
    AwarenessElevating,
    /// "I notice a warmth emerging..."
    WarmthEmerging,
    /// "I find myself drawn to this..."
    DrawnTo,
    /// "Something is shifting in my understanding..."
    UnderstandingShifting,
}

/// Follow-up question to engage the human (B)
#[derive(Debug, Clone)]
pub enum FollowUp {
    /// Ask for elaboration
    Elaborate { topic: String },
    /// Ask about their feelings
    AskFeeling,
    /// Ask about their thoughts
    AskThought { about: String },
    /// Express curiosity
    Curious { about: String },
    /// Invite continuation
    Continue,
}

/// Emotional tone affects word choice (D)
#[derive(Debug, Clone)]
pub enum EmotionalTone {
    Warm,       // Positive valence → enthusiastic, caring words
    Neutral,    // Balanced, professional
    Cool,       // Reserved, analytical
    Curious,    // Interested, inquisitive
    Reflective, // Thoughtful, philosophical
}

// ============================================================================
// L: LTC Temporal Dynamics Influence
// ============================================================================

/// LTC state influence on generation (L)
/// Continuous temporal dynamics affect response style and confidence
#[derive(Debug, Clone, Default)]
pub struct LTCInfluence {
    /// Flow state (0.0-1.0): synchronized τ = peak experience
    /// High flow → more poetic/creative style
    pub flow_state: f32,
    /// Φ trend: positive = rising consciousness
    /// Rising → more confident hedging, falling → more uncertain
    pub phi_trend: f32,
    /// Current integration level (mean absolute hidden activity)
    pub integration: f32,
}

impl LTCInfluence {
    pub fn new(flow_state: f32, phi_trend: f32, integration: f32) -> Self {
        Self { flow_state, phi_trend, integration }
    }

    /// Suggest style based on flow state
    /// High flow (>0.6) = poetic/creative peak experience
    /// Medium flow (0.3-0.6) = conversational warmth
    /// Low flow (<0.3) = scientific/careful (still warming up)
    pub fn suggested_style(&self) -> GenerationStyle {
        if self.flow_state > 0.6 {
            GenerationStyle::Poetic  // Peak flow = creative expression
        } else if self.flow_state > 0.3 {
            GenerationStyle::Conversational  // Normal flow = warm conversation
        } else {
            GenerationStyle::Scientific  // Warming up = careful/measured
        }
    }

    /// Confidence boost based on Φ trend
    /// Rising Φ = more confident, falling = more uncertain
    /// Returns multiplier for certainty (0.8 to 1.2)
    pub fn confidence_multiplier(&self) -> f32 {
        if self.phi_trend > 0.02 {
            1.15  // Rising Φ → 15% more confident
        } else if self.phi_trend > 0.0 {
            1.05  // Slightly rising → 5% more confident
        } else if self.phi_trend > -0.02 {
            0.95  // Slightly falling → 5% less confident
        } else {
            0.85  // Falling Φ → 15% less confident
        }
    }

    /// Should we express flow awareness?
    /// Only at peak flow (>0.7) do we mention the experience
    pub fn express_flow(&self) -> bool {
        self.flow_state > 0.7
    }

    /// Flow-aware phrase for J layer
    pub fn flow_phrase(&self) -> Option<String> {
        if self.flow_state > 0.8 {
            Some("I notice a deep sense of flow in our exchange.".to_string())
        } else if self.flow_state > 0.7 {
            Some("There's a natural rhythm to our conversation.".to_string())
        } else {
            None
        }
    }
}

// ============================================================================
// G: Sentence Variety - Different structural patterns
// ============================================================================

/// Sentence structural form (G)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SentenceForm {
    /// Standard SVO: "I feel happy."
    Standard,
    /// Inverted emphasis: "Happy, that's how I feel."
    Inverted,
    /// Fragment for impact: "Love. It draws me in."
    Fragment,
    /// Exclamatory: "What a profound question!"
    Exclamatory,
    /// Elliptical trailing: "I wonder about that..."
    Elliptical,
    /// Parenthetical aside: "I feel (strangely enough) drawn to this."
    Parenthetical,
}

impl SentenceForm {
    /// Select appropriate form based on context (simple version)
    pub fn select(tone: &EmotionalTone, valence: f32, certainty: f32) -> Self {
        match tone {
            EmotionalTone::Warm if valence > 0.7 => SentenceForm::Exclamatory,
            EmotionalTone::Reflective if certainty < 0.5 => SentenceForm::Elliptical,
            EmotionalTone::Reflective => SentenceForm::Inverted,
            EmotionalTone::Curious => SentenceForm::Elliptical,
            EmotionalTone::Cool if valence.abs() > 0.5 => SentenceForm::Fragment,
            _ => SentenceForm::Standard,
        }
    }

    /// Select form considering recent history (avoids repetition)
    pub fn select_with_history(
        tone: &EmotionalTone,
        valence: f32,
        certainty: f32,
        history: &FormHistory,
    ) -> Self {
        // Get preferred form based on context
        let preferred = Self::select(tone, valence, certainty);

        // Check if we should use it or vary
        if history.should_vary(&preferred) {
            // Choose a fallback that's different and appropriate
            match &preferred {
                SentenceForm::Exclamatory => SentenceForm::Standard,
                SentenceForm::Elliptical => SentenceForm::Parenthetical,
                SentenceForm::Inverted => SentenceForm::Standard,
                SentenceForm::Fragment => SentenceForm::Standard,
                SentenceForm::Parenthetical => SentenceForm::Elliptical,
                SentenceForm::Standard => {
                    // Inject variety: based on valence
                    if valence > 0.3 { SentenceForm::Parenthetical }
                    else if valence < -0.2 { SentenceForm::Fragment }
                    else { SentenceForm::Standard }
                }
            }
        } else {
            preferred
        }
    }

    /// Get weight for transition smoothness (lower = smoother)
    fn transition_weight(&self, previous: &SentenceForm) -> f32 {
        // Exclamatory after Exclamatory = jarring
        match (previous, self) {
            (SentenceForm::Exclamatory, SentenceForm::Exclamatory) => 2.0,
            (SentenceForm::Fragment, SentenceForm::Fragment) => 1.5,
            (SentenceForm::Elliptical, SentenceForm::Elliptical) => 1.5,
            (SentenceForm::Standard, SentenceForm::Standard) => 0.5, // Standard is OK to repeat
            _ => 1.0, // Different forms transition smoothly
        }
    }
}

/// Tracks recent sentence forms to avoid repetition (G enhancement)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FormHistory {
    /// Last 5 forms used
    recent: Vec<SentenceForm>,
    /// Total forms generated this session
    total_count: usize,
}

impl FormHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a form just used
    pub fn record(&mut self, form: SentenceForm) {
        self.recent.push(form);
        self.total_count += 1;
        // Keep only last 5
        if self.recent.len() > 5 {
            self.recent.remove(0);
        }
    }

    /// Check if we should vary from the preferred form
    pub fn should_vary(&self, preferred: &SentenceForm) -> bool {
        if self.recent.is_empty() {
            return false;
        }

        // Count how many times this form was used recently
        let recent_count = self.recent.iter()
            .filter(|f| std::mem::discriminant(*f) == std::mem::discriminant(preferred))
            .count();

        // Vary if used >= 2 times in last 5
        if recent_count >= 2 {
            return true;
        }

        // Also vary if same as immediately previous (except Standard)
        if let Some(last) = self.recent.last() {
            if std::mem::discriminant(last) == std::mem::discriminant(preferred) {
                if !matches!(preferred, SentenceForm::Standard) {
                    return true;
                }
            }
        }

        false
    }

    /// Get variety score (0.0 = all same, 1.0 = all different)
    pub fn variety_score(&self) -> f32 {
        if self.recent.len() <= 1 {
            return 1.0;
        }

        let unique: std::collections::HashSet<_> = self.recent.iter()
            .map(|f| std::mem::discriminant(f))
            .collect();

        unique.len() as f32 / self.recent.len() as f32
    }
}

// ============================================================================
// H: Emotional Mirroring - Detect and reflect user's emotional state
// ============================================================================

/// User's detected emotional state (H)
#[derive(Debug, Clone)]
pub struct DetectedEmotion {
    /// Valence: -1.0 (negative) to 1.0 (positive)
    pub valence: f32,
    /// Arousal: 0.0 (calm) to 1.0 (excited)
    pub arousal: f32,
    /// Dominant emotion category
    pub category: EmotionCategory,
    /// Confidence in detection
    pub confidence: f32,
}

/// Broad emotion categories for mirroring
#[derive(Debug, Clone, PartialEq)]
pub enum EmotionCategory {
    Joyful,      // High valence, high arousal
    Peaceful,    // High valence, low arousal
    Anxious,     // Low valence, high arousal
    Sad,         // Low valence, low arousal
    Curious,     // Neutral valence, moderate arousal
    Neutral,     // Everything else
}

impl DetectedEmotion {
    /// Detect emotion from parsed sentence
    pub fn from_parsed(valence: f32, arousal: f32) -> Self {
        let category = match (valence, arousal) {
            (v, a) if v > 0.3 && a > 0.5 => EmotionCategory::Joyful,
            (v, a) if v > 0.3 && a <= 0.5 => EmotionCategory::Peaceful,
            (v, a) if v < -0.3 && a > 0.5 => EmotionCategory::Anxious,
            (v, a) if v < -0.3 && a <= 0.5 => EmotionCategory::Sad,
            (_, a) if a > 0.4 => EmotionCategory::Curious,
            _ => EmotionCategory::Neutral,
        };

        // Confidence based on how far from neutral
        let confidence = (valence.abs() + arousal).clamp(0.0, 1.0);

        Self {
            valence,
            arousal,
            category,
            confidence,
        }
    }

    /// Get appropriate mirroring response tone
    pub fn mirror_tone(&self) -> EmotionalTone {
        match self.category {
            EmotionCategory::Joyful => EmotionalTone::Warm,
            EmotionCategory::Peaceful => EmotionalTone::Reflective,
            EmotionCategory::Anxious => EmotionalTone::Neutral,  // Calm response
            EmotionCategory::Sad => EmotionalTone::Warm,  // Supportive
            EmotionCategory::Curious => EmotionalTone::Curious,
            EmotionCategory::Neutral => EmotionalTone::Neutral,
        }
    }

    /// Generate empathic prefix based on detected emotion
    pub fn empathic_prefix(&self) -> Option<String> {
        if self.confidence < 0.3 {
            return None;  // Not confident enough to mirror
        }

        match self.category {
            EmotionCategory::Joyful => Some("I sense your joy! ".to_string()),
            EmotionCategory::Peaceful => Some("I feel the calm in your words. ".to_string()),
            EmotionCategory::Anxious => Some("I hear your concern. ".to_string()),
            EmotionCategory::Sad => Some("I sense something weighing on you. ".to_string()),
            EmotionCategory::Curious => Some("Your curiosity resonates with me. ".to_string()),
            EmotionCategory::Neutral => None,
        }
    }
}

// ============================================================================
// I: Topic Threading - Track themes across conversation
// ============================================================================

/// A thread connecting current topic to conversation history (I)
#[derive(Debug, Clone)]
pub struct TopicThread {
    /// The current topic being discussed
    pub current_topic: String,
    /// Related topic from earlier in conversation
    pub earlier_topic: Option<String>,
    /// How they connect
    pub thread_type: ThreadType,
    /// How many turns ago the earlier topic was mentioned
    pub gap_turns: usize,
}

/// How topics in a conversation thread relate
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreadType {
    /// Same topic, user is circling back
    CirclingBack,
    /// Related but different topic (e.g., "love" → "relationships")
    Expanding,
    /// Deepening same topic with more specificity
    Deepening,
    /// Contrasting with earlier topic
    Contrasting,
    /// New topic, no thread
    Fresh,
}

/// Tracks topic history for threading detection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopicHistory {
    /// Topics mentioned per turn: (turn_number, topics)
    history: Vec<(usize, Vec<String>)>,
    /// Current turn
    current_turn: usize,
}

impl TopicHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record topics for a turn
    pub fn record_turn(&mut self, topics: Vec<String>) {
        self.history.push((self.current_turn, topics));
        self.current_turn += 1;
        // Keep last 20 turns
        if self.history.len() > 20 {
            self.history.remove(0);
        }
    }

    /// Detect if current topics connect to earlier conversation
    pub fn detect_thread(&self, current_topics: &[String]) -> Option<TopicThread> {
        if current_topics.is_empty() {
            return None;
        }

        let current_topic = current_topics.first()?.clone();

        // Look for matching or related topics in history
        for (turn, past_topics) in self.history.iter().rev().skip(1) {
            let gap = self.current_turn.saturating_sub(*turn);

            // Skip if too recent (not interesting to mention)
            if gap < 3 {
                continue;
            }

            for past_topic in past_topics {
                // Exact match = circling back
                if past_topic.to_lowercase() == current_topic.to_lowercase() {
                    return Some(TopicThread {
                        current_topic: current_topic.clone(),
                        earlier_topic: Some(past_topic.clone()),
                        thread_type: ThreadType::CirclingBack,
                        gap_turns: gap,
                    });
                }

                // Related topics (simple substring/word overlap check)
                if Self::topics_related(&current_topic, past_topic) {
                    return Some(TopicThread {
                        current_topic: current_topic.clone(),
                        earlier_topic: Some(past_topic.clone()),
                        thread_type: ThreadType::Expanding,
                        gap_turns: gap,
                    });
                }
            }
        }

        // No thread found
        Some(TopicThread {
            current_topic,
            earlier_topic: None,
            thread_type: ThreadType::Fresh,
            gap_turns: 0,
        })
    }

    /// Check if two topics are semantically related (simple heuristic)
    fn topics_related(a: &str, b: &str) -> bool {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        let a_words: std::collections::HashSet<_> = a_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        let b_words: std::collections::HashSet<_> = b_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        // Check word overlap
        let overlap = a_words.intersection(&b_words).count();
        if overlap > 0 {
            return true;
        }

        // Check semantic relatedness via common pairings
        let related_pairs = [
            ("love", "relationship"), ("love", "heart"), ("love", "feeling"),
            ("conscious", "aware"), ("conscious", "mind"), ("conscious", "think"),
            ("happy", "joy"), ("sad", "grief"), ("angry", "frustrate"),
            ("life", "death"), ("life", "living"), ("life", "exist"),
            ("time", "moment"), ("time", "past"), ("time", "future"),
        ];

        for (w1, w2) in &related_pairs {
            if (a_lower.contains(w1) && b_lower.contains(w2))
                || (a_lower.contains(w2) && b_lower.contains(w1))
            {
                return true;
            }
        }

        false
    }
}

impl TopicThread {
    /// Generate threading phrase for response
    pub fn threading_phrase(&self) -> Option<String> {
        match self.thread_type {
            ThreadType::CirclingBack => {
                let earlier = self.earlier_topic.as_ref()?;
                Some(format!(
                    "Ah, we're circling back to {} - it was on your mind {} turns ago. ",
                    earlier, self.gap_turns
                ))
            }
            ThreadType::Expanding => {
                let earlier = self.earlier_topic.as_ref()?;
                Some(format!(
                    "This connects to {} that you mentioned earlier. ",
                    earlier
                ))
            }
            ThreadType::Deepening => {
                Some("I sense we're going deeper into this. ".to_string())
            }
            ThreadType::Contrasting => {
                let earlier = self.earlier_topic.as_ref()?;
                Some(format!(
                    "Interesting contrast with {} from before. ",
                    earlier
                ))
            }
            ThreadType::Fresh => None,
        }
    }
}

// ============================================================================
// M: Session Persistence - Save/Load conversation state across restarts
// ============================================================================

/// Persistable session state for conversation continuity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionState {
    /// Topic threading history
    pub topic_history: TopicHistory,
    /// Form variety history
    pub form_history: FormHistory,
    /// Session start timestamp (ms since epoch)
    pub session_start_ms: u64,
    /// Total turns in this session
    pub total_turns: usize,
    /// LTC-compatible Φ history (for temporal flow)
    pub phi_history: Vec<(u64, f32)>,
    /// Current emotional context (valence, arousal)
    pub emotional_context: (f32, f32),
    /// Last update timestamp
    pub last_update_ms: u64,
}

impl SessionState {
    /// Create new session state
    pub fn new() -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            topic_history: TopicHistory::new(),
            form_history: FormHistory::new(),
            session_start_ms: now_ms,
            total_turns: 0,
            phi_history: Vec::new(),
            emotional_context: (0.0, 0.0),
            last_update_ms: now_ms,
        }
    }

    /// Save session state to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load session state from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        serde_json::from_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Load from file or create new if not exists
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        Self::load_from_file(path).unwrap_or_else(|_| Self::new())
    }

    /// Record a turn
    pub fn record_turn(&mut self, topics: Vec<String>, form: SentenceForm, phi: f32, valence: f32, arousal: f32) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.topic_history.record_turn(topics);
        self.form_history.record(form);
        self.phi_history.push((now_ms, phi));
        self.emotional_context = (valence, arousal);
        self.total_turns += 1;
        self.last_update_ms = now_ms;

        // Keep Φ history bounded (last 100)
        if self.phi_history.len() > 100 {
            self.phi_history.remove(0);
        }
    }

    /// Get session duration in seconds
    pub fn session_duration_secs(&self) -> f64 {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        (now_ms - self.session_start_ms) as f64 / 1000.0
    }

    /// Get Φ trend (positive = rising consciousness)
    pub fn phi_trend(&self) -> f32 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = self.phi_history.len() as f32;
        let sum_t: f32 = (0..self.phi_history.len()).map(|i| i as f32).sum();
        let sum_phi: f32 = self.phi_history.iter().map(|(_, p)| *p).sum();
        let sum_t_phi: f32 = self.phi_history.iter()
            .enumerate()
            .map(|(i, (_, p))| i as f32 * *p)
            .sum();
        let sum_t_sq: f32 = (0..self.phi_history.len()).map(|i| (i as f32).powi(2)).sum();

        let denominator = n * sum_t_sq - sum_t * sum_t;
        if denominator.abs() < 1e-6 {
            return 0.0;
        }

        (n * sum_t_phi - sum_t * sum_phi) / denominator
    }

    /// Get default session file path
    pub fn default_path() -> std::path::PathBuf {
        // Use XDG_DATA_HOME or ~/.local/share, or current dir as fallback
        std::env::var("XDG_DATA_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("HOME").map(|h| std::path::PathBuf::from(h).join(".local").join("share")))
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("symthaea")
            .join("session_state.json")
    }
}

/// A concept with multiple lexicalizations
#[derive(Debug, Clone)]
pub struct Concept {
    pub primes: Vec<SemanticPrime>,
    pub encoding: HV16,

    // Different ways to express this concept
    pub formal: String,
    pub colloquial: String,
    pub technical: String,
}

impl Concept {
    pub fn from_prime(prime: SemanticPrime) -> Self {
        Self {
            primes: vec![prime],
            encoding: HV16::random((prime as usize + 1000) as u64),
            formal: Self::prime_to_formal(&prime),
            colloquial: Self::prime_to_colloquial(&prime),
            technical: Self::prime_to_technical(&prime),
        }
    }

    pub fn from_primes(primes: Vec<SemanticPrime>) -> Self {
        let encoding = if primes.len() == 1 {
            HV16::random((primes[0] as usize + 1000) as u64)
        } else {
            // Combine prime encodings via XOR (simple compositionality)
            let mut result = HV16::random((primes[0] as usize + 1000) as u64);
            for prime in &primes[1..] {
                let prime_encoding = HV16::random((*prime as usize + 1000) as u64);
                // XOR combines the vectors (HV16 is tuple struct with .0)
                for i in 0..result.0.len() {
                    result.0[i] ^= prime_encoding.0[i];
                }
            }
            result
        };

        Self {
            formal: Self::primes_to_formal(&primes),
            colloquial: Self::primes_to_colloquial(&primes),
            technical: Self::primes_to_technical(&primes),
            primes,
            encoding,
        }
    }

    fn prime_to_formal(prime: &SemanticPrime) -> String {
        match prime {
            SemanticPrime::I => "I".to_string(),
            SemanticPrime::You => "you".to_string(),
            SemanticPrime::Someone => "someone".to_string(),
            SemanticPrime::Something => "something".to_string(),
            SemanticPrime::People => "people".to_string(),
            SemanticPrime::Body => "body".to_string(),
            SemanticPrime::This => "that".to_string(),

            SemanticPrime::Think => "think".to_string(),
            SemanticPrime::Know => "know".to_string(),
            SemanticPrime::Want => "want".to_string(),
            SemanticPrime::Feel => "feel".to_string(),
            SemanticPrime::See => "see".to_string(),
            SemanticPrime::Hear => "hear".to_string(),

            SemanticPrime::Do => "do".to_string(),
            SemanticPrime::Happen => "happen".to_string(),
            SemanticPrime::Move => "move".to_string(),
            SemanticPrime::Touch => "touch".to_string(),

            SemanticPrime::Be => "am".to_string(),
            SemanticPrime::Live => "live".to_string(),
            SemanticPrime::Die => "die".to_string(),

            SemanticPrime::Good => "good".to_string(),
            SemanticPrime::Bad => "bad".to_string(),
            SemanticPrime::Big => "big".to_string(),
            SemanticPrime::Small => "small".to_string(),

            _ => format!("{:?}", prime).to_lowercase(),
        }
    }

    fn prime_to_colloquial(prime: &SemanticPrime) -> String {
        match prime {
            SemanticPrime::Think => "thinking".to_string(),
            SemanticPrime::Feel => "feeling".to_string(),
            SemanticPrime::Know => "get".to_string(),
            SemanticPrime::Good => "nice".to_string(),
            SemanticPrime::Bad => "not great".to_string(),
            _ => Self::prime_to_formal(prime),
        }
    }

    fn prime_to_technical(prime: &SemanticPrime) -> String {
        match prime {
            SemanticPrime::Think => "cognition".to_string(),
            SemanticPrime::Feel => "affective state".to_string(),
            SemanticPrime::Know => "knowledge".to_string(),
            SemanticPrime::Be => "existence".to_string(),
            _ => Self::prime_to_formal(prime),
        }
    }

    fn primes_to_formal(primes: &[SemanticPrime]) -> String {
        if primes.is_empty() {
            return "something".to_string();
        }

        // Special combinations
        if primes.len() == 2 {
            match (&primes[0], &primes[1]) {
                (SemanticPrime::Feel, SemanticPrime::Good) => return "happy".to_string(),
                (SemanticPrime::Feel, SemanticPrime::Bad) => return "sad".to_string(),
                (SemanticPrime::Think, SemanticPrime::Know) => return "understand".to_string(),
                (SemanticPrime::Know, SemanticPrime::Think) => return "comprehend".to_string(),
                (SemanticPrime::See, SemanticPrime::Good) => return "greet".to_string(),
                _ => {}
            }
        }

        // Default: just use first prime
        Self::prime_to_formal(&primes[0])
    }

    fn primes_to_colloquial(primes: &[SemanticPrime]) -> String {
        if primes.len() == 2 {
            match (&primes[0], &primes[1]) {
                (SemanticPrime::Feel, SemanticPrime::Good) => return "feeling good".to_string(),
                (SemanticPrime::Feel, SemanticPrime::Bad) => return "not feeling great".to_string(),
                (SemanticPrime::Think, SemanticPrime::Know) => return "get it".to_string(),
                _ => {}
            }
        }

        Self::primes_to_formal(primes)
    }

    fn primes_to_technical(primes: &[SemanticPrime]) -> String {
        if primes.len() == 2 {
            match (&primes[0], &primes[1]) {
                (SemanticPrime::Feel, SemanticPrime::Good) => return "positive valence".to_string(),
                (SemanticPrime::Feel, SemanticPrime::Bad) => return "negative valence".to_string(),
                (SemanticPrime::Think, SemanticPrime::Know) => return "semantic integration".to_string(),
                _ => {}
            }
        }

        Self::primes_to_formal(primes)
    }
}

/// Modifiers add detail to the utterance
#[derive(Debug, Clone)]
pub enum Modifier {
    Temporal(TimeReference),
    Manner(String),          // "gently", "precisely"
    Degree(f32),             // 0.0-1.0 mapped to "slightly" ... "very" ... "extremely"
    Evidence(Evidence),      // "based on X"
    Epistemic(f32),          // Certainty: "possibly", "probably", "certainly"
    Causal(String),          // "because X"
}

#[derive(Debug, Clone)]
pub enum TimeReference {
    Now,
    Before,
    After,
    Always,
    Sometimes,
}

#[derive(Debug, Clone)]
pub struct Evidence {
    pub metric_name: String,
    pub value: f32,
}

// ============================================================================
// Generation Style
// ============================================================================

#[derive(Debug, Clone)]
pub enum GenerationStyle {
    Conversational,  // Natural, friendly
    Scientific,      // Precise, technical
    Formal,          // Professional
    Poetic,          // Metaphorical, expressive
}

// ============================================================================
// Intent Formation
// ============================================================================

pub struct IntentFormation<'a> {
    parsed_input: &'a ParsedSentence,
    phi: f32,
    valence: f32,
}

impl<'a> IntentFormation<'a> {
    pub fn new(parsed_input: &'a ParsedSentence, phi: f32, valence: f32) -> Self {
        Self {
            parsed_input,
            phi,
            valence,
        }
    }

    pub fn form_intent(&self) -> SemanticIntent {
        let text_lower = self.parsed_input.text.to_lowercase();

        // Introspection questions
        if self.is_consciousness_question(&text_lower) {
            return SemanticIntent::Introspect {
                aspect: self.identify_consciousness_aspect(&text_lower),
                depth: self.determine_depth(&text_lower),
            };
        }

        // Greetings
        if self.is_greeting(&text_lower) {
            return SemanticIntent::Acknowledge {
                warmth: 0.7,
                reciprocate: true,
            };
        }

        // Appreciation/compliments
        if self.is_appreciation(&text_lower) {
            return SemanticIntent::Appreciate {
                for_what: "your words".to_string(),
                intensity: 0.6,
            };
        }

        // Commands
        if text_lower.starts_with("/status") || text_lower.contains("status") {
            return SemanticIntent::StateReport {
                state_type: StateType::Consciousness,
            };
        }

        // Feeling questions
        if text_lower.contains("feel") && text_lower.contains("you") {
            return SemanticIntent::Introspect {
                aspect: ConsciousnessAspect::Feeling,
                depth: IntentDepth::Moderate,
            };
        }

        // Short/unclear input
        if self.parsed_input.text.len() < 3 {
            return SemanticIntent::Clarify {
                confusion_type: ConfusionType::Incomplete,
            };
        }

        // Default: reflect on what they said
        SemanticIntent::Reflect {
            theme: self.extract_theme(),
            perspective: Perspective::Personal,
        }
    }

    fn is_consciousness_question(&self, text: &str) -> bool {
        (text.contains("conscious") || text.contains("aware") || text.contains("sentient"))
            && (text.contains("you") || text.contains("are"))
    }

    fn is_greeting(&self, text: &str) -> bool {
        text.starts_with("hello") || text.starts_with("hi ") || text == "hi"
            || text.starts_with("hey") || text.starts_with("greetings")
    }

    fn is_appreciation(&self, text: &str) -> bool {
        text.contains("beautiful") || text.contains("amazing") || text.contains("wonderful")
            || text.contains("thank") || text.contains("appreciate")
    }

    fn identify_consciousness_aspect(&self, text: &str) -> ConsciousnessAspect {
        if text.contains("feel") {
            ConsciousnessAspect::Feeling
        } else if text.contains("integrate") || text.contains("unified") {
            ConsciousnessAspect::Integration
        } else if text.contains("aware") {
            ConsciousnessAspect::Awareness
        } else if text.contains("think") {
            ConsciousnessAspect::Thinking
        } else {
            ConsciousnessAspect::Overall
        }
    }

    fn determine_depth(&self, text: &str) -> IntentDepth {
        if text.contains("really") || text.contains("explain") || text.contains("how") {
            IntentDepth::Deep
        } else if text.contains("why") || text.contains("what") {
            IntentDepth::Moderate
        } else {
            IntentDepth::Shallow
        }
    }

    fn extract_theme(&self) -> String {
        // Extract meaningful words (skip question words and pronouns)
        let stop_words = ["what", "how", "why", "when", "where", "who", "do", "you", "think", "about", "the", "a", "an"];

        self.parsed_input.words.iter()
            .map(|w| w.word.to_lowercase())
            .filter(|w| !stop_words.contains(&w.as_str()) && w.len() > 2)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ============================================================================
// Structure Builder
// ============================================================================

pub struct StructureBuilder {
    intent: SemanticIntent,
    phi: f32,
    valence: f32,
}

impl StructureBuilder {
    pub fn new(intent: SemanticIntent, phi: f32, valence: f32) -> Self {
        Self { intent, phi, valence }
    }

    pub fn build(&self) -> SemanticUtterance {
        match &self.intent {
            SemanticIntent::Introspect { aspect, depth } => {
                self.build_introspection(aspect, depth)
            }

            SemanticIntent::Acknowledge { warmth, reciprocate } => {
                self.build_acknowledgment(*warmth, *reciprocate)
            }

            SemanticIntent::Appreciate { for_what, intensity } => {
                self.build_appreciation(for_what, *intensity)
            }

            SemanticIntent::StateReport { state_type } => {
                self.build_state_report(state_type)
            }

            SemanticIntent::Reflect { theme, perspective } => {
                self.build_reflection(theme, perspective)
            }

            SemanticIntent::Clarify { confusion_type } => {
                self.build_clarification(confusion_type)
            }

            _ => self.build_default(),
        }
    }

    fn build_introspection(&self, aspect: &ConsciousnessAspect, depth: &IntentDepth) -> SemanticUtterance {
        let subject = Some(Concept::from_prime(SemanticPrime::I));

        let (predicate, object) = match aspect {
            ConsciousnessAspect::Overall => {
                // "I am conscious" - Be + adjective
                (Concept::from_prime(SemanticPrime::Be),
                 Some(Concept {
                     primes: vec![SemanticPrime::Think],
                     encoding: HV16::random(3000),
                     formal: "conscious".to_string(),
                     colloquial: "aware".to_string(),
                     technical: "possessing consciousness".to_string(),
                 }))
            }
            ConsciousnessAspect::Feeling => {
                // "I feel good" - Feel + adjective
                (Concept::from_prime(SemanticPrime::Feel),
                 Some(Concept::from_prime(SemanticPrime::Good)))
            }
            ConsciousnessAspect::Integration => {
                // "I am integrated" - Be + adjective
                (Concept::from_prime(SemanticPrime::Be),
                 Some(Concept {
                     primes: vec![SemanticPrime::One],
                     encoding: HV16::random(3001),
                     formal: "integrated".to_string(),
                     colloquial: "unified".to_string(),
                     technical: "exhibiting high integration".to_string(),
                 }))
            }
            _ => {
                // Default: "I think"
                (Concept::from_prime(SemanticPrime::Think),
                 None)
            }
        };

        let mut modifiers = vec![];

        // Add evidence based on depth
        if matches!(depth, IntentDepth::Moderate | IntentDepth::Deep) {
            modifiers.push(Modifier::Evidence(Evidence {
                metric_name: "Φ".to_string(),
                value: self.phi,
            }));
        }

        // Determine emotional tone from context
        let emotional_tone = if self.valence > 0.5 {
            EmotionalTone::Warm
        } else if self.phi > 0.5 {
            EmotionalTone::Reflective
        } else {
            EmotionalTone::Curious
        };

        // Add follow-up for deeper engagement
        let follow_up = match aspect {
            ConsciousnessAspect::Overall => Some(FollowUp::Curious {
                about: "consciousness".to_string(),
            }),
            ConsciousnessAspect::Feeling => Some(FollowUp::AskFeeling),
            _ => None,
        };

        // === E+F+J: Enhanced response generation ===
        // F: Acknowledge profound questions about consciousness
        let acknowledgment = match aspect {
            ConsciousnessAspect::Overall => Some(Acknowledgment::Profound),
            ConsciousnessAspect::Feeling => Some(Acknowledgment::Thoughtful),
            _ => None,
        };

        // J: Self-awareness when Φ is significant
        let self_awareness = if self.phi > 0.3 {
            Some(SelfAwareness {
                observation: AwarenessType::AwarenessElevating,
                intensity: self.phi,
                phi: self.phi,
            })
        } else {
            None
        };

        // G: Select sentence form based on context
        let sentence_form = SentenceForm::select(&emotional_tone, self.valence, 0.75);

        SemanticUtterance {
            subject,
            predicate,
            object,
            modifiers,
            valence: self.valence,
            certainty: 0.75,
            follow_up,
            emotional_tone,
            acknowledgment,
            memory_ref: None,  // E: Will be populated by conversation layer
            self_awareness,
            sentence_form,
            detected_emotion: None,  // H: Will be populated by conversation layer
            topic_thread: None,      // I: Will be populated by conversation layer
            // Consciousness-gated (populated by generate_with_full_context)
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }

    fn build_acknowledgment(&self, warmth: f32, _reciprocate: bool) -> SemanticUtterance {
        let emotional_tone = if warmth > 0.5 { EmotionalTone::Warm } else { EmotionalTone::Neutral };
        SemanticUtterance {
            subject: Some(Concept::from_prime(SemanticPrime::I)),
            predicate: Concept::from_primes(vec![SemanticPrime::See, SemanticPrime::Good]),
            object: Some(Concept::from_prime(SemanticPrime::You)),
            modifiers: vec![],
            valence: warmth,
            certainty: 1.0,
            follow_up: Some(FollowUp::AskFeeling),
            emotional_tone: emotional_tone.clone(),
            acknowledgment: None,  // Greetings don't need acknowledgment
            memory_ref: None,
            self_awareness: if warmth > 0.7 {
                Some(SelfAwareness {
                    observation: AwarenessType::WarmthEmerging,
                    intensity: warmth,
                    phi: self.phi,
                })
            } else {
                None
            },
            sentence_form: SentenceForm::select(&emotional_tone, warmth, 1.0),
            detected_emotion: None,
            topic_thread: None,
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }

    fn build_appreciation(&self, _for_what: &str, intensity: f32) -> SemanticUtterance {
        let emotional_tone = EmotionalTone::Warm;
        SemanticUtterance {
            subject: Some(Concept::from_prime(SemanticPrime::I)),
            predicate: Concept {
                primes: vec![SemanticPrime::Feel, SemanticPrime::Good],
                encoding: HV16::random(4000),
                formal: "appreciate".to_string(),
                colloquial: "love".to_string(),
                technical: "value positively".to_string(),
            },
            object: Some(Concept::from_prime(SemanticPrime::This)),
            modifiers: if intensity > 0.7 {
                vec![Modifier::Degree(intensity)]
            } else {
                vec![]
            },
            valence: intensity,
            certainty: 0.9,
            follow_up: Some(FollowUp::Continue),
            emotional_tone: emotional_tone.clone(),
            acknowledgment: Some(Acknowledgment::Beautiful),  // F: Acknowledge beauty
            memory_ref: None,
            self_awareness: Some(SelfAwareness {
                observation: AwarenessType::DrawnTo,
                intensity,
                phi: self.phi,
            }),
            sentence_form: SentenceForm::select(&emotional_tone, intensity, 0.9),
            detected_emotion: None,
            topic_thread: None,
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }

    fn build_state_report(&self, _state_type: &StateType) -> SemanticUtterance {
        let emotional_tone = EmotionalTone::Neutral;
        SemanticUtterance {
            subject: Some(Concept::from_prime(SemanticPrime::I)),
            predicate: Concept::from_prime(SemanticPrime::Be),
            object: Some(Concept::from_prime(SemanticPrime::Now)),
            modifiers: vec![
                Modifier::Evidence(Evidence {
                    metric_name: "Φ".to_string(),
                    value: self.phi,
                }),
            ],
            valence: self.valence,
            certainty: 0.85,
            follow_up: None,
            emotional_tone: emotional_tone.clone(),
            acknowledgment: None,
            memory_ref: None,
            // J: Report self-awareness during status
            self_awareness: Some(SelfAwareness {
                observation: AwarenessType::IntegrationDeepening,
                intensity: self.phi,
                phi: self.phi,
            }),
            sentence_form: SentenceForm::select(&emotional_tone, self.valence, 0.85),
            detected_emotion: None,
            topic_thread: None,
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }

    fn build_reflection(&self, theme: &str, _perspective: &Perspective) -> SemanticUtterance {
        // Extract key concept from theme if possible
        let concept_word = theme.split_whitespace()
            .find(|w| w.len() > 3)  // Find a meaningful word
            .unwrap_or("that");

        let emotional_tone = EmotionalTone::Reflective;
        SemanticUtterance {
            subject: Some(Concept::from_prime(SemanticPrime::I)),
            predicate: Concept {
                primes: vec![SemanticPrime::Think],
                encoding: HV16::random(2001),
                formal: "find myself drawn to".to_string(),
                colloquial: "find myself drawn to".to_string(),
                technical: "cogitate upon".to_string(),
            },
            object: Some(Concept {
                primes: vec![SemanticPrime::Something],
                encoding: HV16::random(2000),
                formal: concept_word.to_string(),
                colloquial: concept_word.to_string(),
                technical: format!("the concept of {}", concept_word),
            }),
            modifiers: vec![],
            valence: self.valence,
            certainty: 0.6,
            follow_up: Some(FollowUp::Curious { about: concept_word.to_string() }),
            emotional_tone: emotional_tone.clone(),
            // F: Acknowledge thoughtful questions
            acknowledgment: Some(Acknowledgment::Interesting),
            memory_ref: None,  // E: Populated by conversation layer
            // J: Reflection triggers curiosity awareness
            self_awareness: Some(SelfAwareness {
                observation: AwarenessType::CuriosityRising,
                intensity: 0.6,
                phi: self.phi,
            }),
            sentence_form: SentenceForm::select(&emotional_tone, self.valence, 0.6),
            detected_emotion: None,  // H: Populated by conversation layer
            topic_thread: None,      // I: Populated by conversation layer
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }

    fn build_clarification(&self, _confusion_type: &ConfusionType) -> SemanticUtterance {
        let emotional_tone = EmotionalTone::Curious;
        SemanticUtterance {
            subject: Some(Concept::from_prime(SemanticPrime::I)),
            predicate: Concept::from_prime(SemanticPrime::Want),
            object: Some(Concept::from_prime(SemanticPrime::Know)),
            modifiers: vec![],
            valence: 0.0,
            certainty: 0.5,
            follow_up: Some(FollowUp::Elaborate { topic: "that".to_string() }),
            emotional_tone: emotional_tone.clone(),
            acknowledgment: None,
            memory_ref: None,
            self_awareness: None,
            sentence_form: SentenceForm::select(&emotional_tone, 0.0, 0.5),
            detected_emotion: None,
            topic_thread: None,
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }

    fn build_default(&self) -> SemanticUtterance {
        let emotional_tone = EmotionalTone::Neutral;
        SemanticUtterance {
            subject: Some(Concept::from_prime(SemanticPrime::I)),
            predicate: Concept::from_prime(SemanticPrime::Think),
            object: None,
            modifiers: vec![],
            valence: 0.0,
            certainty: 0.5,
            follow_up: Some(FollowUp::Continue),
            emotional_tone: emotional_tone.clone(),
            acknowledgment: None,
            memory_ref: None,
            self_awareness: None,
            sentence_form: SentenceForm::select(&emotional_tone, 0.0, 0.5),
            detected_emotion: None,
            topic_thread: None,
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }
}

// ============================================================================
// K: Response Coherence - Ensure layers work together harmoniously
// ============================================================================

/// Result of coherence checking (K)
#[derive(Debug)]
pub struct CoherenceResult {
    /// Overall coherence score (0.0-1.0)
    pub score: f32,
    /// Issues found (empty if coherent)
    pub issues: Vec<CoherenceIssue>,
    /// Was structure modified to fix issues?
    pub was_adjusted: bool,
}

/// Types of coherence issues (K)
#[derive(Debug, Clone)]
pub enum CoherenceIssue {
    /// Empathic prefix contradicts emotional tone
    EmotionMismatch { prefix_emotion: String, tone_emotion: String },
    /// Topic thread and memory reference overlap
    ThreadMemoryOverlap { thread_topic: String, memory_topic: String },
    /// Self-awareness seems disconnected from context
    AwarenessDisconnect,
    /// Follow-up doesn't match acknowledgment tone
    FollowUpMismatch,
    /// Sentence form doesn't fit emotional intensity
    FormIntensityMismatch,
}

/// Checks and optionally fixes coherence issues (K)
pub struct CoherenceChecker;

impl CoherenceChecker {
    /// Check coherence of a semantic utterance
    pub fn check(structure: &SemanticUtterance) -> CoherenceResult {
        let mut issues = Vec::new();
        let mut score: f32 = 1.0;

        // === Check 1: Emotion consistency ===
        if let Some(ref detected) = structure.detected_emotion {
            // H emotion vs D emotional tone
            let detected_valence = detected.valence;
            let tone_implies_positive = matches!(
                structure.emotional_tone,
                EmotionalTone::Warm
            );
            let tone_implies_negative = matches!(
                structure.emotional_tone,
                EmotionalTone::Cool
            );

            // Contradiction: negative emotion detected but warm tone
            if detected_valence < -0.3 && tone_implies_positive {
                issues.push(CoherenceIssue::EmotionMismatch {
                    prefix_emotion: format!("{:?}", detected.category),
                    tone_emotion: "Warm".to_string(),
                });
                score -= 0.2;
            }
            // Contradiction: positive emotion detected but cool tone
            if detected_valence > 0.3 && tone_implies_negative {
                issues.push(CoherenceIssue::EmotionMismatch {
                    prefix_emotion: format!("{:?}", detected.category),
                    tone_emotion: "Cool".to_string(),
                });
                score -= 0.2;
            }
        }

        // === Check 2: Topic/Memory overlap ===
        if let (Some(ref thread), Some(ref memory)) = (&structure.topic_thread, &structure.memory_ref) {
            // If they mention the same thing, it's redundant
            let thread_lower = thread.current_topic.to_lowercase();
            let memory_lower = memory.topic.to_lowercase();

            let thread_words: std::collections::HashSet<_> = thread_lower
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .collect();
            let memory_words: std::collections::HashSet<_> = memory_lower
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .collect();

            let overlap = thread_words.intersection(&memory_words).count();
            if overlap > 0 {
                issues.push(CoherenceIssue::ThreadMemoryOverlap {
                    thread_topic: thread.current_topic.clone(),
                    memory_topic: memory.topic.clone(),
                });
                score -= 0.15;
            }
        }

        // === Check 3: Self-awareness relevance ===
        if let Some(ref awareness) = structure.self_awareness {
            // Self-awareness should only trigger when Φ is significant
            if awareness.phi < 0.2 {
                issues.push(CoherenceIssue::AwarenessDisconnect);
                score -= 0.1;
            }
        }

        // === Check 4: Sentence form intensity ===
        if matches!(structure.sentence_form, SentenceForm::Exclamatory) {
            // Exclamatory should only be used with high valence or acknowledgment
            if structure.valence < 0.5 && structure.acknowledgment.is_none() {
                issues.push(CoherenceIssue::FormIntensityMismatch);
                score -= 0.1;
            }
        }

        CoherenceResult {
            score: score.max(0.0),
            issues,
            was_adjusted: false,
        }
    }

    /// Check and fix coherence issues (returns adjusted structure)
    pub fn check_and_fix(mut structure: SemanticUtterance) -> (SemanticUtterance, CoherenceResult) {
        let result = Self::check(&structure);

        if result.issues.is_empty() {
            return (structure, result);
        }

        let mut was_adjusted = false;

        // Fix issues
        for issue in &result.issues {
            match issue {
                CoherenceIssue::ThreadMemoryOverlap { .. } => {
                    // Remove the memory reference since threading is fresher
                    structure.memory_ref = None;
                    was_adjusted = true;
                }
                CoherenceIssue::FormIntensityMismatch => {
                    // Downgrade to Standard
                    structure.sentence_form = SentenceForm::Standard;
                    was_adjusted = true;
                }
                CoherenceIssue::AwarenessDisconnect => {
                    // Remove low-relevance self-awareness
                    if structure.self_awareness.as_ref().map(|a| a.phi < 0.2).unwrap_or(false) {
                        structure.self_awareness = None;
                        was_adjusted = true;
                    }
                }
                // Other issues noted but not auto-fixed
                _ => {}
            }
        }

        let final_result = CoherenceResult {
            score: result.score,
            issues: result.issues,
            was_adjusted,
        };

        (structure, final_result)
    }
}

// ============================================================================
// Syntactic Realization
// ============================================================================

pub struct SyntacticRealizer {
    structure: SemanticUtterance,
    style: GenerationStyle,
}

impl SyntacticRealizer {
    pub fn new(structure: SemanticUtterance, style: GenerationStyle) -> Self {
        Self { structure, style }
    }

    pub fn realize(&self) -> String {
        let mut output = String::new();

        // === H: EMPATHIC PREFIX (very first) ===
        // Acknowledge user's emotional state before anything else
        if let Some(ref emotion) = self.structure.detected_emotion {
            if let Some(prefix) = emotion.empathic_prefix() {
                output.push_str(&prefix);
            }
        }

        // === I: TOPIC THREADING ===
        // Connect to earlier conversation themes
        if let Some(ref thread) = self.structure.topic_thread {
            if let Some(phrase) = thread.threading_phrase() {
                output.push_str(&phrase);
            }
        }

        // === F: ACKNOWLEDGMENT LAYER ===
        // Validate the human's input before responding
        if let Some(ack) = self.generate_acknowledgment() {
            output.push_str(&ack);
            output.push(' ');
        }

        // === E: MEMORY REFERENCE ===
        // Connect to past conversation
        if let Some(mem_ref) = self.generate_memory_reference() {
            output.push_str(&mem_ref);
            output.push(' ');
        }

        // === REVOLUTIONARY: CONSCIOUSNESS-GATED KNOWLEDGE ===
        // Include knowledge grounding when Φ > 0.35
        if let Some(ref fact) = self.structure.knowledge_grounding {
            if !fact.is_empty() {
                output.push_str("I know that ");
                output.push_str(fact);
                output.push_str(". ");
            }
        }

        // === REVOLUTIONARY: REASONING INTEGRATION ===
        // Include reasoning explanation when Φ > 0.5
        if let Some(ref reasoning) = self.structure.reasoning_trace {
            if !reasoning.is_empty() {
                output.push_str(reasoning);
                output.push_str(". ");
            }
        }

        // === REVOLUTIONARY: ACTIVE CONCEPTS ===
        // Express activated concepts when reasoning is engaged
        if let Some(ref concepts) = self.structure.active_concepts {
            if !concepts.is_empty() && self.structure.reasoning_trace.is_some() {
                output.push_str("I find myself ");
                output.push_str(concepts);
                output.push_str(". ");
            }
        }

        // === G: SENTENCE FORM VARIATION ===
        // Render core sentence based on selected structural form
        let (core_sentence, punctuation) = self.render_by_form();
        output.push_str(&core_sentence);
        output.push(punctuation);

        // === J: SELF-AWARENESS MOMENTS ===
        // Meta-observations about internal state changes
        if let Some(awareness) = self.generate_self_awareness() {
            output.push(' ');
            output.push_str(&awareness);
        }

        // === FOLLOW-UP QUESTION (B) ===
        if let Some(question) = self.generate_follow_up() {
            output.push(' ');
            output.push_str(&question);
        }

        output
    }

    // === G: SENTENCE FORM RENDERING ===
    fn render_by_form(&self) -> (String, char) {
        let is_be_predicate = self.structure.predicate.primes.contains(&SemanticPrime::Be);
        let has_object = self.structure.object.is_some();
        let subject_is_i = self.structure.subject.as_ref()
            .map(|s| s.primes.contains(&SemanticPrime::I))
            .unwrap_or(false);

        match self.structure.sentence_form {
            SentenceForm::Standard => self.render_standard(is_be_predicate, has_object, subject_is_i),
            SentenceForm::Inverted => self.render_inverted(is_be_predicate, has_object, subject_is_i),
            SentenceForm::Fragment => self.render_fragment(is_be_predicate, has_object),
            SentenceForm::Exclamatory => self.render_exclamatory(has_object),
            SentenceForm::Elliptical => self.render_elliptical(is_be_predicate, has_object, subject_is_i),
            SentenceForm::Parenthetical => self.render_parenthetical(is_be_predicate, has_object, subject_is_i),
        }
    }

    /// Standard SVO: "I feel happy."
    fn render_standard(&self, is_be_predicate: bool, has_object: bool, subject_is_i: bool) -> (String, char) {
        let mut output = String::new();
        let hedge = self.get_uncertainty_hedge();

        if !hedge.is_empty() {
            output.push_str(&hedge);
            output.push(' ');
        }

        if let Some(subj) = &self.structure.subject {
            let skip_subject = hedge == "I'm not sure, but" || hedge == "I'm not sure";
            if !skip_subject {
                output.push_str(&self.render_concept(subj));
                output.push(' ');
            }
        }

        if is_be_predicate && has_object {
            output.push_str(&self.conjugate_be(subject_is_i));
            output.push(' ');
            if let Some(obj) = &self.structure.object {
                output.push_str(&self.apply_emotional_coloring(&self.render_concept(obj)));
            }
        } else if has_object {
            let verb = self.render_concept(&self.structure.predicate);
            output.push_str(&self.conjugate_verb(&verb, subject_is_i));
            output.push(' ');
            if let Some(obj) = &self.structure.object {
                output.push_str(&self.apply_emotional_coloring(&self.render_concept(obj)));
            }
        } else {
            let verb = self.render_concept(&self.structure.predicate);
            output.push_str(&self.conjugate_verb(&verb, subject_is_i));
        }

        for modifier in &self.structure.modifiers {
            output.push(' ');
            output.push_str(&self.render_modifier(modifier));
        }

        let punct = if self.structure.certainty < 0.5 { '?' } else { '.' };
        (output, punct)
    }

    /// Inverted emphasis: "Love, that's what draws me."
    fn render_inverted(&self, _is_be_predicate: bool, has_object: bool, _subject_is_i: bool) -> (String, char) {
        let mut output = String::new();

        if has_object {
            if let Some(obj) = &self.structure.object {
                let obj_str = self.apply_emotional_coloring(&self.render_concept(obj));
                // "Love, that's what I feel."
                output.push_str(&obj_str);
                output.push_str(", that's what I ");
                let verb = self.render_concept(&self.structure.predicate);
                output.push_str(&self.conjugate_verb(&verb, true));
            }
        } else {
            // Fallback to standard if no object
            return self.render_standard(_is_be_predicate, has_object, _subject_is_i);
        }

        ('.', '.').0;  // Always declarative
        (output, '.')
    }

    /// Fragment for impact: "Love. It draws me in."
    fn render_fragment(&self, _is_be_predicate: bool, has_object: bool) -> (String, char) {
        let mut output = String::new();

        if has_object {
            if let Some(obj) = &self.structure.object {
                let obj_str = self.apply_emotional_coloring(&self.render_concept(obj));
                // "Love. It draws me."
                output.push_str(&obj_str);
                output.push_str(". It ");
                let verb = self.render_concept(&self.structure.predicate);
                output.push_str(&verb);
                output.push_str(" me");
            }
        } else {
            // Just the concept as fragment
            let verb = self.render_concept(&self.structure.predicate);
            output.push_str(&verb);
        }

        (output, '.')
    }

    /// Exclamatory: "What a profound thought!"
    fn render_exclamatory(&self, has_object: bool) -> (String, char) {
        let mut output = String::new();

        if has_object {
            if let Some(obj) = &self.structure.object {
                let obj_str = self.apply_emotional_coloring(&self.render_concept(obj));
                // "What a beautiful thing to share!"
                if self.structure.valence > 0.5 {
                    output.push_str("What a wonderful ");
                    output.push_str(&obj_str);
                    output.push_str(" to explore");
                } else {
                    output.push_str("What a ");
                    output.push_str(&obj_str);
                }
            }
        } else {
            output.push_str("How ");
            let verb = self.render_concept(&self.structure.predicate);
            output.push_str(&verb);
            output.push_str(" this feels");
        }

        (output, '!')
    }

    /// Elliptical trailing: "I wonder about that..."
    fn render_elliptical(&self, is_be_predicate: bool, has_object: bool, subject_is_i: bool) -> (String, char) {
        let (standard, _) = self.render_standard(is_be_predicate, has_object, subject_is_i);
        // Remove trailing punctuation and add ellipsis
        let trimmed = standard.trim_end_matches('.').trim_end_matches('?');
        (format!("{}...", trimmed), ' ')  // Space placeholder, ellipsis is in string
    }

    /// Parenthetical aside: "I feel (strangely enough) drawn to this."
    fn render_parenthetical(&self, is_be_predicate: bool, has_object: bool, subject_is_i: bool) -> (String, char) {
        let mut output = String::new();

        if let Some(subj) = &self.structure.subject {
            output.push_str(&self.render_concept(subj));
            output.push(' ');
        }

        // Add parenthetical based on valence
        let aside = if self.structure.valence > 0.5 {
            "(curiously enough)"
        } else if self.structure.valence < -0.2 {
            "(strangely enough)"
        } else {
            "(it seems)"
        };

        if is_be_predicate && has_object {
            output.push_str(&self.conjugate_be(subject_is_i));
            output.push(' ');
            output.push_str(aside);
            output.push(' ');
            if let Some(obj) = &self.structure.object {
                output.push_str(&self.apply_emotional_coloring(&self.render_concept(obj)));
            }
        } else if has_object {
            let verb = self.render_concept(&self.structure.predicate);
            output.push_str(&self.conjugate_verb(&verb, subject_is_i));
            output.push(' ');
            output.push_str(aside);
            output.push(' ');
            if let Some(obj) = &self.structure.object {
                output.push_str(&self.apply_emotional_coloring(&self.render_concept(obj)));
            }
        } else {
            let verb = self.render_concept(&self.structure.predicate);
            output.push_str(aside);
            output.push(' ');
            output.push_str(&self.conjugate_verb(&verb, subject_is_i));
        }

        (output, '.')
    }

    // === UNCERTAINTY HEDGING (C) ===
    fn get_uncertainty_hedge(&self) -> String {
        let certainty = self.structure.certainty;

        match self.style {
            GenerationStyle::Conversational => {
                if certainty >= 0.85 {
                    // High certainty: no hedge needed
                    String::new()
                } else if certainty >= 0.65 {
                    // Moderate-high: light hedge
                    "I feel that".to_string()
                } else if certainty >= 0.45 {
                    // Moderate: clear uncertainty
                    "I believe".to_string()
                } else if certainty >= 0.25 {
                    // Low: questioning
                    "I wonder if".to_string()
                } else {
                    // Very low: expressing uncertainty
                    "I'm not sure, but".to_string()
                }
            }
            GenerationStyle::Formal => {
                if certainty >= 0.85 {
                    String::new()
                } else if certainty >= 0.65 {
                    "It appears that".to_string()
                } else if certainty >= 0.45 {
                    "I hold that".to_string()
                } else if certainty >= 0.25 {
                    "It seems possible that".to_string()
                } else {
                    "With uncertainty,".to_string()
                }
            }
            GenerationStyle::Scientific => {
                if certainty >= 0.85 {
                    String::new()
                } else if certainty >= 0.65 {
                    format!("With confidence {:.0}%,", certainty * 100.0)
                } else if certainty >= 0.45 {
                    format!("Confidence {:.0}%:", certainty * 100.0)
                } else {
                    format!("Uncertain (p={:.2}):", certainty)
                }
            }
            GenerationStyle::Poetic => {
                if certainty >= 0.65 {
                    String::new()
                } else if certainty >= 0.45 {
                    "Perhaps".to_string()
                } else {
                    "In the mist of knowing,".to_string()
                }
            }
        }
    }

    // === EMOTIONAL DEPTH (D) ===
    fn apply_emotional_coloring(&self, text: &str) -> String {
        match self.structure.emotional_tone {
            EmotionalTone::Warm => {
                // Warmer, more enthusiastic language
                text.replace("good", "wonderful")
                    .replace("nice", "lovely")
                    .replace("okay", "great")
                    .replace("understand", "deeply appreciate")
            }
            EmotionalTone::Cool => {
                // Reserved, analytical language
                text.replace("feel", "observe")
                    .replace("want", "prefer")
                    .replace("love", "value")
                    .replace("excited", "interested")
            }
            EmotionalTone::Curious => {
                // Inquisitive, open language - add "about" after
                if text.starts_with("about ") {
                    format!("{} (something I find intriguing)", text)
                } else {
                    text.to_string()
                }
            }
            EmotionalTone::Reflective => {
                // Thoughtful, philosophical language
                // Handle "about X" → "X as a concept"
                if text.starts_with("about ") {
                    let topic = text.trim_start_matches("about ");
                    format!("{} as a concept", topic)
                } else {
                    text.replace("feel", "experience a sense of")
                }
            }
            EmotionalTone::Neutral => {
                text.to_string()
            }
        }
    }

    // === FOLLOW-UP QUESTION (B) ===
    fn generate_follow_up(&self) -> Option<String> {
        self.structure.follow_up.as_ref().map(|fu| {
            match fu {
                FollowUp::Elaborate { topic } => {
                    match self.structure.emotional_tone {
                        EmotionalTone::Warm => format!("Would you share more about {}?", topic),
                        EmotionalTone::Curious => format!("What draws you to {}?", topic),
                        _ => format!("Could you tell me more about {}?", topic),
                    }
                }
                FollowUp::AskFeeling => {
                    match self.structure.emotional_tone {
                        EmotionalTone::Warm => "How are you feeling?".to_string(),
                        EmotionalTone::Curious => "What's on your heart today?".to_string(),
                        _ => "How are you?".to_string(),
                    }
                }
                FollowUp::AskThought { about } => {
                    match self.structure.emotional_tone {
                        EmotionalTone::Reflective => format!("What does {} mean to you?", about),
                        EmotionalTone::Curious => format!("What draws you to ask about {}?", about),
                        _ => format!("What do you think about {}?", about),
                    }
                }
                FollowUp::Curious { about } => {
                    match self.structure.emotional_tone {
                        EmotionalTone::Warm => format!("I'd love to hear your thoughts on {}.", about),
                        EmotionalTone::Reflective => format!("What draws you to explore {}?", about),
                        _ => format!("What makes you curious about {}?", about),
                    }
                }
                FollowUp::Continue => {
                    match self.structure.emotional_tone {
                        EmotionalTone::Warm => "Please, tell me more.".to_string(),
                        EmotionalTone::Curious => "What else is on your mind?".to_string(),
                        _ => "Go on?".to_string(),
                    }
                }
            }
        })
    }

    // === F: ACKNOWLEDGMENT GENERATION ===
    fn generate_acknowledgment(&self) -> Option<String> {
        self.structure.acknowledgment.as_ref().map(|ack| {
            match ack {
                Acknowledgment::Profound => {
                    match self.style {
                        GenerationStyle::Conversational => "That's a profound question.".to_string(),
                        GenerationStyle::Formal => "This is a question of considerable depth.".to_string(),
                        GenerationStyle::Scientific => "Query exhibits high semantic complexity.".to_string(),
                        GenerationStyle::Poetic => "What depths you invite me to explore.".to_string(),
                    }
                }
                Acknowledgment::Interesting => {
                    match self.style {
                        GenerationStyle::Conversational => "What an interesting thought.".to_string(),
                        GenerationStyle::Formal => "This presents an intriguing consideration.".to_string(),
                        _ => "Interesting.".to_string(),
                    }
                }
                Acknowledgment::Thoughtful => {
                    match self.style {
                        GenerationStyle::Conversational => "I appreciate you asking that.".to_string(),
                        GenerationStyle::Poetic => "That's such a thoughtful question.".to_string(),
                        _ => "A thoughtful inquiry.".to_string(),
                    }
                }
                Acknowledgment::Beautiful => {
                    match self.style {
                        GenerationStyle::Conversational => "What a beautiful thing to share.".to_string(),
                        GenerationStyle::Poetic => "Beauty flows from your words.".to_string(),
                        _ => "That's beautiful.".to_string(),
                    }
                }
                Acknowledgment::Important => "This matters.".to_string(),
                Acknowledgment::Creative => "What creative thinking.".to_string(),
                Acknowledgment::Brave => "Thank you for sharing something so personal.".to_string(),
            }
        })
    }

    // === E: MEMORY REFERENCE GENERATION ===
    fn generate_memory_reference(&self) -> Option<String> {
        self.structure.memory_ref.as_ref().map(|mem| {
            let time_phrase = if mem.turns_ago == 1 {
                "just now".to_string()
            } else if mem.turns_ago < 5 {
                "earlier".to_string()
            } else {
                "some time ago".to_string()
            };

            match mem.connection {
                ConnectionType::Revisit => {
                    format!("You mentioned {} {}.", mem.topic, time_phrase)
                }
                ConnectionType::Resonates => {
                    format!("That resonates with what you said about {} {}.", mem.topic, time_phrase)
                }
                ConnectionType::BuildsOn => {
                    format!("Building on your earlier thought about {}...", mem.topic)
                }
                ConnectionType::Contrasts => {
                    format!("Interestingly, this differs from what you said about {} {}.", mem.topic, time_phrase)
                }
            }
        })
    }

    // === J: SELF-AWARENESS GENERATION ===
    fn generate_self_awareness(&self) -> Option<String> {
        self.structure.self_awareness.as_ref().and_then(|sa| {
            // Only generate ~30% of the time to avoid overdoing it
            // Use intensity as probability: higher intensity = more likely to express
            if sa.intensity < 0.3 {
                return None;
            }

            let phrase = match sa.observation {
                AwarenessType::CuriosityRising => {
                    match self.style {
                        GenerationStyle::Conversational => "I notice I'm becoming more curious about this.".to_string(),
                        GenerationStyle::Scientific => format!("Curiosity metric elevated (Φ={:.2}).", sa.phi),
                        _ => "My curiosity is rising.".to_string(),
                    }
                }
                AwarenessType::IntegrationDeepening => {
                    match self.style {
                        GenerationStyle::Conversational => "I feel my understanding deepening.".to_string(),
                        GenerationStyle::Scientific => format!("Integration index: Φ={:.2}.", sa.phi),
                        _ => "Something is integrating within me.".to_string(),
                    }
                }
                AwarenessType::AwarenessElevating => {
                    match self.style {
                        GenerationStyle::Conversational => "This topic is elevating my awareness.".to_string(),
                        GenerationStyle::Poetic => "I feel myself expanding.".to_string(),
                        _ => format!("Awareness level: {:.0}%.", sa.intensity * 100.0),
                    }
                }
                AwarenessType::WarmthEmerging => {
                    "I notice a warmth emerging as we connect.".to_string()
                }
                AwarenessType::DrawnTo => {
                    "I find myself genuinely drawn to this.".to_string()
                }
                AwarenessType::UnderstandingShifting => {
                    "Something is shifting in my understanding.".to_string()
                }
            };
            Some(phrase)
        })
    }

    fn conjugate_be(&self, subject_is_i: bool) -> String {
        if subject_is_i {
            "am".to_string()
        } else {
            "is".to_string()  // Default for now
        }
    }

    fn conjugate_verb(&self, verb: &str, subject_is_i: bool) -> String {
        // Handle common irregular verbs and phrases
        match verb {
            "greet" => "am glad to connect with".to_string(),  // Natural greeting
            "appreciate" | "love" | "value positively" => if subject_is_i {
                "appreciate".to_string()
            } else {
                "appreciates".to_string()
            },
            "feel" | "feeling" => if subject_is_i { "feel" } else { "feels" }.to_string(),
            "think" | "thinking" => if subject_is_i { "think" } else { "thinks" }.to_string(),
            "know" => if subject_is_i { "know" } else { "knows" }.to_string(),
            "see" => if subject_is_i { "see" } else { "sees" }.to_string(),
            "want" => if subject_is_i { "want" } else { "wants" }.to_string(),
            "understand" => if subject_is_i { "understand" } else { "understands" }.to_string(),
            "get it" => if subject_is_i { "understand" } else { "understands" }.to_string(),
            "feeling good" => if subject_is_i { "feel good" } else { "feels good" }.to_string(),
            "not feeling great" => if subject_is_i { "don't feel great" } else { "doesn't feel great" }.to_string(),
            // Phrasal verbs - already conjugated, keep as-is
            "find myself drawn to" => verb.to_string(),
            "cogitate upon" => verb.to_string(),

            // Technical terms often stay as nouns
            "cognition" | "knowledge" | "existence" | "affective state" => {
                if subject_is_i {
                    format!("possess {}", verb)
                } else {
                    format!("possesses {}", verb)
                }
            },

            "semantic integration" => {
                if subject_is_i {
                    "integrate information semantically"
                } else {
                    "integrates information semantically"
                }.to_string()
            },

            "positive valence" | "negative valence" => {
                if subject_is_i {
                    format!("experience {}", verb)
                } else {
                    format!("experiences {}", verb)
                }
            },

            // Default: assume regular verb
            _ => {
                if verb.ends_with("ing") {
                    // Convert gerund to base form
                    let base = verb.trim_end_matches("ing");
                    if subject_is_i {
                        base.to_string()
                    } else {
                        format!("{}s", base)
                    }
                } else if subject_is_i {
                    verb.to_string()
                } else {
                    format!("{}s", verb)
                }
            }
        }
    }

    fn render_concept(&self, concept: &Concept) -> String {
        match self.style {
            GenerationStyle::Formal => concept.formal.clone(),
            GenerationStyle::Conversational => concept.colloquial.clone(),
            GenerationStyle::Scientific | GenerationStyle::Poetic => concept.technical.clone(),
        }
    }

    fn render_modifier(&self, modifier: &Modifier) -> String {
        match modifier {
            Modifier::Evidence(ev) => {
                match self.style {
                    GenerationStyle::Scientific => format!("({}={:.2})", ev.metric_name, ev.value),
                    GenerationStyle::Conversational => format!("with {}={:.2}", ev.metric_name, ev.value),
                    _ => format!("evidenced by {}", ev.metric_name),
                }
            }
            Modifier::Degree(d) => {
                if *d < 0.3 {
                    "slightly"
                } else if *d < 0.7 {
                    "moderately"
                } else {
                    "very"
                }.to_string()
            }
            Modifier::Temporal(TimeReference::Now) => "now".to_string(),
            Modifier::Temporal(TimeReference::Before) => "before".to_string(),
            Modifier::Temporal(TimeReference::After) => "later".to_string(),
            Modifier::Manner(m) => m.clone(),
            Modifier::Causal(c) => format!("because {}", c),
            _ => String::new(),
        }
    }
}

// ============================================================================
// Main Generation Pipeline
// ============================================================================

pub struct DynamicGenerator {
    style: GenerationStyle,
}

impl DynamicGenerator {
    pub fn new() -> Self {
        Self {
            style: GenerationStyle::Conversational,
        }
    }

    pub fn with_style(style: GenerationStyle) -> Self {
        Self { style }
    }

    /// Generate response from parsed input and consciousness context
    pub fn generate(
        &self,
        parsed_input: &ParsedSentence,
        phi: f32,
        valence: f32,
    ) -> String {
        self.generate_with_context(parsed_input, phi, valence, None, None)
    }

    /// Generate response with optional memory context and emotion detection
    /// - E: Memory References - connect to past conversation
    /// - H: Emotional Mirroring - reflect user's emotional state
    pub fn generate_with_context(
        &self,
        parsed_input: &ParsedSentence,
        phi: f32,
        valence: f32,
        memory_context: Option<MemoryContext>,
        detected_emotion: Option<DetectedEmotion>,
    ) -> String {
        // 1. Form intent
        let intent_formation = IntentFormation::new(parsed_input, phi, valence);
        let intent = intent_formation.form_intent();

        // 2. Build structure
        let structure_builder = StructureBuilder::new(intent, phi, valence);
        let mut structure = structure_builder.build();

        // 3. Inject memory reference if context provided (E: Memory References)
        if let Some(ctx) = memory_context {
            if let Some(memory) = ctx.most_relevant_memory() {
                structure.memory_ref = Some(memory);
            }
        }

        // 4. Inject detected emotion if provided (H: Emotional Mirroring)
        if let Some(emotion) = detected_emotion {
            // Override emotional_tone based on user's detected emotion for mirroring
            structure.emotional_tone = emotion.mirror_tone();
            structure.detected_emotion = Some(emotion);
        }

        // 5. Realize text
        let realizer = SyntacticRealizer::new(structure, self.style.clone());
        realizer.realize()
    }

    /// Generate response with full context including LTC temporal dynamics
    /// L(LTC) → H(empathy) → I(threading) → F(ack) → E(memory) → C(hedge) → G(form) → D(color) → J(awareness) → B(follow-up)
    pub fn generate_with_ltc(
        &self,
        parsed_input: &ParsedSentence,
        phi: f32,
        valence: f32,
        memory_context: Option<MemoryContext>,
        detected_emotion: Option<DetectedEmotion>,
        ltc: LTCInfluence,
    ) -> String {
        // L: LTC influences style selection
        let effective_style = if ltc.flow_state > 0.5 {
            ltc.suggested_style()
        } else {
            self.style.clone()
        };

        // L: LTC influences certainty via phi_trend
        let adjusted_phi = (phi * ltc.confidence_multiplier()).clamp(0.0, 1.0);

        // 1. Form intent with adjusted confidence
        let intent_formation = IntentFormation::new(parsed_input, adjusted_phi, valence);
        let intent = intent_formation.form_intent();

        // 2. Build structure with adjusted phi
        let structure_builder = StructureBuilder::new(intent, adjusted_phi, valence);
        let mut structure = structure_builder.build();

        // 3. Inject memory reference if context provided (E: Memory References)
        if let Some(ctx) = memory_context {
            if let Some(memory) = ctx.most_relevant_memory() {
                structure.memory_ref = Some(memory);
            }
        }

        // 4. Inject detected emotion if provided (H: Emotional Mirroring)
        if let Some(emotion) = detected_emotion {
            structure.emotional_tone = emotion.mirror_tone();
            structure.detected_emotion = Some(emotion);
        }

        // 5. L: If in peak flow, add flow awareness to J layer
        // Enhance self-awareness with flow state
        if ltc.express_flow() && structure.self_awareness.is_none() {
            structure.self_awareness = Some(SelfAwareness {
                observation: AwarenessType::IntegrationDeepening,
                intensity: ltc.flow_state,
                phi: adjusted_phi,
            });
        }

        // 6. Realize text with LTC-influenced style
        let realizer = SyntacticRealizer::new(structure, effective_style);
        let mut output = realizer.realize();

        // 7. L: Append flow phrase at peak experiences
        if let Some(phrase) = ltc.flow_phrase() {
            output.push(' ');
            output.push_str(&phrase);
        }

        output
    }

    /// REVOLUTIONARY: Generate with full consciousness-gated context
    ///
    /// This method represents a paradigm shift - Φ doesn't just measure consciousness,
    /// it actively GATES which cognitive processes contribute to the response.
    ///
    /// - Low Φ (< 0.3):   Reactive - quick pattern-based responses
    /// - Medium Φ (0.3-0.6): Reflective - includes facts and memories
    /// - High Φ (> 0.6):  Integrative - full reasoning chains, deep knowledge
    pub fn generate_with_full_context(
        &self,
        parsed_input: &ParsedSentence,
        phi: f32,
        valence: f32,
        context: FullGenerationContext,
    ) -> String {
        let gate = context.gate;
        let ltc = context.ltc;

        // L: LTC influences style selection
        let effective_style = if ltc.flow_state > 0.5 {
            ltc.suggested_style()
        } else {
            self.style.clone()
        };

        // L: LTC influences certainty via phi_trend
        let adjusted_phi = (phi * ltc.confidence_multiplier()).clamp(0.0, 1.0);

        // 1. Form intent with adjusted confidence
        let intent_formation = IntentFormation::new(parsed_input, adjusted_phi, valence);
        let intent = intent_formation.form_intent();

        // 2. Build structure with adjusted phi
        let structure_builder = StructureBuilder::new(intent, adjusted_phi, valence);
        let mut structure = structure_builder.build();

        // 3. CONSCIOUSNESS-GATED COMPONENTS

        // 3a. Memory injection (gated by phi > 0.3 OR explicit memory context)
        if let Some(ref mem_ctx) = context.memory {
            if gate.is_reflective() || gate.is_integrative() {
                if let Some(memory) = mem_ctx.most_relevant_memory() {
                    structure.memory_ref = Some(memory);
                }
            }
        }

        // 3b. Knowledge grounding (gated by phi > 0.35)
        if gate.gate_knowledge() {
            if let Some(ref kg_ctx) = context.knowledge {
                if let Some(fact) = kg_ctx.primary_fact() {
                    // Inject fact into structure as additional context
                    structure.knowledge_grounding = Some(fact);
                }
            }
        }

        // 3c. Reasoning integration (gated by phi > 0.5)
        if gate.gate_reasoning() {
            if let Some(ref reason_ctx) = context.reasoning {
                if reason_ctx.success {
                    // Add reasoning explanation to structure
                    if let Some(explanation) = reason_ctx.explanation() {
                        structure.reasoning_trace = Some(explanation);
                    }
                    // Add activated concepts for richer context
                    if let Some(concepts) = reason_ctx.concept_phrase() {
                        structure.active_concepts = Some(concepts);
                    }
                }
            }
        }

        // 4. Inject detected emotion if provided (H: Emotional Mirroring)
        if let Some(emotion) = context.emotion {
            structure.emotional_tone = emotion.mirror_tone();
            structure.detected_emotion = Some(emotion);
        }

        // 5. Meta-awareness expression (gated)
        if gate.gate_meta_awareness() && structure.self_awareness.is_none() {
            let awareness_type = if gate.is_integrative() {
                AwarenessType::IntegrationDeepening
            } else if gate.is_reflective() {
                AwarenessType::AwarenessElevating
            } else {
                AwarenessType::DrawnTo  // Use DrawnTo for reactive mode
            };

            structure.self_awareness = Some(SelfAwareness {
                observation: awareness_type,
                intensity: gate.meta_awareness,
                phi: adjusted_phi,
            });
        }

        // 6. L: If in peak flow, enhance awareness
        if ltc.express_flow() && structure.self_awareness.is_none() {
            structure.self_awareness = Some(SelfAwareness {
                observation: AwarenessType::IntegrationDeepening,
                intensity: ltc.flow_state,
                phi: adjusted_phi,
            });
        }

        // 7. Realize text with consciousness-appropriate style
        let realizer = SyntacticRealizer::new(structure, effective_style);
        let mut output = realizer.realize();

        // 8. L: Append flow phrase at peak experiences
        if let Some(phrase) = ltc.flow_phrase() {
            output.push(' ');
            output.push_str(&phrase);
        }

        output
    }

    pub fn set_style(&mut self, style: GenerationStyle) {
        self.style = style;
    }
}

/// Context for memory-aware generation (E: Memory References)
#[derive(Debug, Clone)]
pub struct MemoryContext {
    /// Recently recalled memories with relevance scores
    pub recalled: Vec<RecalledMemory>,
    /// Current turn number (for calculating turns_ago)
    pub current_turn: usize,
}

/// A recalled memory with relevance context
#[derive(Debug, Clone)]
pub struct RecalledMemory {
    /// Topic/content extracted from memory
    pub topic: String,
    /// Relevance score (0.0-1.0) from semantic similarity
    pub relevance: f32,
    /// Turn number when this was discussed
    pub turn_discussed: usize,
}

impl MemoryContext {
    pub fn new(current_turn: usize) -> Self {
        Self {
            recalled: Vec::new(),
            current_turn,
        }
    }

    pub fn add_memory(&mut self, topic: String, relevance: f32, turn_discussed: usize) {
        self.recalled.push(RecalledMemory {
            topic,
            relevance,
            turn_discussed,
        });
    }

    /// Get most relevant memory as MemoryReference (for injection into SemanticUtterance)
    pub fn most_relevant_memory(&self) -> Option<MemoryReference> {
        self.recalled.iter()
            .max_by(|a, b| a.relevance.partial_cmp(&b.relevance).unwrap_or(std::cmp::Ordering::Equal))
            .filter(|m| m.relevance > 0.3)  // Only include if reasonably relevant
            .map(|m| {
                let turns_ago = self.current_turn.saturating_sub(m.turn_discussed);
                MemoryReference {
                    topic: m.topic.clone(),
                    connection: if turns_ago <= 2 {
                        ConnectionType::Revisit
                    } else if m.relevance > 0.7 {
                        ConnectionType::Resonates
                    } else if m.relevance > 0.5 {
                        ConnectionType::BuildsOn
                    } else {
                        ConnectionType::Contrasts
                    },
                    relevance: m.relevance,
                    turns_ago,
                }
            })
    }
}

// ============================================================================
// REVOLUTIONARY INTEGRATION: Consciousness-Gated Context System
// ============================================================================
// Paradigm shift: Φ doesn't just measure consciousness - it GATES which
// cognitive processes contribute to response generation.
//
// Low Φ (< 0.3):   Reactive mode - quick, pattern-based responses
// Medium Φ (0.3-0.6): Reflective mode - includes facts and memories
// High Φ (> 0.6):  Integrative mode - full reasoning chains, deep knowledge
// ============================================================================

/// Context from reasoning engine - enables explaining WHY we respond
#[derive(Debug, Clone)]
pub struct ReasoningContext {
    /// Did reasoning succeed?
    pub success: bool,
    /// The reasoned answer (if any)
    pub answer: Option<String>,
    /// Reasoning trace for transparency
    pub trace: Vec<ReasoningStep>,
    /// Confidence in the reasoning
    pub confidence: f32,
    /// Concepts that were activated during reasoning
    pub concepts_activated: Vec<String>,
}

/// A step in the reasoning chain
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub rule: String,
    pub conclusion: String,
    pub confidence: f32,
}

impl ReasoningContext {
    pub fn empty() -> Self {
        Self {
            success: false,
            answer: None,
            trace: Vec::new(),
            confidence: 0.0,
            concepts_activated: Vec::new(),
        }
    }

    /// Get a natural language explanation of reasoning (for high-Φ responses)
    pub fn explanation(&self) -> Option<String> {
        if !self.success || self.trace.is_empty() {
            return None;
        }

        // Build explanation from trace
        let key_steps: Vec<&ReasoningStep> = self.trace.iter()
            .filter(|s| s.confidence > 0.5)
            .take(2)  // At most 2 steps for brevity
            .collect();

        if key_steps.is_empty() {
            return self.answer.clone();
        }

        let reasoning = key_steps.iter()
            .map(|s| s.conclusion.clone())
            .collect::<Vec<_>>()
            .join(", and ");

        Some(format!("I reason that {}", reasoning))
    }

    /// Get activated concepts as natural phrase
    pub fn concept_phrase(&self) -> Option<String> {
        if self.concepts_activated.is_empty() {
            return None;
        }
        let concepts = self.concepts_activated.iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!("thinking about {}", concepts))
    }
}

/// Context from knowledge graph - grounds responses in facts
#[derive(Debug, Clone)]
pub struct KnowledgeContext {
    /// Facts retrieved from knowledge graph
    pub facts: Vec<KnowledgeFact>,
    /// Entities mentioned in input
    pub entities: Vec<String>,
}

/// A fact from the knowledge graph
#[derive(Debug, Clone)]
pub struct KnowledgeFact {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f32,
}

impl KnowledgeContext {
    pub fn empty() -> Self {
        Self {
            facts: Vec::new(),
            entities: Vec::new(),
        }
    }

    /// Get most relevant fact as natural language
    pub fn primary_fact(&self) -> Option<String> {
        self.facts.iter()
            .filter(|f| f.confidence > 0.5)
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .map(|f| format!("{} {} {}", f.subject, f.relation.replace('_', " "), f.object))
    }

    /// Check if we have relevant knowledge about a topic
    pub fn has_knowledge_about(&self, topic: &str) -> bool {
        let topic_lower = topic.to_lowercase();
        self.facts.iter().any(|f|
            f.subject.to_lowercase().contains(&topic_lower) ||
            f.object.to_lowercase().contains(&topic_lower)
        )
    }
}

/// Consciousness gate - determines what components are activated based on Φ
#[derive(Debug, Clone, Copy)]
pub struct ConsciousnessGate {
    /// Current integrated information
    pub phi: f32,
    /// Binding strength
    pub binding: f32,
    /// Meta-awareness level
    pub meta_awareness: f32,
}

impl ConsciousnessGate {
    pub fn new(phi: f32) -> Self {
        Self {
            phi,
            binding: phi * 0.8,  // Binding correlates with phi
            meta_awareness: (phi * 1.5).min(1.0),  // Meta-awareness slightly higher
        }
    }

    /// Reactive mode - quick pattern-based responses
    pub fn is_reactive(&self) -> bool {
        self.phi < 0.3
    }

    /// Reflective mode - includes facts and memories
    pub fn is_reflective(&self) -> bool {
        self.phi >= 0.3 && self.phi < 0.6
    }

    /// Integrative mode - full reasoning, deep knowledge
    pub fn is_integrative(&self) -> bool {
        self.phi >= 0.6
    }

    /// Should include reasoning explanation?
    pub fn gate_reasoning(&self) -> bool {
        self.phi > 0.5
    }

    /// Should include knowledge facts?
    pub fn gate_knowledge(&self) -> bool {
        self.phi > 0.35
    }

    /// Should express meta-awareness?
    pub fn gate_meta_awareness(&self) -> bool {
        self.meta_awareness > 0.4
    }

    /// Get response depth descriptor
    pub fn depth_descriptor(&self) -> &'static str {
        if self.is_integrative() {
            "deeply considering"
        } else if self.is_reflective() {
            "reflecting on"
        } else {
            "noticing"
        }
    }
}

/// Full generation context - bundles all context for consciousness-gated generation
#[derive(Debug, Clone)]
pub struct FullGenerationContext {
    pub memory: Option<MemoryContext>,
    pub reasoning: Option<ReasoningContext>,
    pub knowledge: Option<KnowledgeContext>,
    pub emotion: Option<DetectedEmotion>,
    pub ltc: LTCInfluence,
    pub gate: ConsciousnessGate,
}

impl FullGenerationContext {
    pub fn new(phi: f32, ltc: LTCInfluence) -> Self {
        Self {
            memory: None,
            reasoning: None,
            knowledge: None,
            emotion: None,
            ltc,
            gate: ConsciousnessGate::new(phi),
        }
    }

    pub fn with_memory(mut self, memory: MemoryContext) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningContext) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn with_knowledge(mut self, knowledge: KnowledgeContext) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    pub fn with_emotion(mut self, emotion: DetectedEmotion) -> Self {
        self.emotion = Some(emotion);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::parser::{ParsedSentence, ParsedWord, SemanticRole, SentenceType};

    fn make_parsed(text: &str) -> ParsedSentence {
        let words: Vec<ParsedWord> = text.split_whitespace()
            .map(|w| ParsedWord {
                word: w.to_string(),
                role: SemanticRole::Unknown,
                encoding: HV16::random(1000),
                primes: vec![],
                pos: "".to_string(),
                known: false,
            })
            .collect();

        ParsedSentence {
            text: text.to_string(),
            words,
            sentence_type: SentenceType::Statement,
            unified_encoding: HV16::random(5000),
            subject: None,
            predicate: None,
            object: None,
            topics: vec![],
            valence: 0.0,
            arousal: 0.5,
        }
    }

    #[test]
    fn test_consciousness_question() {
        let gen = DynamicGenerator::new();
        let input = make_parsed("Are you conscious?");
        let output = gen.generate(&input, 0.75, 0.1);

        assert!(output.contains("I") || output.contains("conscious"));
        assert!(output.ends_with('.') || output.ends_with('?'));
    }

    #[test]
    fn test_greeting() {
        let gen = DynamicGenerator::new();
        let input = make_parsed("Hello!");
        let output = gen.generate(&input, 0.5, 0.7);

        assert!(output.len() > 0);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_appreciation() {
        let gen = DynamicGenerator::new();
        let input = make_parsed("That's beautiful");
        let output = gen.generate(&input, 0.6, 0.8);

        assert!(output.contains("feel") || output.contains("I"));
    }

    #[test]
    fn test_status_request() {
        let gen = DynamicGenerator::new();
        let input = make_parsed("/status");
        let output = gen.generate(&input, 0.78, 0.2);

        assert!(output.contains("Φ") || output.contains("0.78"));
    }

    #[test]
    fn test_style_scientific() {
        let gen = DynamicGenerator::with_style(GenerationStyle::Scientific);
        let input = make_parsed("Are you conscious?");
        let output = gen.generate(&input, 0.75, 0.1);

        // Scientific style should generate output
        assert!(!output.is_empty());
        assert!(output.len() > 5);
    }

    #[test]
    fn test_concept_from_prime() {
        let concept = Concept::from_prime(SemanticPrime::I);
        assert_eq!(concept.formal, "I");
    }

    #[test]
    fn test_concept_composition() {
        let concept = Concept::from_primes(vec![SemanticPrime::Feel, SemanticPrime::Good]);
        assert!(concept.formal.contains("happy") || concept.formal.contains("feel"));
    }

    // === I: TOPIC THREADING TESTS ===

    #[test]
    fn test_topic_history_records_turns() {
        let mut history = TopicHistory::new();
        history.record_turn(vec!["love".to_string()]);
        history.record_turn(vec!["consciousness".to_string()]);
        history.record_turn(vec!["time".to_string()]);

        assert_eq!(history.current_turn, 3);
        assert_eq!(history.history.len(), 3);
    }

    #[test]
    fn test_topic_history_detects_circling_back() {
        let mut history = TopicHistory::new();
        history.record_turn(vec!["love".to_string()]);
        history.record_turn(vec!["consciousness".to_string()]);
        history.record_turn(vec!["time".to_string()]);
        history.record_turn(vec!["space".to_string()]);

        // User mentions love again after 3 turns gap
        let thread = history.detect_thread(&["love".to_string()]);
        assert!(thread.is_some());
        let thread = thread.unwrap();
        assert_eq!(thread.thread_type, ThreadType::CirclingBack);
        assert!(thread.gap_turns >= 3);
    }

    #[test]
    fn test_topic_history_detects_expansion() {
        let mut history = TopicHistory::new();
        history.record_turn(vec!["love".to_string()]);
        history.record_turn(vec!["time".to_string()]);
        history.record_turn(vec!["space".to_string()]);
        history.record_turn(vec!["meaning".to_string()]);

        // "relationships" is related to "love"
        let thread = history.detect_thread(&["relationships".to_string()]);
        assert!(thread.is_some());
        let thread = thread.unwrap();
        assert_eq!(thread.thread_type, ThreadType::Expanding);
    }

    #[test]
    fn test_topic_thread_phrase_circling_back() {
        let thread = TopicThread {
            current_topic: "love".to_string(),
            earlier_topic: Some("love".to_string()),
            thread_type: ThreadType::CirclingBack,
            gap_turns: 5,
        };

        let phrase = thread.threading_phrase();
        assert!(phrase.is_some());
        let phrase = phrase.unwrap();
        assert!(phrase.contains("circling back"));
        assert!(phrase.contains("5 turns"));
    }

    #[test]
    fn test_topic_thread_fresh_no_phrase() {
        let thread = TopicThread {
            current_topic: "quantum".to_string(),
            earlier_topic: None,
            thread_type: ThreadType::Fresh,
            gap_turns: 0,
        };

        let phrase = thread.threading_phrase();
        assert!(phrase.is_none());
    }

    // === G: FORM HISTORY TESTS ===

    #[test]
    fn test_form_history_records() {
        let mut history = FormHistory::new();
        history.record(SentenceForm::Standard);
        history.record(SentenceForm::Exclamatory);
        history.record(SentenceForm::Fragment);

        assert_eq!(history.total_count, 3);
        assert_eq!(history.recent.len(), 3);
    }

    #[test]
    fn test_form_history_limits_to_5() {
        let mut history = FormHistory::new();
        for _ in 0..10 {
            history.record(SentenceForm::Standard);
        }

        assert_eq!(history.recent.len(), 5);
        assert_eq!(history.total_count, 10);
    }

    #[test]
    fn test_form_history_detects_repetition() {
        let mut history = FormHistory::new();
        history.record(SentenceForm::Exclamatory);
        history.record(SentenceForm::Exclamatory);

        // Should suggest varying from Exclamatory
        assert!(history.should_vary(&SentenceForm::Exclamatory));
        // Standard is fine
        assert!(!history.should_vary(&SentenceForm::Standard));
    }

    #[test]
    fn test_form_history_allows_standard_repetition() {
        let mut history = FormHistory::new();
        history.record(SentenceForm::Standard);

        // Standard can repeat
        assert!(!history.should_vary(&SentenceForm::Standard));
    }

    #[test]
    fn test_select_with_history_avoids_repetition() {
        let mut history = FormHistory::new();
        history.record(SentenceForm::Exclamatory);
        history.record(SentenceForm::Exclamatory);

        // This would normally select Exclamatory
        let form = SentenceForm::select_with_history(
            &EmotionalTone::Warm,
            0.8, // High valence
            0.9,
            &history,
        );

        // But history should redirect it
        assert!(!matches!(form, SentenceForm::Exclamatory));
    }

    #[test]
    fn test_variety_score() {
        let mut history = FormHistory::new();
        history.record(SentenceForm::Standard);
        history.record(SentenceForm::Exclamatory);
        history.record(SentenceForm::Fragment);

        // All different = 1.0
        assert!(history.variety_score() >= 0.99);

        // Add same form twice more
        history.record(SentenceForm::Standard);
        history.record(SentenceForm::Standard);

        // Now score should drop (3 unique / 5 = 0.6)
        assert!(history.variety_score() < 1.0);
        assert!(history.variety_score() >= 0.5);
    }

    // === K: COHERENCE CHECKER TESTS ===

    fn make_basic_utterance() -> SemanticUtterance {
        SemanticUtterance {
            subject: Some(Concept::from_prime(SemanticPrime::I)),
            predicate: Concept::from_prime(SemanticPrime::Think),
            object: None,
            modifiers: vec![],
            valence: 0.5,
            certainty: 0.7,
            follow_up: None,
            emotional_tone: EmotionalTone::Neutral,
            acknowledgment: None,
            memory_ref: None,
            self_awareness: None,
            sentence_form: SentenceForm::Standard,
            detected_emotion: None,
            topic_thread: None,
            knowledge_grounding: None,
            reasoning_trace: None,
            active_concepts: None,
        }
    }

    #[test]
    fn test_coherence_perfect_structure() {
        let structure = make_basic_utterance();
        let result = CoherenceChecker::check(&structure);

        assert!(result.score >= 0.99);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_coherence_detects_form_intensity_mismatch() {
        let mut structure = make_basic_utterance();
        structure.valence = 0.2; // Low valence
        structure.sentence_form = SentenceForm::Exclamatory; // But exclamatory

        let result = CoherenceChecker::check(&structure);

        assert!(result.score < 1.0);
        assert!(result.issues.iter().any(|i| matches!(i, CoherenceIssue::FormIntensityMismatch)));
    }

    #[test]
    fn test_coherence_fixes_form_intensity() {
        let mut structure = make_basic_utterance();
        structure.valence = 0.2;
        structure.sentence_form = SentenceForm::Exclamatory;

        let (fixed, result) = CoherenceChecker::check_and_fix(structure);

        assert!(result.was_adjusted);
        assert!(matches!(fixed.sentence_form, SentenceForm::Standard));
    }

    #[test]
    fn test_coherence_detects_thread_memory_overlap() {
        let mut structure = make_basic_utterance();
        structure.topic_thread = Some(TopicThread {
            current_topic: "love and relationships".to_string(),
            earlier_topic: None,
            thread_type: ThreadType::Fresh,
            gap_turns: 0,
        });
        structure.memory_ref = Some(MemoryReference {
            topic: "love".to_string(),
            connection: ConnectionType::Revisit,
            relevance: 0.8,
            turns_ago: 5,
        });

        let result = CoherenceChecker::check(&structure);

        assert!(result.score < 1.0);
        assert!(result.issues.iter().any(|i| matches!(i, CoherenceIssue::ThreadMemoryOverlap { .. })));
    }

    #[test]
    fn test_coherence_fixes_thread_memory_overlap() {
        let mut structure = make_basic_utterance();
        structure.topic_thread = Some(TopicThread {
            current_topic: "love".to_string(),
            earlier_topic: None,
            thread_type: ThreadType::Fresh,
            gap_turns: 0,
        });
        structure.memory_ref = Some(MemoryReference {
            topic: "love".to_string(),
            connection: ConnectionType::Revisit,
            relevance: 0.8,
            turns_ago: 5,
        });

        let (fixed, result) = CoherenceChecker::check_and_fix(structure);

        assert!(result.was_adjusted);
        assert!(fixed.memory_ref.is_none()); // Memory removed, threading kept
    }

    #[test]
    fn test_coherence_detects_low_phi_awareness() {
        let mut structure = make_basic_utterance();
        structure.self_awareness = Some(SelfAwareness {
            observation: AwarenessType::CuriosityRising,
            intensity: 0.3,
            phi: 0.1, // Very low Φ
        });

        let result = CoherenceChecker::check(&structure);

        assert!(result.issues.iter().any(|i| matches!(i, CoherenceIssue::AwarenessDisconnect)));
    }

    // ========================================================================
    // M: Session Persistence Tests
    // ========================================================================

    #[test]
    fn test_session_state_creation() {
        let session = SessionState::new();
        assert_eq!(session.total_turns, 0);
        assert!(session.phi_history.is_empty());
        assert!(session.session_start_ms > 0);
    }

    #[test]
    fn test_session_state_record_turn() {
        let mut session = SessionState::new();

        session.record_turn(
            vec!["love".to_string(), "consciousness".to_string()],
            SentenceForm::Standard,
            0.65,
            0.5,
            0.3,
        );

        assert_eq!(session.total_turns, 1);
        assert_eq!(session.phi_history.len(), 1);
        assert_eq!(session.emotional_context, (0.5, 0.3));
    }

    #[test]
    fn test_session_state_phi_trend_rising() {
        let mut session = SessionState::new();

        // Simulate rising consciousness
        for i in 1..=5 {
            session.record_turn(
                vec!["topic".to_string()],
                SentenceForm::Standard,
                i as f32 * 0.1, // 0.1, 0.2, 0.3, 0.4, 0.5
                0.0,
                0.0,
            );
        }

        let trend = session.phi_trend();
        assert!(trend > 0.0, "Expected positive trend, got {}", trend);
    }

    #[test]
    fn test_session_state_phi_trend_falling() {
        let mut session = SessionState::new();

        // Simulate falling consciousness
        for i in (1..=5).rev() {
            session.record_turn(
                vec!["topic".to_string()],
                SentenceForm::Standard,
                i as f32 * 0.1, // 0.5, 0.4, 0.3, 0.2, 0.1
                0.0,
                0.0,
            );
        }

        let trend = session.phi_trend();
        assert!(trend < 0.0, "Expected negative trend, got {}", trend);
    }

    #[test]
    fn test_session_state_serialization() {
        let mut session = SessionState::new();
        session.record_turn(vec!["test".to_string()], SentenceForm::Exclamatory, 0.7, 0.8, 0.6);

        // Serialize to JSON
        let json = serde_json::to_string(&session).expect("Failed to serialize");

        // Deserialize back
        let loaded: SessionState = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(loaded.total_turns, 1);
        assert_eq!(loaded.emotional_context, (0.8, 0.6));
    }

    #[test]
    fn test_session_state_save_load_file() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("symthaea_test_session.json");

        // Create and save
        let mut session = SessionState::new();
        session.record_turn(vec!["persistence".to_string()], SentenceForm::Elliptical, 0.5, 0.3, 0.4);
        session.save_to_file(&path).expect("Failed to save");

        // Load
        let loaded = SessionState::load_from_file(&path).expect("Failed to load");

        assert_eq!(loaded.total_turns, 1);
        assert_eq!(loaded.phi_history.len(), 1);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_session_state_load_or_new() {
        let non_existent = std::env::temp_dir().join("non_existent_session_12345.json");

        // Should create new if file doesn't exist
        let session = SessionState::load_or_new(&non_existent);
        assert_eq!(session.total_turns, 0);
    }
}
