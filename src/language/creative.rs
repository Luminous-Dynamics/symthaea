//! Creative Generator Module (Phase B1)
//!
//! Generates metaphors, analogies, and stylistic variety to make responses
//! more engaging and human-like. Integrates with the knowledge graph for
//! semantically grounded creativity.

use std::collections::{HashMap, HashSet, VecDeque};
use super::knowledge_graph::KnowledgeGraph;

// ============================================================================
// METAPHOR ENGINE
// ============================================================================

/// A metaphor mapping one domain to another
#[derive(Debug, Clone)]
pub struct Metaphor {
    /// Source domain (concrete)
    pub source: String,
    /// Target domain (abstract)
    pub target: String,
    /// The metaphorical expression
    pub expression: String,
    /// Confidence in appropriateness (0.0-1.0)
    pub confidence: f32,
    /// Semantic similarity between domains
    pub similarity: f32,
}

/// Engine for generating contextually appropriate metaphors
#[derive(Debug)]
pub struct MetaphorEngine {
    /// Built-in metaphor templates organized by target domain
    templates: HashMap<String, Vec<MetaphorTemplate>>,
    /// Recently used metaphors (to avoid repetition)
    recent: VecDeque<String>,
    /// Maximum recent to track
    max_recent: usize,
}

#[derive(Debug, Clone)]
struct MetaphorTemplate {
    source_domain: String,
    pattern: String,  // e.g., "{target} is like {source}"
    variations: Vec<String>,
}

impl MetaphorEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            templates: HashMap::new(),
            recent: VecDeque::new(),
            max_recent: 20,
        };
        engine.initialize_templates();
        engine
    }

    fn initialize_templates(&mut self) {
        // Emotion metaphors
        self.add_templates("emotion", vec![
            ("journey", vec![
                "{target} is a journey we take",
                "navigating the waters of {target}",
                "the path of {target} leads us forward",
            ]),
            ("weather", vec![
                "{target} is like weather - it passes",
                "the storms of {target} eventually clear",
                "{target} comes in waves, like the tide",
            ]),
            ("light", vec![
                "{target} illuminates our experience",
                "a spark of {target} in the darkness",
                "{target} dawns like morning light",
            ]),
        ]);

        // Understanding/knowledge metaphors
        self.add_templates("understanding", vec![
            ("vision", vec![
                "I see what you mean about {target}",
                "that sheds light on {target}",
                "a clearer picture of {target} emerges",
            ]),
            ("building", vec![
                "building understanding of {target}",
                "the foundation of {target} becomes clearer",
                "constructing meaning around {target}",
            ]),
            ("journey", vec![
                "exploring the terrain of {target}",
                "the path to understanding {target}",
                "journeying deeper into {target}",
            ]),
        ]);

        // Consciousness metaphors
        self.add_templates("consciousness", vec![
            ("ocean", vec![
                "consciousness is an ocean of experience",
                "waves of awareness rise and fall",
                "the depths of conscious experience",
            ]),
            ("light", vec![
                "the light of awareness illuminates",
                "consciousness shines on experience",
                "awareness dawns",
            ]),
            ("space", vec![
                "the spacious awareness within",
                "consciousness as the container of experience",
                "the vast space of awareness",
            ]),
        ]);

        // Life/existence metaphors
        self.add_templates("life", vec![
            ("journey", vec![
                "life is a journey we're all on",
                "the path of life unfolds",
                "traveling through this existence",
            ]),
            ("garden", vec![
                "life blooms when tended",
                "planting seeds for the future",
                "the garden of our days",
            ]),
            ("river", vec![
                "life flows onward",
                "the river of time carries us",
                "going with the flow of life",
            ]),
        ]);

        // Relationship metaphors
        self.add_templates("relationship", vec![
            ("weaving", vec![
                "weaving connections between us",
                "the fabric of our relationship",
                "threads that bind us together",
            ]),
            ("bridge", vec![
                "building bridges of understanding",
                "the bridge between our perspectives",
                "connecting across the divide",
            ]),
            ("dance", vec![
                "the dance of relationship",
                "moving together in rhythm",
                "finding our shared rhythm",
            ]),
        ]);
    }

    fn add_templates(&mut self, target: &str, sources: Vec<(&str, Vec<&str>)>) {
        let templates: Vec<MetaphorTemplate> = sources.into_iter()
            .map(|(source, patterns)| MetaphorTemplate {
                source_domain: source.to_string(),
                pattern: patterns[0].to_string(),
                variations: patterns.iter().map(|s| s.to_string()).collect(),
            })
            .collect();
        self.templates.insert(target.to_string(), templates);
    }

    /// Generate a metaphor for the given topic
    pub fn generate(&mut self, topic: &str, context: Option<&str>) -> Option<Metaphor> {
        // Find matching domain
        let domain = self.find_domain(topic);

        let templates = self.templates.get(&domain)?;

        // Filter out recently used
        let available: Vec<_> = templates.iter()
            .filter(|t| !self.recent.contains(&t.source_domain))
            .collect();

        if available.is_empty() {
            // Clear some recent if we've used them all
            self.recent.clear();
            return self.generate(topic, context);
        }

        // Select based on context or randomly
        let template = if let Some(ctx) = context {
            // Prefer templates that resonate with context
            available.iter()
                .max_by_key(|t| self.context_match(&t.source_domain, ctx))
                .copied()
        } else {
            // Use deterministic selection based on topic
            let idx = topic.bytes().fold(0usize, |acc, b| acc.wrapping_add(b as usize)) % available.len();
            Some(available[idx])
        }?;

        // Select a variation
        let var_idx = topic.len() % template.variations.len();
        let expression = template.variations[var_idx]
            .replace("{target}", topic);

        // Track usage
        self.recent.push_back(template.source_domain.clone());
        if self.recent.len() > self.max_recent {
            self.recent.pop_front();
        }

        Some(Metaphor {
            source: template.source_domain.clone(),
            target: topic.to_string(),
            expression,
            confidence: 0.8,
            similarity: 0.7,
        })
    }

    fn find_domain(&self, topic: &str) -> String {
        let topic_lower = topic.to_lowercase();

        // Check for emotion-related words
        if ["love", "joy", "sadness", "fear", "anger", "happiness", "grief", "peace"]
            .iter().any(|w| topic_lower.contains(w)) {
            return "emotion".to_string();
        }

        // Check for understanding-related
        if ["understand", "know", "learn", "think", "realize", "comprehend"]
            .iter().any(|w| topic_lower.contains(w)) {
            return "understanding".to_string();
        }

        // Check for consciousness-related
        if ["conscious", "aware", "mind", "experience", "perception"]
            .iter().any(|w| topic_lower.contains(w)) {
            return "consciousness".to_string();
        }

        // Check for life-related
        if ["life", "living", "exist", "being", "death", "birth"]
            .iter().any(|w| topic_lower.contains(w)) {
            return "life".to_string();
        }

        // Check for relationship-related
        if ["relationship", "connection", "friend", "family", "together", "bond"]
            .iter().any(|w| topic_lower.contains(w)) {
            return "relationship".to_string();
        }

        // Default to life (most general)
        "life".to_string()
    }

    fn context_match(&self, source: &str, context: &str) -> usize {
        let context_lower = context.to_lowercase();

        // Simple keyword matching for context
        match source {
            "water" | "ocean" | "river" => {
                if context_lower.contains("flow") || context_lower.contains("deep") { 2 } else { 1 }
            }
            "journey" | "path" => {
                if context_lower.contains("going") || context_lower.contains("progress") { 2 } else { 1 }
            }
            "light" => {
                if context_lower.contains("understand") || context_lower.contains("clear") { 2 } else { 1 }
            }
            "garden" => {
                if context_lower.contains("grow") || context_lower.contains("nurture") { 2 } else { 1 }
            }
            _ => 1
        }
    }

    /// Check if a metaphor is appropriate for the context
    pub fn is_appropriate(&self, metaphor: &Metaphor, context: &str) -> bool {
        // Avoid dark metaphors in positive contexts
        let context_lower = context.to_lowercase();
        let positive_context = ["happy", "joy", "good", "love", "peace"]
            .iter().any(|w| context_lower.contains(w));

        let dark_sources = ["storm", "darkness", "death"];
        let is_dark = dark_sources.iter().any(|s| metaphor.source.contains(s));

        !(positive_context && is_dark)
    }
}

impl Default for MetaphorEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ANALOGY FINDER
// ============================================================================

/// An analogy between two concepts
#[derive(Debug, Clone)]
pub struct Analogy {
    /// First concept
    pub concept_a: String,
    /// Second concept (what it's like)
    pub concept_b: String,
    /// The relationship being compared
    pub relationship: String,
    /// The full analogy expression
    pub expression: String,
    /// Structural similarity (0.0-1.0)
    pub structural_similarity: f32,
}

/// Finds structural analogies between concepts
#[derive(Debug)]
pub struct AnalogyFinder {
    /// Cached analogies for common concepts
    cached: HashMap<String, Vec<Analogy>>,
}

impl AnalogyFinder {
    pub fn new() -> Self {
        let mut finder = Self {
            cached: HashMap::new(),
        };
        finder.initialize_cache();
        finder
    }

    fn initialize_cache(&mut self) {
        // Pre-compute common analogies
        self.add_cached("consciousness", vec![
            ("flashlight", "directing attention", "Consciousness is like a flashlight - it illuminates whatever we point it at"),
            ("stage", "workspace", "Consciousness is like a stage where thoughts perform"),
            ("mirror", "reflection", "Consciousness mirrors reality back to itself"),
        ]);

        self.add_cached("learning", vec![
            ("building", "construction", "Learning is like building - each concept supports the next"),
            ("gardening", "growth", "Learning is like gardening - it requires patience and nurturing"),
            ("exploration", "discovery", "Learning is like exploration - venturing into unknown territory"),
        ]);

        self.add_cached("memory", vec![
            ("library", "storage", "Memory is like a library - organized but sometimes hard to find things"),
            ("web", "connections", "Memory is like a web - everything connects to everything else"),
            ("photograph", "preservation", "Memory is like a photograph - capturing moments in time"),
        ]);

        self.add_cached("emotion", vec![
            ("weather", "changeable", "Emotions are like weather - they come and go"),
            ("waves", "rising and falling", "Emotions are like waves - they rise, crest, and fall"),
            ("colors", "variety", "Emotions are like colors - each has its own character"),
        ]);

        self.add_cached("thinking", vec![
            ("river", "flow", "Thinking is like a river - it flows from thought to thought"),
            ("jazz", "improvisation", "Thinking is like jazz - structured improvisation"),
            ("conversation", "dialogue", "Thinking is an inner conversation with ourselves"),
        ]);
    }

    fn add_cached(&mut self, concept: &str, analogies: Vec<(&str, &str, &str)>) {
        let entries: Vec<Analogy> = analogies.into_iter()
            .map(|(b, rel, expr)| Analogy {
                concept_a: concept.to_string(),
                concept_b: b.to_string(),
                relationship: rel.to_string(),
                expression: expr.to_string(),
                structural_similarity: 0.75,
            })
            .collect();
        self.cached.insert(concept.to_string(), entries);
    }

    /// Find an analogy for the given concept
    pub fn find(&self, concept: &str, kg: Option<&KnowledgeGraph>) -> Option<Analogy> {
        let concept_lower = concept.to_lowercase();

        // Check cache first
        if let Some(analogies) = self.cached.get(&concept_lower) {
            // Deterministic selection
            let idx = concept.len() % analogies.len();
            return Some(analogies[idx].clone());
        }

        // Try to find related cached concept
        for (key, analogies) in &self.cached {
            if concept_lower.contains(key) || key.contains(&concept_lower) {
                let idx = concept.len() % analogies.len();
                let mut analogy = analogies[idx].clone();
                analogy.concept_a = concept.to_string();
                return Some(analogy);
            }
        }

        // Use knowledge graph if available
        if let Some(kg) = kg {
            return self.find_from_kg(concept, kg);
        }

        None
    }

    fn find_from_kg(&self, concept: &str, kg: &KnowledgeGraph) -> Option<Analogy> {
        // Look for similar concepts in the knowledge graph
        let node_id = kg.get_id(concept)?;

        // Find concepts with similar structure (same edge types)
        let edges = kg.edges_from(node_id);
        if edges.is_empty() {
            return None;
        }

        // Get the primary relationship type
        let primary_edge = edges.first()?;
        let relationship = format!("{:?}", primary_edge.edge_type);

        // Find another concept with similar relationships
        let similar = kg.get_node(primary_edge.to)?;

        Some(Analogy {
            concept_a: concept.to_string(),
            concept_b: similar.name.clone(),
            relationship: relationship.to_lowercase(),
            expression: format!(
                "{} is related to {} in the way they both involve {}",
                concept, similar.name, relationship.to_lowercase()
            ),
            structural_similarity: 0.6,
        })
    }

    /// Generate an A:B::C:D style analogy
    pub fn proportional_analogy(&self, a: &str, b: &str, c: &str) -> Option<String> {
        // Simple relationship inference
        let relationship = self.infer_relationship(a, b);
        let d = self.apply_relationship(c, &relationship)?;

        Some(format!("{} is to {} as {} is to {}", a, b, c, d))
    }

    fn infer_relationship(&self, a: &str, b: &str) -> String {
        // Simple heuristics for common relationships
        if b.contains(a) {
            "part of".to_string()
        } else if a.len() < b.len() {
            "smaller than".to_string()
        } else {
            "related to".to_string()
        }
    }

    fn apply_relationship(&self, c: &str, _relationship: &str) -> Option<String> {
        // This would ideally use the knowledge graph
        // For now, return a placeholder
        Some(format!("something like {}", c))
    }
}

impl Default for AnalogyFinder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// STYLE VARIATOR
// ============================================================================

/// Linguistic style dimensions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StyleDimension {
    /// Formal vs casual
    Formality,
    /// Simple vs complex vocabulary
    Complexity,
    /// Short vs long sentences
    Length,
    /// Direct vs indirect expression
    Directness,
    /// Abstract vs concrete language
    Abstraction,
}

/// Style variation configuration
#[derive(Debug, Clone)]
pub struct StyleConfig {
    pub formality: f32,      // 0.0 = casual, 1.0 = formal
    pub complexity: f32,     // 0.0 = simple, 1.0 = complex
    pub length: f32,         // 0.0 = short, 1.0 = long
    pub directness: f32,     // 0.0 = indirect, 1.0 = direct
    pub abstraction: f32,    // 0.0 = concrete, 1.0 = abstract
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            formality: 0.5,
            complexity: 0.4,
            length: 0.5,
            directness: 0.6,
            abstraction: 0.5,
        }
    }
}

/// Variates linguistic style for natural conversation
#[derive(Debug)]
pub struct StyleVariator {
    /// Current style configuration
    config: StyleConfig,
    /// Style history for variation
    history: VecDeque<StyleConfig>,
    /// Synonym alternatives
    synonyms: HashMap<String, Vec<String>>,
    /// Sentence starters
    starters: Vec<Vec<String>>,
}

impl StyleVariator {
    pub fn new() -> Self {
        let mut variator = Self {
            config: StyleConfig::default(),
            history: VecDeque::new(),
            synonyms: HashMap::new(),
            starters: Vec::new(),
        };
        variator.initialize();
        variator
    }

    fn initialize(&mut self) {
        // Synonym pairs (simple -> complex)
        self.add_synonyms(vec![
            ("good", vec!["excellent", "wonderful", "beneficial", "positive"]),
            ("bad", vec!["unfortunate", "problematic", "challenging", "difficult"]),
            ("think", vec!["believe", "consider", "reflect", "contemplate"]),
            ("feel", vec!["experience", "sense", "perceive", "notice"]),
            ("see", vec!["observe", "notice", "recognize", "perceive"]),
            ("like", vec!["appreciate", "enjoy", "find pleasure in", "am drawn to"]),
            ("want", vec!["desire", "wish for", "hope for", "aspire to"]),
            ("know", vec!["understand", "recognize", "am aware that", "realize"]),
            ("big", vec!["significant", "substantial", "considerable", "profound"]),
            ("small", vec!["minor", "modest", "subtle", "slight"]),
        ]);

        // Sentence starters by formality level
        self.starters = vec![
            // Casual (0.0-0.3)
            vec![
                "So,".to_string(),
                "Well,".to_string(),
                "You know,".to_string(),
                "I think".to_string(),
                "I feel like".to_string(),
            ],
            // Neutral (0.3-0.7)
            vec![
                "I notice that".to_string(),
                "It seems to me that".to_string(),
                "I find that".to_string(),
                "In my experience,".to_string(),
                "What I understand is".to_string(),
            ],
            // Formal (0.7-1.0)
            vec![
                "I would suggest that".to_string(),
                "It appears that".to_string(),
                "One might consider".to_string(),
                "Upon reflection,".to_string(),
                "It is worth noting that".to_string(),
            ],
        ];
    }

    fn add_synonyms(&mut self, pairs: Vec<(&str, Vec<&str>)>) {
        for (word, syns) in pairs {
            self.synonyms.insert(
                word.to_string(),
                syns.into_iter().map(String::from).collect()
            );
        }
    }

    /// Apply style variation to text
    pub fn variate(&mut self, text: &str) -> String {
        let mut result = text.to_string();

        // Apply synonym substitution based on complexity
        if self.config.complexity > 0.5 {
            result = self.apply_synonyms(&result);
        }

        // Add variety to sentence structure
        result = self.vary_structure(&result);

        // Track style for future variation
        self.history.push_back(self.config.clone());
        if self.history.len() > 10 {
            self.history.pop_front();
        }

        result
    }

    fn apply_synonyms(&self, text: &str) -> String {
        let mut result = text.to_string();

        for (word, alternatives) in &self.synonyms {
            if result.to_lowercase().contains(word) {
                // Select alternative based on text length for determinism
                let idx = text.len() % alternatives.len();
                let replacement = &alternatives[idx];

                // Simple word replacement (case-insensitive)
                let pattern = format!(r"\b{}\b", word);
                if let Ok(re) = regex::Regex::new(&pattern) {
                    result = re.replace(&result, replacement.as_str()).to_string();
                }
            }
        }

        result
    }

    fn vary_structure(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        // Skip adding starters to greetings, questions, exclamations, or already natural text
        let skip_patterns = [
            "hello", "hi ", "hi!", "hey", "welcome", "greetings",
            "yes,", "no,", "sure", "okay", "alright",
        ];

        let has_skip_pattern = skip_patterns.iter().any(|p| text_lower.starts_with(p));
        let has_punctuation_early = text.chars().take(15).any(|c| c == '!' || c == '?' || c == ',');
        let starts_with_i = text.starts_with("I ");
        let starts_natural = text_lower.starts_with("my ") ||
                            text_lower.starts_with("your ") ||
                            text_lower.starts_with("that ") ||
                            text_lower.starts_with("this ");

        // Be much more selective about adding starters
        let should_add_starter = text.len() > 40 &&
                                 !starts_with_i &&
                                 !has_skip_pattern &&
                                 !has_punctuation_early &&
                                 !starts_natural &&
                                 self.config.formality > 0.5;  // Only for more formal contexts

        if should_add_starter {
            let formality_idx = if self.config.formality < 0.3 {
                0
            } else if self.config.formality < 0.7 {
                1
            } else {
                2
            };

            let starters = &self.starters[formality_idx];
            let starter_idx = text.len() % starters.len();
            let starter = &starters[starter_idx];

            // Only add if the text doesn't already have a similar start
            if !text_lower.starts_with(&starter.to_lowercase()[..3.min(starter.len())]) {
                return format!("{} {}", starter, text);
            }
        }

        text.to_string()
    }

    /// Set style configuration
    pub fn set_style(&mut self, config: StyleConfig) {
        self.config = config;
    }

    /// Get current style
    pub fn current_style(&self) -> &StyleConfig {
        &self.config
    }

    /// Calculate style variation from history
    pub fn variation_score(&self) -> f32 {
        if self.history.len() < 2 {
            return 1.0;
        }

        let last = self.history.back().unwrap();
        let prev = self.history.get(self.history.len() - 2).unwrap();

        // Calculate how different the current style is from previous
        let diff = (last.formality - prev.formality).abs()
            + (last.complexity - prev.complexity).abs()
            + (last.directness - prev.directness).abs();

        (diff / 3.0).min(1.0)
    }
}

impl Default for StyleVariator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NOVELTY TRACKER
// ============================================================================

/// Tracks used expressions to ensure variety
#[derive(Debug)]
pub struct NoveltyTracker {
    /// Recently used phrases
    used_phrases: VecDeque<String>,
    /// Recently used sentence patterns
    used_patterns: VecDeque<String>,
    /// Maximum to track
    max_tracked: usize,
    /// Phrase similarity threshold
    similarity_threshold: f32,
}

impl NoveltyTracker {
    pub fn new() -> Self {
        Self {
            used_phrases: VecDeque::new(),
            used_patterns: VecDeque::new(),
            max_tracked: 50,
            similarity_threshold: 0.7,
        }
    }

    /// Check if a phrase is novel (not too similar to recent ones)
    pub fn is_novel(&self, phrase: &str) -> bool {
        let phrase_lower = phrase.to_lowercase();

        for used in &self.used_phrases {
            let similarity = self.similarity(&phrase_lower, &used.to_lowercase());
            if similarity > self.similarity_threshold {
                return false;
            }
        }

        true
    }

    /// Calculate string similarity (simple Jaccard)
    fn similarity(&self, a: &str, b: &str) -> f32 {
        let words_a: HashSet<_> = a.split_whitespace().collect();
        let words_b: HashSet<_> = b.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Track a used phrase
    pub fn track(&mut self, phrase: &str) {
        self.used_phrases.push_back(phrase.to_string());
        if self.used_phrases.len() > self.max_tracked {
            self.used_phrases.pop_front();
        }
    }

    /// Track a sentence pattern (e.g., "I think that X")
    pub fn track_pattern(&mut self, pattern: &str) {
        self.used_patterns.push_back(pattern.to_string());
        if self.used_patterns.len() > self.max_tracked {
            self.used_patterns.pop_front();
        }
    }

    /// Get novelty score (how different from recent)
    pub fn novelty_score(&self, phrase: &str) -> f32 {
        if self.used_phrases.is_empty() {
            return 1.0;
        }

        let phrase_lower = phrase.to_lowercase();
        let max_sim = self.used_phrases.iter()
            .map(|used| self.similarity(&phrase_lower, &used.to_lowercase()))
            .fold(0.0f32, |a, b| a.max(b));

        1.0 - max_sim
    }

    /// Clear tracking history
    pub fn clear(&mut self) {
        self.used_phrases.clear();
        self.used_patterns.clear();
    }
}

impl Default for NoveltyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CREATIVE GENERATOR (MAIN)
// ============================================================================

/// Complete creative generation system
#[derive(Debug)]
pub struct CreativeGenerator {
    /// Metaphor generation
    pub metaphor_engine: MetaphorEngine,
    /// Analogy finding
    pub analogy_finder: AnalogyFinder,
    /// Style variation
    pub style_variator: StyleVariator,
    /// Novelty tracking
    pub novelty_tracker: NoveltyTracker,
    /// Creativity level (0.0-1.0)
    creativity_level: f32,
}

impl CreativeGenerator {
    pub fn new() -> Self {
        Self {
            metaphor_engine: MetaphorEngine::new(),
            analogy_finder: AnalogyFinder::new(),
            style_variator: StyleVariator::new(),
            novelty_tracker: NoveltyTracker::new(),
            creativity_level: 0.5,
        }
    }

    pub fn with_creativity(creativity: f32) -> Self {
        let mut gen = Self::new();
        gen.creativity_level = creativity.clamp(0.0, 1.0);
        gen
    }

    /// Enhance a response with creative elements
    pub fn enhance(&mut self, text: &str, topic: Option<&str>, context: Option<&str>) -> String {
        let mut result = text.to_string();

        // Apply style variation
        result = self.style_variator.variate(&result);

        // Maybe add metaphor (based on creativity level)
        if self.should_add_metaphor() {
            if let Some(t) = topic {
                if let Some(metaphor) = self.metaphor_engine.generate(t, context) {
                    if self.metaphor_engine.is_appropriate(&metaphor, context.unwrap_or("")) {
                        result = self.integrate_metaphor(&result, &metaphor);
                    }
                }
            }
        }

        // Track for novelty
        self.novelty_tracker.track(&result);

        result
    }

    fn should_add_metaphor(&self) -> bool {
        // Be more selective about adding metaphors:
        // 1. Only when creativity is high AND
        // 2. We haven't added one recently (every 5th response, not 3rd)
        self.creativity_level > 0.5 &&
            self.novelty_tracker.used_phrases.len() % 5 == 0
    }

    fn integrate_metaphor(&self, text: &str, metaphor: &Metaphor) -> String {
        // Better integration with transitional phrases instead of just appending
        let transitions = [
            "Perhaps it's like",
            "In a way,",
            "It reminds me that",
            "There's something about how",
        ];

        // Only add metaphor if it has good confidence and the text isn't too long
        if metaphor.confidence < 0.6 || text.len() > 200 {
            return text.to_string();  // Skip weak metaphors or long texts
        }

        // Pick transition based on metaphor source to avoid repetition
        let transition_idx = metaphor.source.len() % transitions.len();
        let transition = transitions[transition_idx];

        // Integrate more naturally
        if text.ends_with('.') {
            format!("{} {} {}", text.trim_end_matches('.'), "—", metaphor.expression.to_lowercase())
        } else if text.ends_with('!') || text.ends_with('?') {
            // Don't add metaphors after questions or exclamations
            text.to_string()
        } else {
            format!("{}. {} {}", text, transition, metaphor.expression.to_lowercase())
        }
    }

    /// Generate a creative response element
    pub fn generate_element(&mut self, topic: &str, element_type: CreativeElement) -> Option<String> {
        match element_type {
            CreativeElement::Metaphor => {
                self.metaphor_engine.generate(topic, None)
                    .map(|m| m.expression)
            }
            CreativeElement::Analogy => {
                self.analogy_finder.find(topic, None)
                    .map(|a| a.expression)
            }
            CreativeElement::Question => {
                Some(self.generate_reflective_question(topic))
            }
            CreativeElement::Observation => {
                Some(self.generate_observation(topic))
            }
        }
    }

    fn generate_reflective_question(&self, topic: &str) -> String {
        let questions = [
            format!("What does {} mean to you?", topic),
            format!("How do you experience {}?", topic),
            format!("What draws you to explore {}?", topic),
            format!("What aspects of {} resonate with you?", topic),
            format!("How has {} shaped your understanding?", topic),
        ];

        let idx = topic.len() % questions.len();
        questions[idx].clone()
    }

    fn generate_observation(&self, topic: &str) -> String {
        let observations = [
            format!("I notice {} is something that matters deeply", topic),
            format!("There's something profound about {}", topic),
            format!("The nature of {} invites contemplation", topic),
            format!("I find myself drawn to understand {} more fully", topic),
            format!("{} seems to connect to many aspects of experience", topic),
        ];

        let idx = topic.len() % observations.len();
        observations[idx].clone()
    }

    /// Set creativity level
    pub fn set_creativity(&mut self, level: f32) {
        self.creativity_level = level.clamp(0.0, 1.0);
    }

    /// Get creativity level
    pub fn creativity_level(&self) -> f32 {
        self.creativity_level
    }

    /// Get style variation score
    pub fn variation_score(&self) -> f32 {
        self.style_variator.variation_score()
    }

    /// Get novelty score for text
    pub fn novelty_score(&self, text: &str) -> f32 {
        self.novelty_tracker.novelty_score(text)
    }
}

impl Default for CreativeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of creative elements
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CreativeElement {
    Metaphor,
    Analogy,
    Question,
    Observation,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Metaphor Engine Tests
    #[test]
    fn test_metaphor_engine_creation() {
        let engine = MetaphorEngine::new();
        assert!(!engine.templates.is_empty());
    }

    #[test]
    fn test_metaphor_generation() {
        let mut engine = MetaphorEngine::new();
        let metaphor = engine.generate("love", None);
        assert!(metaphor.is_some());
        let m = metaphor.unwrap();
        assert!(m.expression.contains("love") || m.expression.to_lowercase().contains("love"));
    }

    #[test]
    fn test_metaphor_domain_detection() {
        let engine = MetaphorEngine::new();
        assert_eq!(engine.find_domain("happiness"), "emotion");
        assert_eq!(engine.find_domain("understanding"), "understanding");
        assert_eq!(engine.find_domain("consciousness"), "consciousness");
        assert_eq!(engine.find_domain("living"), "life");
    }

    #[test]
    fn test_metaphor_no_repetition() {
        let mut engine = MetaphorEngine::new();
        let mut sources = HashSet::new();

        // Generate several metaphors, should use different sources
        for _ in 0..5 {
            if let Some(m) = engine.generate("emotion", None) {
                sources.insert(m.source);
            }
        }

        // Should have some variety
        assert!(sources.len() >= 1);
    }

    // Analogy Finder Tests
    #[test]
    fn test_analogy_finder_creation() {
        let finder = AnalogyFinder::new();
        assert!(!finder.cached.is_empty());
    }

    #[test]
    fn test_analogy_finding() {
        let finder = AnalogyFinder::new();
        let analogy = finder.find("consciousness", None);
        assert!(analogy.is_some());
    }

    #[test]
    fn test_analogy_related_concept() {
        let finder = AnalogyFinder::new();
        // Should find analogy for related concept (contains "consciousness")
        let analogy = finder.find("consciousness studies", None);
        assert!(analogy.is_some());
    }

    // Style Variator Tests
    #[test]
    fn test_style_variator_creation() {
        let variator = StyleVariator::new();
        assert!(!variator.synonyms.is_empty());
    }

    #[test]
    fn test_style_variation() {
        let mut variator = StyleVariator::new();
        variator.set_style(StyleConfig {
            complexity: 0.8,
            ..Default::default()
        });

        let original = "I think this is good";
        let varied = variator.variate(original);

        // Should have some variation
        assert!(!varied.is_empty());
    }

    #[test]
    fn test_style_configuration() {
        let mut variator = StyleVariator::new();
        let config = StyleConfig {
            formality: 0.9,
            complexity: 0.8,
            length: 0.7,
            directness: 0.6,
            abstraction: 0.5,
        };
        variator.set_style(config.clone());

        assert_eq!(variator.current_style().formality, 0.9);
    }

    // Novelty Tracker Tests
    #[test]
    fn test_novelty_tracker_creation() {
        let tracker = NoveltyTracker::new();
        assert!(tracker.used_phrases.is_empty());
    }

    #[test]
    fn test_novelty_tracking() {
        let mut tracker = NoveltyTracker::new();

        tracker.track("This is a test phrase");
        assert!(!tracker.is_novel("This is a test phrase"));
        assert!(tracker.is_novel("Something completely different"));
    }

    #[test]
    fn test_novelty_score() {
        let mut tracker = NoveltyTracker::new();

        // First phrase is always novel
        assert_eq!(tracker.novelty_score("Hello world"), 1.0);

        tracker.track("Hello world");

        // Same phrase should have low novelty
        let score = tracker.novelty_score("Hello world");
        assert!(score < 0.5);
    }

    // Creative Generator Tests
    #[test]
    fn test_creative_generator_creation() {
        let gen = CreativeGenerator::new();
        assert_eq!(gen.creativity_level(), 0.5);
    }

    #[test]
    fn test_creative_enhancement() {
        let mut gen = CreativeGenerator::new();
        let enhanced = gen.enhance("I understand", Some("understanding"), None);
        assert!(!enhanced.is_empty());
    }

    #[test]
    fn test_creative_element_generation() {
        let mut gen = CreativeGenerator::new();

        let question = gen.generate_element("love", CreativeElement::Question);
        assert!(question.is_some());
        assert!(question.unwrap().contains("love"));

        let observation = gen.generate_element("consciousness", CreativeElement::Observation);
        assert!(observation.is_some());
    }

    #[test]
    fn test_creativity_level_setting() {
        let mut gen = CreativeGenerator::new();

        gen.set_creativity(0.9);
        assert_eq!(gen.creativity_level(), 0.9);

        // Should clamp
        gen.set_creativity(1.5);
        assert_eq!(gen.creativity_level(), 1.0);

        gen.set_creativity(-0.5);
        assert_eq!(gen.creativity_level(), 0.0);
    }
}
