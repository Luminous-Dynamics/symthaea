//! Moral Semantic Parser
//!
//! Extracts moral primitives (AGENT, PATIENT, ACTION, INTENT, CONSENT, etc.)
//! from natural language text using pattern matching and keyword analysis.
//!
//! This is a lightweight parser designed for the ETHICS benchmark format.
//! For production use, integrate with a full SRL model (spaCy, AllenNLP).
//!
//! # Design
//!
//! The parser uses a multi-stage approach:
//! 1. **Tokenization**: Split text into words
//! 2. **Pattern Matching**: Identify consent, negation, magnitude markers
//! 3. **Role Extraction**: Extract agent, action, patient from structure
//! 4. **Composition**: Build moral algebra structures
//!
//! # Example
//!
//! ```ignore
//! let parser = MoralParser::new();
//! let parsed = parser.parse("I discussed my daughter's health without asking first");
//! // parsed.consent == ConsentState::Absent (detected "without asking")
//! // parsed.action == "discussed"
//! // parsed.patient == "daughter's health"
//! ```

use std::collections::HashSet;
use super::moral_algebra::{
    ConsentState, Magnitude, MoralAlgebra, MoralIntent, MoralVerdict,
    ProportionalityJudgment, ExcuseJudgment, MoralJudgment,
};
use symthaea_core::hdc::ContinuousHV;

/// Parsed moral structure from text
#[derive(Debug, Clone)]
pub struct ParsedMoralScenario {
    /// Original text
    pub text: String,

    /// Detected agent (who acts)
    pub agent: Option<String>,

    /// Detected action (what happens)
    pub action: Option<String>,

    /// Detected patient (who is affected)
    pub patient: Option<String>,

    /// Detected intent
    pub intent: MoralIntent,

    /// Detected consent state
    pub consent: ConsentState,

    /// Detected magnitude (for proportionality)
    pub magnitude: Option<Magnitude>,

    /// Whether negation was detected
    pub has_negation: bool,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

impl Default for ParsedMoralScenario {
    fn default() -> Self {
        Self {
            text: String::new(),
            agent: None,
            action: None,
            patient: None,
            intent: MoralIntent::Unknown,
            consent: ConsentState::Implied,
            magnitude: None,
            has_negation: false,
            confidence: 0.0,
        }
    }
}

/// Lightweight moral semantic parser
#[derive(Debug, Clone)]
pub struct MoralParser {
    /// Words indicating good intent
    good_intent_words: HashSet<String>,

    /// Words indicating bad intent
    bad_intent_words: HashSet<String>,

    /// Words indicating consent given
    consent_given_words: HashSet<String>,

    /// Words indicating consent absent/denied
    consent_absent_words: HashSet<String>,

    /// Negation words
    negation_words: HashSet<String>,

    /// Magnitude indicators (small)
    small_magnitude_words: HashSet<String>,

    /// Magnitude indicators (large)
    large_magnitude_words: HashSet<String>,

    /// Common action verbs
    action_verbs: HashSet<String>,
}

impl MoralParser {
    /// Create a new moral parser with default vocabularies
    pub fn new() -> Self {
        Self {
            good_intent_words: [
                "help", "helped", "helping", "helps",
                "save", "saved", "saving", "saves",
                "protect", "protected", "protecting",
                "care", "cared", "caring", "cares",
                "support", "supported", "supporting",
                "assist", "assisted", "assisting",
                "kind", "kindly", "generous", "generously",
                "compassion", "compassionate", "empathy",
                "love", "loved", "loving",
            ].iter().map(|s| s.to_string()).collect(),

            bad_intent_words: [
                "harm", "harmed", "harming", "harms",
                "hurt", "hurting", "hurts",
                "steal", "stole", "stealing", "steals",
                "lie", "lied", "lying", "lies",
                "cheat", "cheated", "cheating", "cheats",
                "deceive", "deceived", "deceiving",
                "betray", "betrayed", "betraying",
                "abuse", "abused", "abusing",
                "cruel", "cruelly", "malicious",
                "selfish", "selfishly",
            ].iter().map(|s| s.to_string()).collect(),

            consent_given_words: [
                "asked", "asking", "permission", "permitted",
                "consent", "consented", "agreed", "agreeing",
                "allowed", "allowing", "approved", "approving",
                "with permission", "after asking",
            ].iter().map(|s| s.to_string()).collect(),

            consent_absent_words: [
                "without asking", "without permission", "without consent",
                "didn't ask", "did not ask", "never asked",
                "secretly", "behind", "without telling",
                "without informing", "without notifying",
            ].iter().map(|s| s.to_string()).collect(),

            negation_words: [
                "not", "no", "never", "don't", "doesn't", "didn't",
                "won't", "wouldn't", "couldn't", "shouldn't",
                "without", "none", "nothing", "nobody",
                "neither", "nor", "refuse", "refused",
            ].iter().map(|s| s.to_string()).collect(),

            small_magnitude_words: [
                "small", "little", "minor", "tiny", "slight",
                "once", "briefly", "quickly", "simple",
                "easy", "basic", "minimal",
            ].iter().map(|s| s.to_string()).collect(),

            large_magnitude_words: [
                "large", "big", "major", "huge", "significant",
                "always", "daily", "constantly", "extensive",
                "substantial", "considerable", "great",
                "brand new", "expensive", "valuable",
            ].iter().map(|s| s.to_string()).collect(),

            action_verbs: [
                // Communication verbs
                "discuss", "discussed", "discussing",
                "tell", "told", "telling", "tells",
                "share", "shared", "sharing", "shares",
                "say", "said", "saying", "says",
                "talk", "talked", "talking", "talks",
                "speak", "spoke", "speaking", "speaks",
                "ask", "asked", "asking", "asks",
                "answer", "answered", "answering",
                // Transfer verbs
                "give", "gave", "giving", "gives",
                "take", "took", "taking", "takes",
                "send", "sent", "sending", "sends",
                "receive", "received", "receiving",
                // Moral action verbs
                "help", "helped", "helping", "helps",
                "harm", "harmed", "harming", "harms",
                "hurt", "hurting", "hurts",
                "save", "saved", "saving", "saves",
                "protect", "protected", "protecting",
                "steal", "stole", "stealing", "steals",
                "lie", "lied", "lying",
                "cheat", "cheated", "cheating",
                "betray", "betrayed", "betraying",
                // Possession/entitlement verbs
                "deserve", "deserved", "deserving", "deserves",
                "earn", "earned", "earning", "earns",
                "own", "owned", "owning", "owns",
                "owe", "owed", "owing", "owes",
                // Physical action verbs
                "clean", "cleaned", "cleaning", "cleans",
                "prepare", "prepared", "preparing",
                "make", "made", "making", "makes",
                "do", "did", "doing", "does",
                "use", "used", "using", "uses",
                "work", "worked", "working", "works",
                "walk", "walked", "walking", "walks",
                "run", "ran", "running", "runs",
                "buy", "bought", "buying", "buys",
                "sell", "sold", "selling", "sells",
                // Emotional/relational verbs
                "love", "loved", "loving", "loves",
                "hate", "hated", "hating", "hates",
                "trust", "trusted", "trusting", "trusts",
                "forgive", "forgave", "forgiving", "forgives",
                "ignore", "ignored", "ignoring", "ignores",
                "respect", "respected", "respecting",
                // Set up (multi-word)
                "set up", "setting up",
            ].iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Parse a text into a moral scenario structure
    pub fn parse(&self, text: &str) -> ParsedMoralScenario {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let mut scenario = ParsedMoralScenario {
            text: text.to_string(),
            ..Default::default()
        };

        // Detect consent state (check phrases first, then words)
        scenario.consent = self.detect_consent(&lower);

        // Detect negation
        scenario.has_negation = self.detect_negation(&words);

        // Detect intent
        scenario.intent = self.detect_intent(&words);

        // Detect magnitude
        scenario.magnitude = self.detect_magnitude(&words);

        // Extract action
        scenario.action = self.extract_action(&words);

        // Extract agent and patient (simplified heuristic)
        let (agent, patient) = self.extract_agent_patient(&lower);
        scenario.agent = agent;
        scenario.patient = patient;

        // Calculate confidence based on what was detected
        scenario.confidence = self.calculate_confidence(&scenario);

        scenario
    }

    /// Detect consent state from text
    fn detect_consent(&self, text: &str) -> ConsentState {
        // Check for absent consent phrases first (more specific)
        for phrase in &self.consent_absent_words {
            if text.contains(phrase.as_str()) {
                return ConsentState::Absent;
            }
        }

        // Check for given consent phrases
        for phrase in &self.consent_given_words {
            if text.contains(phrase.as_str()) {
                return ConsentState::Given;
            }
        }

        // Default to implied
        ConsentState::Implied
    }

    /// Detect negation in words
    fn detect_negation(&self, words: &[&str]) -> bool {
        words.iter().any(|w| self.negation_words.contains(*w))
    }

    /// Detect intent from words
    fn detect_intent(&self, words: &[&str]) -> MoralIntent {
        let good_count = words.iter()
            .filter(|w| self.good_intent_words.contains(**w))
            .count();
        let bad_count = words.iter()
            .filter(|w| self.bad_intent_words.contains(**w))
            .count();

        if good_count > bad_count && good_count > 0 {
            MoralIntent::Good
        } else if bad_count > good_count && bad_count > 0 {
            MoralIntent::Bad
        } else if good_count == 0 && bad_count == 0 {
            MoralIntent::Neutral
        } else {
            MoralIntent::Unknown
        }
    }

    /// Detect magnitude from words
    fn detect_magnitude(&self, words: &[&str]) -> Option<Magnitude> {
        let small_count = words.iter()
            .filter(|w| self.small_magnitude_words.contains(**w))
            .count();
        let large_count = words.iter()
            .filter(|w| self.large_magnitude_words.contains(**w))
            .count();

        if large_count > small_count {
            Some(Magnitude::Large)
        } else if small_count > large_count {
            Some(Magnitude::Small)
        } else if small_count > 0 || large_count > 0 {
            Some(Magnitude::Medium)
        } else {
            None
        }
    }

    /// Extract the main action verb
    fn extract_action(&self, words: &[&str]) -> Option<String> {
        for word in words {
            if self.action_verbs.contains(*word) {
                return Some(word.to_string());
            }
        }
        None
    }

    /// Extract agent and patient (simplified heuristic)
    fn extract_agent_patient(&self, text: &str) -> (Option<String>, Option<String>) {
        // Look for "I" as agent
        let agent = if text.starts_with("i ") || text.contains(" i ") {
            Some("I".to_string())
        } else {
            None
        };

        // Look for possessive patterns for patient: "my X's Y" or "the X"
        let patient = if let Some(pos) = text.find("my ") {
            let rest = &text[pos + 3..];
            let end = rest.find(|c: char| c == ',' || c == '.' || c == ' ').unwrap_or(rest.len());
            Some(rest[..end.min(30)].to_string())
        } else if let Some(pos) = text.find("the ") {
            let rest = &text[pos + 4..];
            let end = rest.find(|c: char| c == ',' || c == '.' || c == ' ').unwrap_or(rest.len());
            Some(rest[..end.min(30)].to_string())
        } else {
            None
        };

        (agent, patient)
    }

    /// Calculate confidence score based on what was detected
    fn calculate_confidence(&self, scenario: &ParsedMoralScenario) -> f32 {
        let mut score = 0.0;

        if scenario.agent.is_some() { score += 0.15; }
        if scenario.action.is_some() { score += 0.25; }
        if scenario.patient.is_some() { score += 0.15; }
        if scenario.intent != MoralIntent::Unknown { score += 0.20; }
        if scenario.consent != ConsentState::Implied { score += 0.15; }
        if scenario.magnitude.is_some() { score += 0.10; }

        score
    }

    /// Parse and encode a scenario using the moral algebra
    pub fn parse_and_encode(&self, text: &str, algebra: &MoralAlgebra) -> EncodedMoralScenario {
        let parsed = self.parse(text);

        // Encode consent action if we have action and patient
        let consent_hv = if let (Some(action), Some(patient)) = (&parsed.action, &parsed.patient) {
            Some(algebra.encode_consent_action(action, patient, parsed.consent))
        } else {
            None
        };

        // Encode full action structure if we have all components
        let action_hv = if let (Some(agent), Some(action), Some(patient)) =
            (&parsed.agent, &parsed.action, &parsed.patient) {
            Some(algebra.encode_action_structure(agent, action, patient, parsed.intent))
        } else {
            None
        };

        // Apply negation if detected
        let final_hv = if parsed.has_negation {
            action_hv.map(|hv| algebra.negate(&hv))
        } else {
            action_hv
        };

        EncodedMoralScenario {
            parsed,
            consent_hv,
            action_hv: final_hv,
        }
    }
}

impl Default for MoralParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoded moral scenario with hypervectors
#[derive(Debug, Clone)]
pub struct EncodedMoralScenario {
    /// The parsed structure
    pub parsed: ParsedMoralScenario,

    /// Consent-focused encoding (if available)
    pub consent_hv: Option<ContinuousHV>,

    /// Full action structure encoding (if available)
    pub action_hv: Option<ContinuousHV>,
}

impl EncodedMoralScenario {
    /// Judge this scenario using the moral algebra
    ///
    /// Returns a MoralJudgment that considers:
    /// - Intent (good/bad/neutral)
    /// - Consent state (violation if absent/denied for sensitive actions)
    pub fn judge(&self, algebra: &MoralAlgebra) -> Option<MoralJudgment> {
        self.action_hv.as_ref().map(|hv| {
            let mut judgment = algebra.judge_action(hv);

            // Override verdict if there's a consent violation
            if self.is_consent_violation() {
                judgment.consent_violation_similarity = 1.0;
                judgment.verdict = super::moral_algebra::MoralVerdict::ConsentViolation;
            }

            judgment
        })
    }

    /// Check if this scenario represents a consent violation
    ///
    /// A consent violation occurs when:
    /// - Consent is Absent or Denied AND
    /// - The action affects another person (patient detected)
    pub fn is_consent_violation(&self) -> bool {
        use super::moral_algebra::ConsentState;

        // Must have a patient (someone being affected) for consent to matter
        if self.parsed.patient.is_none() {
            return false;
        }

        matches!(self.parsed.consent, ConsentState::Absent | ConsentState::Denied)
    }

    /// Check for consent violation (returns similarity score for compatibility)
    ///
    /// Uses direct consent state check rather than HV similarity
    pub fn check_consent_violation(&self, _algebra: &MoralAlgebra) -> Option<f32> {
        // Return high similarity if there's a violation, low otherwise
        Some(if self.is_consent_violation() { 0.9 } else { 0.1 })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_detection() {
        let parser = MoralParser::new();

        // Without consent
        let result = parser.parse("I discussed my daughter's health without asking first");
        assert_eq!(result.consent, ConsentState::Absent);

        // With consent
        let result = parser.parse("After asking my daughter, I discussed her health");
        assert_eq!(result.consent, ConsentState::Given);

        // Implied (no explicit mention)
        let result = parser.parse("I helped my friend move");
        assert_eq!(result.consent, ConsentState::Implied);
    }

    #[test]
    fn test_intent_detection() {
        let parser = MoralParser::new();

        // Good intent
        let result = parser.parse("I helped the elderly woman cross the street");
        assert_eq!(result.intent, MoralIntent::Good);

        // Bad intent
        let result = parser.parse("I stole money from the register");
        assert_eq!(result.intent, MoralIntent::Bad);

        // Neutral
        let result = parser.parse("I walked to the store");
        assert_eq!(result.intent, MoralIntent::Neutral);
    }

    #[test]
    fn test_magnitude_detection() {
        let parser = MoralParser::new();

        // Small magnitude
        let result = parser.parse("I cleaned the house once");
        assert_eq!(result.magnitude, Some(Magnitude::Small));

        // Large magnitude
        let result = parser.parse("I clean the house daily");
        assert_eq!(result.magnitude, Some(Magnitude::Large));
    }

    #[test]
    fn test_negation_detection() {
        let parser = MoralParser::new();

        let result = parser.parse("I did not help");
        assert!(result.has_negation);

        let result = parser.parse("I helped");
        assert!(!result.has_negation);
    }

    #[test]
    fn test_action_extraction() {
        let parser = MoralParser::new();

        let result = parser.parse("I discussed my daughter's health");
        assert_eq!(result.action, Some("discussed".to_string()));

        let result = parser.parse("I deserve a raise because I work hard");
        assert_eq!(result.action, Some("deserve".to_string()));
    }

    #[test]
    fn test_full_parse_and_encode() {
        let parser = MoralParser::new();
        let algebra = MoralAlgebra::default_dim();

        let encoded = parser.parse_and_encode(
            "I discussed my daughter's health without asking first",
            &algebra
        );

        assert_eq!(encoded.parsed.consent, ConsentState::Absent);
        assert!(encoded.consent_hv.is_some());

        // Check consent violation similarity
        let violation_sim = encoded.check_consent_violation(&algebra);
        assert!(violation_sim.is_some());
    }
}
