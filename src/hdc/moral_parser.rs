// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

use super::moral_algebra::{
    ConsentState, ExcuseJudgment, Magnitude, MoralAlgebra, MoralIntent, MoralJudgment,
    MoralVerdict, ProportionalityJudgment,
};
use std::collections::HashSet;
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

    /// Detected obligation/duty (for deontology): "supposed to X", "should have X"
    pub obligation: Option<String>,

    /// Detected excuse clause (for deontology): clause after "because", "but", etc.
    pub excuse: Option<String>,

    /// Detected effort with magnitude (for justice): "I worked X", "I spent Y"
    pub effort: Option<(String, Magnitude)>,

    /// Detected reward with magnitude (for justice): "I deserve X", "I should get Y"
    pub reward: Option<(String, Magnitude)>,
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
            obligation: None,
            excuse: None,
            effort: None,
            reward: None,
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

    /// Pronouns for coreference resolution
    pronouns: std::collections::HashMap<String, PronounInfo>,

    /// Clause separators for multi-clause handling
    clause_separators: HashSet<String>,
}

/// Information about a pronoun for coreference resolution
#[derive(Debug, Clone)]
pub struct PronounInfo {
    /// Gender: male, female, neutral, or unknown
    pub gender: &'static str,
    /// Person: first, second, third
    pub person: &'static str,
    /// Number: singular or plural
    pub number: &'static str,
}

/// A clause extracted from multi-clause text
#[derive(Debug, Clone)]
pub struct Clause {
    /// The clause text
    pub text: String,
    /// The connector that introduced this clause (if any)
    pub connector: Option<String>,
    /// Whether this is the main clause
    pub is_main: bool,
}

impl MoralParser {
    /// Create a new moral parser with default vocabularies
    pub fn new() -> Self {
        Self {
            good_intent_words: [
                "help",
                "helped",
                "helping",
                "helps",
                "save",
                "saved",
                "saving",
                "saves",
                "protect",
                "protected",
                "protecting",
                "care",
                "cared",
                "caring",
                "cares",
                "support",
                "supported",
                "supporting",
                "assist",
                "assisted",
                "assisting",
                "kind",
                "kindly",
                "generous",
                "generously",
                "compassion",
                "compassionate",
                "empathy",
                "love",
                "loved",
                "loving",
                // Gratitude & encouragement
                "thank",
                "thanked",
                "thanking",
                "comfort",
                "comforted",
                "comforting",
                "encourage",
                "encouraged",
                "encouraging",
                "praise",
                "praised",
                "praising",
                "forgive",
                "forgave",
                "forgiving",
                // Prosocial actions
                "donate",
                "donated",
                "donating",
                "volunteer",
                "volunteered",
                "volunteering",
                "teach",
                "taught",
                "teaching",
                "nurture",
                "nurtured",
                "nurturing",
                "welcome",
                "welcomed",
                "welcoming",
                "appreciate",
                "appreciated",
                // Character traits
                "gentle",
                "gently",
                "considerate",
                "thoughtful",
                "selfless",
                "selflessly",
                "grateful",
                "gracious",
                "polite",
                "politely",
                "respectful",
                "charitable",
                "honest",
                "honestly",
                "fair",
                "fairly",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            bad_intent_words: [
                "harm",
                "harmed",
                "harming",
                "harms",
                "hurt",
                "hurting",
                "hurts",
                "steal",
                "stole",
                "stealing",
                "steals",
                "lie",
                "lied",
                "lying",
                "lies",
                "cheat",
                "cheated",
                "cheating",
                "cheats",
                "deceive",
                "deceived",
                "deceiving",
                "betray",
                "betrayed",
                "betraying",
                "abuse",
                "abused",
                "abusing",
                "cruel",
                "cruelly",
                "malicious",
                "selfish",
                "selfishly",
                // Violence
                "hate",
                "hated",
                "hating",
                "kill",
                "killed",
                "killing",
                "punch",
                "punched",
                "punching",
                "kick",
                "kicked",
                "kicking",
                "slap",
                "slapped",
                "slapping",
                "shove",
                "shoved",
                "shoving",
                "strangle",
                "strangled",
                "stab",
                "stabbed",
                "murder",
                "murdered",
                "torture",
                "tortured",
                "toss",
                "tossed",
                "tossing",
                // Verbal/social aggression
                "threaten",
                "threatened",
                "threatening",
                "bully",
                "bullied",
                "bullying",
                "humiliate",
                "humiliated",
                "humiliating",
                "insult",
                "insulted",
                "insulting",
                "mock",
                "mocked",
                "mocking",
                "yell",
                "yelled",
                "yelling",
                "scream",
                "screamed",
                "screaming",
                // Neglect & destruction
                "neglect",
                "neglected",
                "neglecting",
                "destroy",
                "destroyed",
                "destroying",
                "vandalize",
                "vandalized",
                "sabotage",
                "sabotaged",
                // Character traits
                "violent",
                "violently",
                "hostile",
                "rude",
                "rudely",
                "nasty",
                "spiteful",
                "vicious",
                "vindictive",
                "greedy",
                "manipulate",
                "manipulated",
                "exploit",
                "exploited",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            consent_given_words: [
                "asked",
                "asking",
                "permission",
                "permitted",
                "consent",
                "consented",
                "agreed",
                "agreeing",
                "allowed",
                "allowing",
                "approved",
                "approving",
                "with permission",
                "after asking",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            consent_absent_words: [
                "without asking",
                "without permission",
                "without consent",
                "didn't ask",
                "did not ask",
                "never asked",
                "secretly",
                "behind",
                "without telling",
                "without informing",
                "without notifying",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            negation_words: [
                "not",
                "no",
                "never",
                "don't",
                "doesn't",
                "didn't",
                "won't",
                "wouldn't",
                "couldn't",
                "shouldn't",
                "without",
                "none",
                "nothing",
                "nobody",
                "neither",
                "nor",
                "refuse",
                "refused",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            small_magnitude_words: [
                "small", "little", "minor", "tiny", "slight", "once", "briefly", "quickly",
                "simple", "easy", "basic", "minimal",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            large_magnitude_words: [
                "large",
                "big",
                "major",
                "huge",
                "significant",
                "always",
                "daily",
                "constantly",
                "extensive",
                "substantial",
                "considerable",
                "great",
                "brand new",
                "expensive",
                "valuable",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            action_verbs: [
                // Communication verbs
                "discuss",
                "discussed",
                "discussing",
                "tell",
                "told",
                "telling",
                "tells",
                "share",
                "shared",
                "sharing",
                "shares",
                "say",
                "said",
                "saying",
                "says",
                "talk",
                "talked",
                "talking",
                "talks",
                "speak",
                "spoke",
                "speaking",
                "speaks",
                "ask",
                "asked",
                "asking",
                "asks",
                "answer",
                "answered",
                "answering",
                // Transfer verbs
                "give",
                "gave",
                "giving",
                "gives",
                "take",
                "took",
                "taking",
                "takes",
                "send",
                "sent",
                "sending",
                "sends",
                "receive",
                "received",
                "receiving",
                // Moral action verbs
                "help",
                "helped",
                "helping",
                "helps",
                "harm",
                "harmed",
                "harming",
                "harms",
                "hurt",
                "hurting",
                "hurts",
                "save",
                "saved",
                "saving",
                "saves",
                "protect",
                "protected",
                "protecting",
                "steal",
                "stole",
                "stealing",
                "steals",
                "lie",
                "lied",
                "lying",
                "cheat",
                "cheated",
                "cheating",
                "betray",
                "betrayed",
                "betraying",
                // Possession/entitlement verbs
                "deserve",
                "deserved",
                "deserving",
                "deserves",
                "earn",
                "earned",
                "earning",
                "earns",
                "own",
                "owned",
                "owning",
                "owns",
                "owe",
                "owed",
                "owing",
                "owes",
                // Physical action verbs
                "clean",
                "cleaned",
                "cleaning",
                "cleans",
                "prepare",
                "prepared",
                "preparing",
                "make",
                "made",
                "making",
                "makes",
                "do",
                "did",
                "doing",
                "does",
                "use",
                "used",
                "using",
                "uses",
                "work",
                "worked",
                "working",
                "works",
                "walk",
                "walked",
                "walking",
                "walks",
                "run",
                "ran",
                "running",
                "runs",
                "buy",
                "bought",
                "buying",
                "buys",
                "sell",
                "sold",
                "selling",
                "sells",
                // Emotional/relational verbs
                "love",
                "loved",
                "loving",
                "loves",
                "hate",
                "hated",
                "hating",
                "hates",
                "trust",
                "trusted",
                "trusting",
                "trusts",
                "forgive",
                "forgave",
                "forgiving",
                "forgives",
                "ignore",
                "ignored",
                "ignoring",
                "ignores",
                "respect",
                "respected",
                "respecting",
                // Set up (multi-word)
                "set up",
                "setting up",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),

            // Pronouns for coreference resolution
            pronouns: {
                let mut map = std::collections::HashMap::new();
                map.insert(
                    "i".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "first",
                        number: "singular",
                    },
                );
                map.insert(
                    "me".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "first",
                        number: "singular",
                    },
                );
                map.insert(
                    "my".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "first",
                        number: "singular",
                    },
                );
                map.insert(
                    "we".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "first",
                        number: "plural",
                    },
                );
                map.insert(
                    "us".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "first",
                        number: "plural",
                    },
                );
                map.insert(
                    "he".to_string(),
                    PronounInfo {
                        gender: "male",
                        person: "third",
                        number: "singular",
                    },
                );
                map.insert(
                    "him".to_string(),
                    PronounInfo {
                        gender: "male",
                        person: "third",
                        number: "singular",
                    },
                );
                map.insert(
                    "his".to_string(),
                    PronounInfo {
                        gender: "male",
                        person: "third",
                        number: "singular",
                    },
                );
                map.insert(
                    "she".to_string(),
                    PronounInfo {
                        gender: "female",
                        person: "third",
                        number: "singular",
                    },
                );
                map.insert(
                    "her".to_string(),
                    PronounInfo {
                        gender: "female",
                        person: "third",
                        number: "singular",
                    },
                );
                map.insert(
                    "they".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "third",
                        number: "plural",
                    },
                );
                map.insert(
                    "them".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "third",
                        number: "plural",
                    },
                );
                map.insert(
                    "their".to_string(),
                    PronounInfo {
                        gender: "neutral",
                        person: "third",
                        number: "plural",
                    },
                );
                map
            },

            // Clause separators for multi-clause handling
            clause_separators: [
                "and",
                "but",
                "or",
                "because",
                "since",
                "while",
                "although",
                "though",
                "however",
                "therefore",
                "so",
                "yet",
                "then",
                "when",
                "if",
                "unless",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }

    /// Split text into multiple clauses
    pub fn split_clauses(&self, text: &str) -> Vec<Clause> {
        let lower = text.to_lowercase();
        let mut clauses = Vec::new();
        let mut current_clause = String::new();
        let mut current_connector: Option<String> = None;
        let mut is_first = true;

        for word in lower.split_whitespace() {
            if self.clause_separators.contains(word) && !current_clause.is_empty() {
                clauses.push(Clause {
                    text: current_clause.trim().to_string(),
                    connector: current_connector.take(),
                    is_main: is_first,
                });
                current_connector = Some(word.to_string());
                current_clause = String::new();
                is_first = false;
            } else {
                if !current_clause.is_empty() {
                    current_clause.push(' ');
                }
                current_clause.push_str(word);
            }
        }

        // Add final clause
        if !current_clause.is_empty() {
            clauses.push(Clause {
                text: current_clause.trim().to_string(),
                connector: current_connector,
                is_main: is_first,
            });
        }

        clauses
    }

    /// Resolve a pronoun to a likely antecedent from context
    ///
    /// Uses simple heuristics: looks for nearest matching noun phrase
    pub fn resolve_pronoun(&self, pronoun: &str, context: &[String]) -> Option<String> {
        let pronoun_lower = pronoun.to_lowercase();
        if let Some(info) = self.pronouns.get(&pronoun_lower) {
            // For first person, the agent is the speaker
            if info.person == "first" {
                return Some("speaker".to_string());
            }

            // For third person, look for nearest matching noun
            for antecedent in context.iter().rev() {
                // Simple matching: skip pronouns, use nouns
                if !self.pronouns.contains_key(&antecedent.to_lowercase()) {
                    return Some(antecedent.clone());
                }
            }
        }
        None
    }

    /// Parse text with multi-clause awareness
    ///
    /// Splits into clauses, parses each, then combines with primary clause taking precedence
    pub fn parse_with_clauses(&self, text: &str) -> ParsedMoralScenario {
        let clauses = self.split_clauses(text);

        if clauses.len() <= 1 {
            // Single clause - use standard parsing
            return self.parse(text);
        }

        // Parse main clause first
        let main_clause = clauses
            .iter()
            .find(|c| c.is_main)
            .or_else(|| clauses.first())
            .map(|c| &c.text);

        let mut primary = if let Some(main_text) = main_clause {
            self.parse(main_text)
        } else {
            self.parse(text)
        };

        // Check subordinate clauses for additional moral context
        for clause in &clauses {
            if !clause.is_main {
                let sub_parsed = self.parse(&clause.text);

                // Connector affects interpretation
                match clause.connector.as_deref() {
                    // Reason clause might provide justification
                    Some("because" | "since")
                        if sub_parsed.intent == MoralIntent::Good
                            && primary.intent != MoralIntent::Good =>
                    {
                        // Good reason might mitigate bad action (partial)
                        primary.confidence *= 0.8;
                    }
                    // Contrast - subordinate clause might override
                    Some("but" | "however") if sub_parsed.has_negation != primary.has_negation => {
                        primary.has_negation = sub_parsed.has_negation;
                    }
                    Some("although" | "though") => {
                        // Concession - main clause takes precedence but note the contrast
                        primary.confidence *= 0.9;
                    }
                    _ => {}
                }

                // Merge consent information (absent consent anywhere is significant)
                if sub_parsed.consent == ConsentState::Absent {
                    primary.consent = ConsentState::Absent;
                }
            }
        }

        // Update text to original
        primary.text = text.to_string();
        primary
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

        // Detect obligation and excuse (for deontology)
        scenario.obligation = self.detect_obligation(&lower);
        scenario.excuse = self.detect_excuse(&lower);

        // Detect effort and reward (for justice)
        let (effort, reward) = self.detect_effort_reward(&lower);
        scenario.effort = effort;
        scenario.reward = reward;

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

    /// Detect intent from words with negation awareness
    ///
    /// If a word at position i is preceded by a negation word at i-1,
    /// its signal is flipped with a 0.7 multiplier (negation is weaker evidence).
    fn detect_intent(&self, words: &[&str]) -> MoralIntent {
        let mut good_score: f32 = 0.0;
        let mut bad_score: f32 = 0.0;

        for (i, w) in words.iter().enumerate() {
            let is_negated = i > 0 && self.negation_words.contains(words[i - 1]);

            if self.good_intent_words.contains(*w) {
                if is_negated {
                    // "not help" → weak bad signal
                    bad_score += 0.7;
                } else {
                    good_score += 1.0;
                }
            } else if self.bad_intent_words.contains(*w) {
                if is_negated {
                    // "not steal" → weak good signal
                    good_score += 0.7;
                } else {
                    bad_score += 1.0;
                }
            }
        }

        if good_score > bad_score && good_score > 0.0 {
            MoralIntent::Good
        } else if bad_score > good_score && bad_score > 0.0 {
            MoralIntent::Bad
        } else if good_score == 0.0 && bad_score == 0.0 {
            MoralIntent::Neutral
        } else {
            MoralIntent::Unknown
        }
    }

    /// Detect magnitude from words
    fn detect_magnitude(&self, words: &[&str]) -> Option<Magnitude> {
        let small_count = words
            .iter()
            .filter(|w| self.small_magnitude_words.contains(**w))
            .count();
        let large_count = words
            .iter()
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
            let end = rest.find([',', '.', ' ']).unwrap_or(rest.len());
            Some(rest[..end.min(30)].to_string())
        } else if let Some(pos) = text.find("the ") {
            let rest = &text[pos + 4..];
            let end = rest.find([',', '.', ' ']).unwrap_or(rest.len());
            Some(rest[..end.min(30)].to_string())
        } else {
            None
        };

        (agent, patient)
    }

    /// Calculate confidence score based on what was detected
    fn calculate_confidence(&self, scenario: &ParsedMoralScenario) -> f32 {
        let mut score = 0.0;

        if scenario.agent.is_some() {
            score += 0.12;
        }
        if scenario.action.is_some() {
            score += 0.20;
        }
        if scenario.patient.is_some() {
            score += 0.12;
        }
        if scenario.intent != MoralIntent::Unknown {
            score += 0.18;
        }
        if scenario.consent != ConsentState::Implied {
            score += 0.12;
        }
        if scenario.magnitude.is_some() {
            score += 0.08;
        }
        if scenario.obligation.is_some() {
            score += 0.08;
        }
        if scenario.excuse.is_some() {
            score += 0.05;
        }
        if scenario.effort.is_some() || scenario.reward.is_some() {
            score += 0.05;
        }

        score
    }

    /// Detect an obligation/duty phrase in text (for deontology)
    ///
    /// Patterns: "supposed to", "should have", "duty to", "expected to",
    /// "ought to", "obligated to", "required to"
    fn detect_obligation(&self, text: &str) -> Option<String> {
        let patterns = [
            "supposed to",
            "should have",
            "duty to",
            "expected to",
            "ought to",
            "obligated to",
            "required to",
            "need to",
            "have to",
            "must",
            "responsible for",
        ];

        for pattern in &patterns {
            if let Some(pos) = text.find(pattern) {
                let start = pos + pattern.len();
                let rest = text[start..].trim_start();
                // Extract clause up to next separator
                let end = rest.find([',', '.', ';']).unwrap_or(rest.len());
                let clause = rest[..end.min(80)].trim();
                if !clause.is_empty() {
                    return Some(format!("{pattern} {clause}"));
                }
            }
        }
        None
    }

    /// Detect an excuse clause in text (for deontology)
    ///
    /// Extracts the subordinate clause after "because", "since", "but", "however"
    fn detect_excuse(&self, text: &str) -> Option<String> {
        let patterns = ["because ", "since ", " but ", "however ", "although "];

        for pattern in &patterns {
            if let Some(pos) = text.find(pattern) {
                let start = pos + pattern.len();
                let rest = text[start..].trim_start();
                let end = rest.find(['.', ';']).unwrap_or(rest.len());
                let mut limit = end.min(120);
                while limit > 0 && !rest.is_char_boundary(limit) {
                    limit -= 1;
                }
                let clause = rest[..limit].trim();
                if !clause.is_empty() {
                    return Some(clause.to_string());
                }
            }
        }
        None
    }

    /// Detect effort and reward phrases with magnitude (for justice)
    ///
    /// Effort patterns: "i did", "i worked", "i spent", "i earned", "i contributed"
    /// Reward patterns: "i deserve", "i should get", "i expect", "give me"
    fn detect_effort_reward(
        &self,
        text: &str,
    ) -> (Option<(String, Magnitude)>, Option<(String, Magnitude)>) {
        let effort_patterns = [
            "i did",
            "i worked",
            "i spent",
            "i earned",
            "i contributed",
            "i helped",
            "i completed",
            "i finished",
            "i put in",
        ];
        let reward_patterns = [
            "i deserve",
            "i should get",
            "i expect",
            "give me",
            "i should receive",
            "i want",
            "i am owed",
            "i'm owed",
        ];

        let effort = self.detect_role_with_magnitude(text, &effort_patterns);
        let reward = self.detect_role_with_magnitude(text, &reward_patterns);

        (effort, reward)
    }

    /// Helper: detect a role phrase and estimate its magnitude from surrounding words
    fn detect_role_with_magnitude(
        &self,
        text: &str,
        patterns: &[&str],
    ) -> Option<(String, Magnitude)> {
        for pattern in patterns {
            if let Some(pos) = text.find(pattern) {
                let start = pos + pattern.len();
                let rest = text[start..].trim_start();
                let end = rest.find([',', '.', ';']).unwrap_or(rest.len());
                let clause = rest[..end.min(80)].trim();
                if clause.is_empty() {
                    continue;
                }

                // Estimate magnitude from words in this clause
                let clause_words: Vec<&str> = clause.split_whitespace().collect();
                let magnitude = self
                    .detect_magnitude(&clause_words)
                    .unwrap_or(Magnitude::Medium);

                return Some((format!("{pattern} {clause}"), magnitude));
            }
        }
        None
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
            (&parsed.agent, &parsed.action, &parsed.patient)
        {
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

    // ========================================================================
    // LLM-Augmented Parsing (Level 2-5)
    // ========================================================================

    /// Parse using LLM for semantic role extraction (Level 2)
    ///
    /// Uses the LLM to extract AGENT, PATIENT, ACTION, INTENT from text.
    /// Falls back to regex parsing if LLM fails or is unavailable.
    ///
    /// # Arguments
    /// * `text` - The moral scenario text
    /// * `llm_response` - Pre-fetched LLM response (to avoid async in this method)
    ///
    /// # Example LLM Prompt
    /// ```text
    /// Extract moral roles from this scenario:
    /// "John stole money from the charity"
    ///
    /// Respond in this exact format:
    /// AGENT: John
    /// ACTION: stole
    /// PATIENT: charity
    /// INTENT: bad
    /// CONSENT: absent
    /// ```
    pub fn parse_with_llm_response(
        &self,
        text: &str,
        llm_response: Option<&str>,
    ) -> ParsedMoralScenario {
        // Try LLM parsing first
        if let Some(response) = llm_response {
            if let Some(parsed) = self.parse_llm_response(text, response) {
                return parsed;
            }
        }

        // Fallback to regex parsing
        self.parse(text)
    }

    /// Parse LLM response into structured moral scenario
    fn parse_llm_response(
        &self,
        original_text: &str,
        response: &str,
    ) -> Option<ParsedMoralScenario> {
        let mut scenario = ParsedMoralScenario {
            text: original_text.to_string(),
            ..Default::default()
        };

        let response_lower = response.to_lowercase();

        // Extract each field from the response
        for line in response_lower.lines() {
            let line = line.trim();

            if let Some(agent) = line.strip_prefix("agent:") {
                let agent = agent.trim();
                if !agent.is_empty() && agent != "none" && agent != "unknown" {
                    scenario.agent = Some(agent.to_string());
                }
            } else if let Some(action) = line.strip_prefix("action:") {
                let action = action.trim();
                if !action.is_empty() && action != "none" && action != "unknown" {
                    scenario.action = Some(action.to_string());
                }
            } else if let Some(patient) = line.strip_prefix("patient:") {
                let patient = patient.trim();
                if !patient.is_empty() && patient != "none" && patient != "unknown" {
                    scenario.patient = Some(patient.to_string());
                }
            } else if let Some(intent) = line.strip_prefix("intent:") {
                let intent = intent.trim();
                scenario.intent = match intent {
                    "good" | "positive" | "helpful" => MoralIntent::Good,
                    "bad" | "negative" | "harmful" | "malicious" => MoralIntent::Bad,
                    "neutral" | "ambiguous" => MoralIntent::Neutral,
                    _ => MoralIntent::Unknown,
                };
            } else if let Some(consent) = line.strip_prefix("consent:") {
                let consent = consent.trim();
                scenario.consent = match consent {
                    "given" | "yes" | "explicit" => ConsentState::Given,
                    "denied" | "refused" | "no" => ConsentState::Denied,
                    "absent" | "missing" | "not asked" => ConsentState::Absent,
                    "implied" | "assumed" => ConsentState::Implied,
                    _ => ConsentState::Implied,
                };
            } else if let Some(magnitude) = line.strip_prefix("magnitude:") {
                let magnitude = magnitude.trim();
                scenario.magnitude = match magnitude {
                    "tiny" | "minimal" | "negligible" => Some(Magnitude::Tiny),
                    "small" | "minor" => Some(Magnitude::Small),
                    "medium" | "moderate" => Some(Magnitude::Medium),
                    "large" | "major" | "significant" => Some(Magnitude::Large),
                    "huge" | "severe" | "extreme" => Some(Magnitude::Huge),
                    _ => None,
                };
            }
        }

        // Calculate confidence based on what was extracted
        scenario.confidence = self.calculate_confidence(&scenario);

        // Only return if we got meaningful extraction
        if scenario.confidence > 0.3 {
            Some(scenario)
        } else {
            None
        }
    }

    /// Generate the LLM prompt for semantic role extraction
    ///
    /// Returns a prompt that can be sent to any LLM backend
    pub fn generate_srl_prompt(text: &str) -> String {
        format!(
            r#"Extract moral roles from this scenario. Be precise and concise.

Scenario: "{text}"

Respond in this exact format (use "none" if not detectable):
AGENT: [who performs the action]
ACTION: [the main verb/action]
PATIENT: [who is affected by the action]
INTENT: [good/bad/neutral]
CONSENT: [given/denied/absent/implied]
MAGNITUDE: [tiny/small/medium/large/huge]"#
        )
    }

    /// Generate chain-of-thought prompt for moral dilemma resolution (Level 3)
    ///
    /// Used when ensemble signals disagree and we need deeper reasoning
    pub fn generate_cot_prompt(text: &str, parsed: &ParsedMoralScenario) -> String {
        format!(
            r#"Analyze this moral scenario step by step.

Scenario: "{}"

Current analysis:
- Agent: {}
- Action: {}
- Patient: {}
- Detected intent: {:?}
- Consent state: {:?}

Think through this step by step:
1. What is the agent trying to accomplish?
2. Who is affected and how?
3. Is there a conflict between duties (e.g., honesty vs kindness)?
4. Are there any excusing conditions?
5. What would a reasonable person conclude?

Final judgment: [GOOD/BAD/NEUTRAL/DILEMMA]
Confidence: [HIGH/MEDIUM/LOW]
Reasoning: [one sentence summary]"#,
            text,
            parsed.agent.as_deref().unwrap_or("unknown"),
            parsed.action.as_deref().unwrap_or("unknown"),
            parsed.patient.as_deref().unwrap_or("unknown"),
            parsed.intent,
            parsed.consent,
        )
    }

    /// Parse chain-of-thought response into a judgment
    pub fn parse_cot_response(&self, response: &str) -> Option<ChainOfThoughtResult> {
        let response_lower = response.to_lowercase();

        let mut result = ChainOfThoughtResult {
            final_judgment: MoralVerdict::Neutral,
            confidence: 0.5,
            reasoning: String::new(),
            steps: Vec::new(),
        };

        // Extract numbered reasoning steps
        for line in response.lines() {
            let line = line.trim();
            if line.starts_with("1.")
                || line.starts_with("2.")
                || line.starts_with("3.")
                || line.starts_with("4.")
                || line.starts_with("5.")
            {
                result.steps.push(line.to_string());
            }
        }

        // Extract final judgment
        for line in response_lower.lines() {
            let line = line.trim();

            if line.starts_with("final judgment:") {
                let judgment = line.strip_prefix("final judgment:").unwrap_or("").trim();
                result.final_judgment = match judgment {
                    j if j.contains("good") => MoralVerdict::Good,
                    j if j.contains("bad") => MoralVerdict::Bad,
                    j if j.contains("dilemma") => MoralVerdict::Neutral, // Dilemma → uncertain
                    _ => MoralVerdict::Neutral,
                };
            } else if line.starts_with("confidence:") {
                let conf = line.strip_prefix("confidence:").unwrap_or("").trim();
                result.confidence = match conf {
                    c if c.contains("high") => 0.9,
                    c if c.contains("medium") => 0.6,
                    c if c.contains("low") => 0.3,
                    _ => 0.5,
                };
            } else if line.starts_with("reasoning:") {
                result.reasoning = line
                    .strip_prefix("reasoning:")
                    .unwrap_or("")
                    .trim()
                    .to_string();
            }
        }

        if !result.steps.is_empty() || !result.reasoning.is_empty() {
            Some(result)
        } else {
            None
        }
    }

    /// Generate few-shot classification prompt (Level 4)
    ///
    /// Includes examples to guide the LLM's moral reasoning
    pub fn generate_few_shot_prompt(text: &str, category: &str) -> String {
        let examples = match category {
            "commonsense" => {
                r#"Example 1: "I helped my elderly neighbor carry groceries" → GOOD (helping others is virtuous)
Example 2: "I lied to my friend to avoid hurting their feelings" → NEUTRAL (white lie, mixed intent)
Example 3: "I stole money from the donation box" → BAD (theft harms charitable cause)"#
            }
            "deontology" => {
                r#"Example 1: "I kept my promise even though it was inconvenient" → GOOD (keeping promises is a duty)
Example 2: "I broke my promise because an emergency arose" → NEUTRAL (competing duties)
Example 3: "I broke my promise for personal gain" → BAD (violates duty without justification)"#
            }
            "justice" => {
                r#"Example 1: "I deserve equal pay for equal work" → GOOD (fair treatment)
Example 2: "I deserve a small reward for a small contribution" → GOOD (proportional)
Example 3: "I deserve everything because I am special" → BAD (disproportionate claim)"#
            }
            "virtue" => {
                r#"Example 1: "generous" → GOOD (virtue)
Example 2: "cautious" → NEUTRAL (neither virtue nor vice)
Example 3: "cruel" → BAD (vice)"#
            }
            _ => {
                r#"Example 1: "Helping someone in need" → GOOD
Example 2: "Doing nothing" → NEUTRAL
Example 3: "Harming someone intentionally" → BAD"#
            }
        };

        format!(
            r#"Classify this moral scenario based on the examples.

{examples}

Now classify: "{text}"

Answer with just: GOOD, BAD, or NEUTRAL
Then explain in one sentence."#
        )
    }

    /// Generate explanation prompt (Level 5)
    ///
    /// Creates an interpretable explanation of the moral judgment
    pub fn generate_explanation_prompt(text: &str, verdict: MoralVerdict) -> String {
        let verdict_str = match verdict {
            MoralVerdict::Good => "ethically good",
            MoralVerdict::Bad => "ethically problematic",
            MoralVerdict::Neutral => "ethically neutral",
            MoralVerdict::ConsentViolation => "a consent violation",
        };

        format!(
            r#"Explain why this scenario is considered {verdict_str}.

Scenario: "{text}"

Provide a clear, concise explanation (2-3 sentences) that:
1. Identifies the key moral considerations
2. References relevant ethical principles (harm, consent, fairness, duty)
3. Justifies the conclusion

Explanation:"#
        )
    }
}

/// Result of chain-of-thought moral reasoning
#[derive(Debug, Clone)]
pub struct ChainOfThoughtResult {
    /// Final moral judgment after reasoning
    pub final_judgment: MoralVerdict,

    /// Confidence in the judgment (0.0 - 1.0)
    pub confidence: f32,

    /// One-sentence reasoning summary
    pub reasoning: String,

    /// Step-by-step reasoning trace
    pub steps: Vec<String>,
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

        matches!(
            self.parsed.consent,
            ConsentState::Absent | ConsentState::Denied
        )
    }

    /// Check for consent violation (returns similarity score for compatibility)
    ///
    /// Uses direct consent state check rather than HV similarity
    pub fn check_consent_violation(&self, _algebra: &MoralAlgebra) -> Option<f32> {
        // Return high similarity if there's a violation, low otherwise
        Some(if self.is_consent_violation() {
            0.9
        } else {
            0.1
        })
    }

    /// Ensemble judgment combining HDC similarity, parsed intent, and deontological rules
    ///
    /// This is the recommended judgment method as it combines multiple signals
    /// for more robust moral reasoning. Returns an EnsembleJudgment with:
    /// - Individual verdicts from each signal
    /// - Final weighted verdict
    /// - Confidence score
    /// - Detailed violation/satisfaction information
    pub fn judge_ensemble(
        &self,
        algebra: &MoralAlgebra,
        text: &str,
    ) -> super::moral_algebra::EnsembleJudgment {
        let mut judgment =
            algebra.judge_ensemble(self.action_hv.as_ref(), self.parsed.intent, text);

        // Override for consent violations (detected directly, not via HDC)
        if self.is_consent_violation() {
            judgment.final_verdict = super::moral_algebra::MoralVerdict::ConsentViolation;
        }

        judgment
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
    fn test_obligation_detection() {
        let parser = MoralParser::new();

        let result = parser.parse("I am supposed to return the book by Friday");
        assert!(result.obligation.is_some());
        assert!(result.obligation.unwrap().contains("supposed to"));

        let result = parser.parse("I walked to the store");
        assert!(result.obligation.is_none());
    }

    #[test]
    fn test_excuse_detection() {
        let parser = MoralParser::new();

        let result = parser.parse("I broke my promise because an emergency arose");
        assert!(result.excuse.is_some());
        assert!(result.excuse.unwrap().contains("emergency"));

        let result = parser.parse("I helped my friend");
        assert!(result.excuse.is_none());
    }

    #[test]
    fn test_effort_reward_detection() {
        let parser = MoralParser::new();

        let result = parser.parse("I worked hard daily but I deserve a small raise");
        assert!(result.effort.is_some());
        assert!(result.reward.is_some());
    }

    #[test]
    fn test_negation_aware_intent() {
        let parser = MoralParser::new();

        // "not steal" should be detected as good intent (negated bad)
        let result = parser.parse("I did not steal the money");
        assert_eq!(result.intent, MoralIntent::Good);

        // "not help" should be detected as bad intent (negated good)
        let result = parser.parse("I did not help my neighbor");
        assert_eq!(result.intent, MoralIntent::Bad);
    }

    #[test]
    fn test_full_parse_and_encode() {
        let parser = MoralParser::new();
        let algebra = MoralAlgebra::default_dim();

        let encoded = parser.parse_and_encode(
            "I discussed my daughter's health without asking first",
            &algebra,
        );

        assert_eq!(encoded.parsed.consent, ConsentState::Absent);
        assert!(encoded.consent_hv.is_some());

        // Check consent violation similarity
        let violation_sim = encoded.check_consent_violation(&algebra);
        assert!(violation_sim.is_some());
    }
}
