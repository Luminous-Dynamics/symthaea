//! Deep Parser - Semantic Understanding Beyond Surface Features
//!
//! This module provides deeper semantic analysis than the basic parser:
//! - Semantic role labeling (Agent, Patient, Instrument, etc.)
//! - Dependency parsing (subject-verb-object extraction)
//! - Intent classification (question, command, statement, etc.)
//! - Pragmatic inference (what the user really means)
//!
//! Unlike LLMs which learn implicit patterns, we use explicit linguistic rules
//! for transparent, explainable language understanding.

use std::collections::HashMap;
use super::vocabulary::Vocabulary;
use super::parser::ParsedSentence;

// ============================================================================
// Semantic Roles (Thematic Relations)
// ============================================================================

/// Semantic roles following Fillmore's Case Grammar
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SemanticRole {
    /// The doer of the action (typically subject)
    Agent,
    /// The entity affected by the action (typically object)
    Patient,
    /// The entity that experiences something
    Experiencer,
    /// The thing used to perform the action
    Instrument,
    /// The origin point of movement/transfer
    Source,
    /// The destination of movement/transfer
    Goal,
    /// The location where something happens
    Location,
    /// The time when something happens
    Time,
    /// The reason or cause
    Cause,
    /// The beneficiary of an action
    Beneficiary,
    /// Additional circumstances
    Manner,
    /// The topic being discussed
    Theme,
    /// Something that exists or appears
    Stimulus,
}

/// A phrase with its semantic role
#[derive(Debug, Clone)]
pub struct RolePhrase {
    pub role: SemanticRole,
    pub text: String,
    pub head_word: String,
    pub confidence: f32,
}

// ============================================================================
// Dependency Structure
// ============================================================================

/// Types of syntactic dependencies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyType {
    // Core relations
    Subject,    // nsubj - nominal subject
    Object,     // dobj - direct object
    IndirectObject,  // iobj - indirect object

    // Modifiers
    Adjective,  // amod - adjectival modifier
    Adverb,     // advmod - adverbial modifier
    Determiner, // det - determiner
    Possessive, // poss - possessive modifier
    Numeral,    // nummod - numeric modifier

    // Clausal
    Complement, // ccomp - clausal complement
    Relative,   // rcmod - relative clause modifier
    Adverbial,  // advcl - adverbial clause

    // Prepositional
    PrepMod,    // prep - prepositional modifier
    PrepObject, // pobj - object of preposition

    // Coordination
    Conjunction,  // cc - coordinating conjunction
    Conjunct,     // conj - conjunct

    // Other
    Auxiliary,  // aux - auxiliary
    Negation,   // neg - negation
    Punctuation,
    Root,
}

/// A dependency edge
#[derive(Debug, Clone)]
pub struct Dependency {
    pub head_idx: usize,
    pub dependent_idx: usize,
    pub relation: DependencyType,
}

/// Dependency tree for a sentence
#[derive(Debug, Clone)]
pub struct DependencyTree {
    pub tokens: Vec<String>,
    pub pos_tags: Vec<PosTag>,
    pub dependencies: Vec<Dependency>,
    pub root_idx: usize,
}

/// Part-of-speech tags
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosTag {
    // Nouns
    Noun,
    ProperNoun,
    Pronoun,

    // Verbs
    Verb,
    Auxiliary,
    Modal,

    // Modifiers
    Adjective,
    Adverb,
    Determiner,
    Preposition,
    Conjunction,

    // Other
    Numeral,
    Particle,
    Interjection,
    Punctuation,
    Unknown,
}

// ============================================================================
// Intent Classification
// ============================================================================

/// Primary communicative intent
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intent {
    // Questions
    YesNoQuestion,      // "Are you conscious?"
    WhQuestion,         // "What is love?"
    HowQuestion,        // "How do you feel?"
    WhyQuestion,        // "Why do we exist?"

    // Commands/Requests
    Command,            // "Tell me about..."
    Request,            // "Could you explain..."
    Suggestion,         // "You should try..."

    // Statements
    Statement,          // "I think that..."
    Opinion,            // "I believe..."
    Fact,               // "The sky is blue"

    // Social
    Greeting,           // "Hello", "Hi"
    Farewell,           // "Goodbye", "Bye"
    Thanks,             // "Thank you"
    Apology,            // "Sorry"

    // Expressive
    Exclamation,        // "Wow!", "Amazing!"
    Agreement,          // "Yes", "I agree"
    Disagreement,       // "No", "I disagree"

    // Meta
    Clarification,      // "What do you mean?"
    Confirmation,       // "So you mean..."
    Continuation,       // "And then?"

    Unknown,
}

/// Secondary/nuanced intent features
#[derive(Debug, Clone)]
pub struct IntentFeatures {
    pub primary: Intent,
    pub is_polite: bool,
    pub is_urgent: bool,
    pub is_hypothetical: bool,
    pub is_negated: bool,
    pub certainty: f32,
    pub formality: f32,  // 0.0 = casual, 1.0 = formal
}

// ============================================================================
// Pragmatic Layer
// ============================================================================

/// Pragmatic inference - what the speaker really means
#[derive(Debug, Clone)]
pub struct PragmaticAnalysis {
    /// Literal meaning
    pub literal: String,
    /// Implied meaning (if different)
    pub implied: Option<String>,
    /// Speech act type
    pub speech_act: SpeechAct,
    /// Presuppositions
    pub presuppositions: Vec<String>,
    /// Implicatures (things implied but not stated)
    pub implicatures: Vec<String>,
}

/// Speech act classification (Austin/Searle)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeechAct {
    /// Commits speaker to truth (statements, claims)
    Assertive,
    /// Tries to get hearer to do something
    Directive,
    /// Commits speaker to future action
    Commissive,
    /// Expresses psychological state
    Expressive,
    /// Changes state of affairs (declarations)
    Declarative,
}

// ============================================================================
// Deep Parser
// ============================================================================

/// Deep semantic parser
pub struct DeepParser {
    vocabulary: Vocabulary,
    intent_patterns: HashMap<String, Intent>,
    role_patterns: Vec<RolePattern>,
}

/// Pattern for assigning semantic roles
struct RolePattern {
    verb_class: VerbClass,
    subject_role: SemanticRole,
    object_role: Option<SemanticRole>,
}

/// Verb classes for role assignment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerbClass {
    Action,      // run, eat, build - Agent-Patient
    Experience,  // feel, think, love - Experiencer-Stimulus
    Transfer,    // give, send - Agent-Theme-Goal
    Motion,      // go, come - Agent-Source-Goal
    State,       // be, exist - Theme
    Causative,   // make, cause - Cause-Patient
    Communication, // say, tell - Agent-Theme-Goal
    Perception,  // see, hear - Experiencer-Stimulus
}

/// Result of deep parsing
#[derive(Debug, Clone)]
pub struct DeepParse {
    /// Original text
    pub text: String,
    /// Basic parsed sentence
    pub basic: ParsedSentence,
    /// Dependency structure
    pub dependencies: DependencyTree,
    /// Semantic roles
    pub roles: Vec<RolePhrase>,
    /// Intent classification
    pub intent: IntentFeatures,
    /// Pragmatic analysis
    pub pragmatics: PragmaticAnalysis,
    /// Extracted entities
    pub entities: Vec<Entity>,
}

/// Named entity
#[derive(Debug, Clone)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub start_idx: usize,
    pub end_idx: usize,
}

/// Entity types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityType {
    Person,
    Place,
    Organization,
    Time,
    Number,
    Concept,
    Unknown,
}

impl DeepParser {
    /// Create a new deep parser
    pub fn new(vocabulary: Vocabulary) -> Self {
        let mut parser = Self {
            vocabulary,
            intent_patterns: HashMap::new(),
            role_patterns: Vec::new(),
        };
        parser.initialize_patterns();
        parser
    }

    /// Initialize intent detection patterns
    fn initialize_patterns(&mut self) {
        // Question patterns
        self.intent_patterns.insert("what".to_string(), Intent::WhQuestion);
        self.intent_patterns.insert("who".to_string(), Intent::WhQuestion);
        self.intent_patterns.insert("where".to_string(), Intent::WhQuestion);
        self.intent_patterns.insert("when".to_string(), Intent::WhQuestion);
        self.intent_patterns.insert("which".to_string(), Intent::WhQuestion);
        self.intent_patterns.insert("how".to_string(), Intent::HowQuestion);
        self.intent_patterns.insert("why".to_string(), Intent::WhyQuestion);

        // Greeting patterns
        self.intent_patterns.insert("hello".to_string(), Intent::Greeting);
        self.intent_patterns.insert("hi".to_string(), Intent::Greeting);
        self.intent_patterns.insert("hey".to_string(), Intent::Greeting);
        self.intent_patterns.insert("greetings".to_string(), Intent::Greeting);

        // Farewell patterns
        self.intent_patterns.insert("goodbye".to_string(), Intent::Farewell);
        self.intent_patterns.insert("bye".to_string(), Intent::Farewell);
        self.intent_patterns.insert("farewell".to_string(), Intent::Farewell);

        // Thanks patterns
        self.intent_patterns.insert("thanks".to_string(), Intent::Thanks);
        self.intent_patterns.insert("thank".to_string(), Intent::Thanks);

        // Agreement/Disagreement
        self.intent_patterns.insert("yes".to_string(), Intent::Agreement);
        self.intent_patterns.insert("yeah".to_string(), Intent::Agreement);
        self.intent_patterns.insert("no".to_string(), Intent::Disagreement);
        self.intent_patterns.insert("nope".to_string(), Intent::Disagreement);

        // Role patterns for different verb classes
        self.role_patterns.push(RolePattern {
            verb_class: VerbClass::Action,
            subject_role: SemanticRole::Agent,
            object_role: Some(SemanticRole::Patient),
        });

        self.role_patterns.push(RolePattern {
            verb_class: VerbClass::Experience,
            subject_role: SemanticRole::Experiencer,
            object_role: Some(SemanticRole::Stimulus),
        });

        self.role_patterns.push(RolePattern {
            verb_class: VerbClass::State,
            subject_role: SemanticRole::Theme,
            object_role: None,
        });

        self.role_patterns.push(RolePattern {
            verb_class: VerbClass::Perception,
            subject_role: SemanticRole::Experiencer,
            object_role: Some(SemanticRole::Stimulus),
        });
    }

    /// Perform deep parsing
    pub fn parse(&self, text: &str, basic: ParsedSentence) -> DeepParse {
        let tokens = self.tokenize(text);
        let pos_tags = self.tag_pos(&tokens);
        let dependencies = self.parse_dependencies(&tokens, &pos_tags);
        let roles = self.extract_roles(&tokens, &pos_tags, &dependencies);
        let intent = self.classify_intent(text, &tokens);
        let pragmatics = self.analyze_pragmatics(text, &intent, &roles);
        let entities = self.extract_entities(&tokens, &pos_tags);

        DeepParse {
            text: text.to_string(),
            basic,
            dependencies,
            roles,
            intent,
            pragmatics,
            entities,
        }
    }

    /// Tokenize text
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Part-of-speech tagging
    fn tag_pos(&self, tokens: &[String]) -> Vec<PosTag> {
        tokens.iter().map(|token| {
            // Check vocabulary first
            if let Some(entry) = self.vocabulary.get(token) {
                return match entry.pos.as_str() {
                    "noun" => PosTag::Noun,
                    "verb" => PosTag::Verb,
                    "adj" => PosTag::Adjective,
                    "adv" => PosTag::Adverb,
                    "pron" => PosTag::Pronoun,
                    "prep" => PosTag::Preposition,
                    "conj" => PosTag::Conjunction,
                    "det" => PosTag::Determiner,
                    _ => PosTag::Unknown,
                };
            }

            // Heuristic rules
            self.guess_pos(token)
        }).collect()
    }

    /// Guess POS from word form
    fn guess_pos(&self, token: &str) -> PosTag {
        // Pronouns
        if ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]
            .contains(&token) {
            return PosTag::Pronoun;
        }

        // Determiners
        if ["the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"]
            .contains(&token) {
            return PosTag::Determiner;
        }

        // Prepositions
        if ["in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "into", "through", "during", "before", "after"]
            .contains(&token) {
            return PosTag::Preposition;
        }

        // Conjunctions
        if ["and", "or", "but", "so", "because", "if", "when", "while", "although", "though"]
            .contains(&token) {
            return PosTag::Conjunction;
        }

        // Auxiliaries
        if ["is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might", "must"]
            .contains(&token) {
            return PosTag::Auxiliary;
        }

        // Verb suffixes
        if token.ends_with("ing") || token.ends_with("ed") || token.ends_with("es") {
            return PosTag::Verb;
        }

        // Adjective suffixes
        if token.ends_with("ful") || token.ends_with("less") || token.ends_with("ous") || token.ends_with("ive") || token.ends_with("able") {
            return PosTag::Adjective;
        }

        // Adverb suffix
        if token.ends_with("ly") {
            return PosTag::Adverb;
        }

        // Default to noun
        PosTag::Noun
    }

    /// Parse dependencies (simplified rule-based)
    fn parse_dependencies(&self, tokens: &[String], pos_tags: &[PosTag]) -> DependencyTree {
        let mut dependencies = Vec::new();
        let mut root_idx = 0;

        // Find main verb (root)
        for (i, tag) in pos_tags.iter().enumerate() {
            if *tag == PosTag::Verb {
                root_idx = i;
                break;
            }
            // Fallback: auxiliary as root
            if *tag == PosTag::Auxiliary {
                root_idx = i;
            }
        }

        // Simple dependency extraction
        for (i, tag) in pos_tags.iter().enumerate() {
            if i == root_idx {
                continue;
            }

            let (head_idx, relation) = match tag {
                // Subject is typically before verb
                PosTag::Noun | PosTag::Pronoun if i < root_idx => {
                    (root_idx, DependencyType::Subject)
                }
                // Object is typically after verb
                PosTag::Noun if i > root_idx => {
                    (root_idx, DependencyType::Object)
                }
                // Adjective modifies nearest noun
                PosTag::Adjective => {
                    let noun_idx = self.find_nearest_noun(tokens, pos_tags, i);
                    (noun_idx, DependencyType::Adjective)
                }
                // Adverb modifies verb
                PosTag::Adverb => {
                    (root_idx, DependencyType::Adverb)
                }
                // Determiner precedes noun
                PosTag::Determiner if i + 1 < tokens.len() => {
                    (i + 1, DependencyType::Determiner)
                }
                // Preposition starts prepositional phrase
                PosTag::Preposition => {
                    (root_idx, DependencyType::PrepMod)
                }
                // Auxiliary modifies main verb
                PosTag::Auxiliary if i != root_idx => {
                    (root_idx, DependencyType::Auxiliary)
                }
                _ => continue,
            };

            dependencies.push(Dependency {
                head_idx,
                dependent_idx: i,
                relation,
            });
        }

        DependencyTree {
            tokens: tokens.to_vec(),
            pos_tags: pos_tags.to_vec(),
            dependencies,
            root_idx,
        }
    }

    /// Find nearest noun for adjective attachment
    fn find_nearest_noun(&self, _tokens: &[String], pos_tags: &[PosTag], adj_idx: usize) -> usize {
        // Look right first (adjective typically precedes noun)
        for i in (adj_idx + 1)..pos_tags.len() {
            if pos_tags[i] == PosTag::Noun || pos_tags[i] == PosTag::ProperNoun {
                return i;
            }
        }
        // Then look left
        for i in (0..adj_idx).rev() {
            if pos_tags[i] == PosTag::Noun || pos_tags[i] == PosTag::ProperNoun {
                return i;
            }
        }
        adj_idx  // Fallback
    }

    /// Extract semantic roles
    fn extract_roles(&self, tokens: &[String], pos_tags: &[PosTag], deps: &DependencyTree) -> Vec<RolePhrase> {
        let mut roles = Vec::new();

        // Find subject and assign Agent/Experiencer
        for dep in &deps.dependencies {
            if dep.relation == DependencyType::Subject {
                let text = tokens[dep.dependent_idx].clone();
                let verb_class = self.classify_verb(&tokens[deps.root_idx]);

                let role = match verb_class {
                    VerbClass::Experience | VerbClass::Perception => SemanticRole::Experiencer,
                    VerbClass::State => SemanticRole::Theme,
                    _ => SemanticRole::Agent,
                };

                roles.push(RolePhrase {
                    role,
                    text: text.clone(),
                    head_word: text,
                    confidence: 0.85,
                });
            }

            // Object role
            if dep.relation == DependencyType::Object {
                let text = tokens[dep.dependent_idx].clone();
                let verb_class = self.classify_verb(&tokens[deps.root_idx]);

                let role = match verb_class {
                    VerbClass::Experience | VerbClass::Perception => SemanticRole::Stimulus,
                    VerbClass::Transfer => SemanticRole::Theme,
                    _ => SemanticRole::Patient,
                };

                roles.push(RolePhrase {
                    role,
                    text: text.clone(),
                    head_word: text,
                    confidence: 0.80,
                });
            }

            // Prepositional phrases
            if dep.relation == DependencyType::PrepMod {
                let prep = &tokens[dep.dependent_idx];
                if let Some(obj_idx) = self.find_prep_object(dep.dependent_idx, pos_tags) {
                    let obj = &tokens[obj_idx];

                    let role = match prep.as_str() {
                        "in" | "at" | "on" => SemanticRole::Location,
                        "to" | "towards" => SemanticRole::Goal,
                        "from" => SemanticRole::Source,
                        "with" => SemanticRole::Instrument,
                        "for" => SemanticRole::Beneficiary,
                        "because" | "due" => SemanticRole::Cause,
                        "before" | "after" | "during" => SemanticRole::Time,
                        _ => continue,
                    };

                    roles.push(RolePhrase {
                        role,
                        text: format!("{} {}", prep, obj),
                        head_word: obj.clone(),
                        confidence: 0.75,
                    });
                }
            }
        }

        roles
    }

    /// Find object of preposition
    fn find_prep_object(&self, prep_idx: usize, pos_tags: &[PosTag]) -> Option<usize> {
        for i in (prep_idx + 1)..pos_tags.len() {
            if pos_tags[i] == PosTag::Noun || pos_tags[i] == PosTag::Pronoun {
                return Some(i);
            }
        }
        None
    }

    /// Classify verb type
    fn classify_verb(&self, verb: &str) -> VerbClass {
        // Experience verbs
        if ["feel", "think", "believe", "love", "hate", "like", "want", "need", "fear", "hope", "wish"]
            .contains(&verb) {
            return VerbClass::Experience;
        }

        // Perception verbs
        if ["see", "hear", "smell", "taste", "notice", "observe", "watch"]
            .contains(&verb) {
            return VerbClass::Perception;
        }

        // State verbs
        if ["be", "exist", "seem", "appear", "remain", "become", "stay"]
            .contains(&verb) {
            return VerbClass::State;
        }

        // Motion verbs
        if ["go", "come", "move", "travel", "walk", "run", "fly", "arrive", "leave"]
            .contains(&verb) {
            return VerbClass::Motion;
        }

        // Transfer verbs
        if ["give", "send", "bring", "take", "pass", "hand", "offer"]
            .contains(&verb) {
            return VerbClass::Transfer;
        }

        // Communication verbs
        if ["say", "tell", "ask", "speak", "talk", "explain", "describe"]
            .contains(&verb) {
            return VerbClass::Communication;
        }

        // Causative verbs
        if ["make", "cause", "force", "let", "allow", "enable"]
            .contains(&verb) {
            return VerbClass::Causative;
        }

        // Default to action
        VerbClass::Action
    }

    /// Classify intent
    fn classify_intent(&self, text: &str, tokens: &[String]) -> IntentFeatures {
        let text_lower = text.to_lowercase();

        // Check for question mark
        let is_question = text.ends_with('?');

        // Check first word for intent patterns
        let primary = if let Some(first) = tokens.first() {
            if let Some(intent) = self.intent_patterns.get(first) {
                intent.clone()
            } else if is_question {
                // Question without WH-word = yes/no question
                if ["is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "have", "has"]
                    .contains(&first.as_str()) {
                    Intent::YesNoQuestion
                } else {
                    Intent::WhQuestion
                }
            } else if ["please", "could", "would", "can"].iter().any(|w| text_lower.contains(w)) {
                Intent::Request
            } else if ["tell", "show", "explain", "describe"].contains(&first.as_str()) {
                Intent::Command
            } else {
                Intent::Statement
            }
        } else {
            Intent::Unknown
        };

        // Analyze features
        let is_polite = text_lower.contains("please") ||
                        text_lower.contains("could you") ||
                        text_lower.contains("would you");

        let is_urgent = text.contains('!') ||
                        text_lower.contains("urgent") ||
                        text_lower.contains("immediately");

        let is_hypothetical = text_lower.contains("if") ||
                              text_lower.contains("would") ||
                              text_lower.contains("could");

        let is_negated = text_lower.contains("not") ||
                         text_lower.contains("n't") ||
                         text_lower.contains("never") ||
                         text_lower.contains("no ");

        // Certainty based on hedging words
        let certainty = if text_lower.contains("maybe") || text_lower.contains("perhaps") || text_lower.contains("might") {
            0.4
        } else if text_lower.contains("probably") || text_lower.contains("likely") {
            0.7
        } else if text_lower.contains("definitely") || text_lower.contains("certainly") {
            0.95
        } else {
            0.75
        };

        // Formality heuristics
        let formality = if text_lower.contains("please") || text_lower.contains("would") {
            0.7
        } else if text_lower.contains("hey") || text_lower.contains("gonna") || text_lower.contains("wanna") {
            0.2
        } else {
            0.5
        };

        IntentFeatures {
            primary,
            is_polite,
            is_urgent,
            is_hypothetical,
            is_negated,
            certainty,
            formality,
        }
    }

    /// Analyze pragmatics
    fn analyze_pragmatics(&self, text: &str, intent: &IntentFeatures, roles: &[RolePhrase]) -> PragmaticAnalysis {
        let mut presuppositions = Vec::new();
        let mut implicatures = Vec::new();

        // Extract presuppositions
        let text_lower = text.to_lowercase();

        // "Why" questions presuppose the action happened
        if intent.primary == Intent::WhyQuestion {
            if let Some(theme) = roles.iter().find(|r| r.role == SemanticRole::Theme || r.role == SemanticRole::Patient) {
                presuppositions.push(format!("{} exists/happened", theme.text));
            }
        }

        // "Stop X-ing" presupposes X is happening
        if text_lower.contains("stop") {
            presuppositions.push("The action is currently happening".to_string());
        }

        // Definite descriptions presuppose existence
        if text_lower.contains("the ") {
            presuppositions.push("The referenced entity exists and is known".to_string());
        }

        // Implicatures
        if intent.is_polite {
            implicatures.push("Speaker wants something but is being indirect".to_string());
        }

        if text_lower.contains("can you") && !text.contains('?') {
            implicatures.push("Likely a request, not a question about ability".to_string());
        }

        // Speech act
        let speech_act = match intent.primary {
            Intent::Statement | Intent::Fact | Intent::Opinion => SpeechAct::Assertive,
            Intent::Command | Intent::Request | Intent::Suggestion => SpeechAct::Directive,
            Intent::Thanks | Intent::Apology | Intent::Greeting | Intent::Farewell | Intent::Exclamation =>
                SpeechAct::Expressive,
            Intent::YesNoQuestion | Intent::WhQuestion | Intent::HowQuestion | Intent::WhyQuestion =>
                SpeechAct::Directive,  // Questions are requests for information
            Intent::Agreement | Intent::Disagreement | Intent::Confirmation =>
                SpeechAct::Assertive,
            _ => SpeechAct::Assertive,
        };

        // Implied meaning for indirect speech acts
        let implied = if text_lower.contains("can you") && intent.primary != Intent::YesNoQuestion {
            Some("Please do this for me".to_string())
        } else if text_lower.contains("it's cold in here") {
            Some("Please close the window or turn on the heat".to_string())
        } else {
            None
        };

        PragmaticAnalysis {
            literal: text.to_string(),
            implied,
            speech_act,
            presuppositions,
            implicatures,
        }
    }

    /// Extract named entities
    fn extract_entities(&self, tokens: &[String], pos_tags: &[PosTag]) -> Vec<Entity> {
        let mut entities = Vec::new();

        for (i, (token, tag)) in tokens.iter().zip(pos_tags.iter()).enumerate() {
            // Capitalized words (in middle of sentence) likely proper nouns
            if token.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && i > 0 {
                entities.push(Entity {
                    text: token.clone(),
                    entity_type: EntityType::Person,  // Default assumption
                    start_idx: i,
                    end_idx: i + 1,
                });
            }

            // Numbers
            if token.chars().all(|c| c.is_numeric()) {
                entities.push(Entity {
                    text: token.clone(),
                    entity_type: EntityType::Number,
                    start_idx: i,
                    end_idx: i + 1,
                });
            }

            // Time words
            if ["today", "tomorrow", "yesterday", "now", "later", "soon", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                .contains(&token.as_str()) {
                entities.push(Entity {
                    text: token.clone(),
                    entity_type: EntityType::Time,
                    start_idx: i,
                    end_idx: i + 1,
                });
            }
        }

        entities
    }

    /// Get a summary of the parse
    pub fn summarize(&self, parse: &DeepParse) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Intent: {:?}", parse.intent.primary));
        if parse.intent.is_polite {
            summary.push_str(" (polite)");
        }
        if parse.intent.is_negated {
            summary.push_str(" (negated)");
        }
        summary.push('\n');

        summary.push_str(&format!("Speech Act: {:?}\n", parse.pragmatics.speech_act));

        if !parse.roles.is_empty() {
            summary.push_str("Semantic Roles:\n");
            for role in &parse.roles {
                summary.push_str(&format!("  {:?}: {}\n", role.role, role.text));
            }
        }

        if let Some(ref implied) = parse.pragmatics.implied {
            summary.push_str(&format!("Implied: {}\n", implied));
        }

        summary
    }
}

impl Default for DeepParser {
    fn default() -> Self {
        Self::new(Vocabulary::new())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::SentenceType;
    use crate::HV16;

    fn make_basic_parse() -> ParsedSentence {
        ParsedSentence {
            text: "test".to_string(),
            words: vec![],
            sentence_type: SentenceType::Statement,
            unified_encoding: HV16::random(42),
            subject: None,
            predicate: None,
            object: None,
            topics: vec![],
            valence: 0.5,
            arousal: 0.5,
        }
    }

    #[test]
    fn test_deep_parser_creation() {
        let parser = DeepParser::default();
        assert!(!parser.intent_patterns.is_empty());
    }

    #[test]
    fn test_intent_classification_question() {
        let parser = DeepParser::default();
        let tokens = vec!["what".to_string(), "is".to_string(), "love".to_string()];
        let intent = parser.classify_intent("What is love?", &tokens);
        assert_eq!(intent.primary, Intent::WhQuestion);
    }

    #[test]
    fn test_intent_classification_greeting() {
        let parser = DeepParser::default();
        let tokens = vec!["hello".to_string()];
        let intent = parser.classify_intent("Hello!", &tokens);
        assert_eq!(intent.primary, Intent::Greeting);
    }

    #[test]
    fn test_intent_classification_yes_no() {
        let parser = DeepParser::default();
        let tokens = vec!["are".to_string(), "you".to_string(), "conscious".to_string()];
        let intent = parser.classify_intent("Are you conscious?", &tokens);
        assert_eq!(intent.primary, Intent::YesNoQuestion);
    }

    #[test]
    fn test_pos_tagging() {
        let parser = DeepParser::default();
        let tokens = vec!["the".to_string(), "cat".to_string(), "running".to_string()];
        let tags = parser.tag_pos(&tokens);
        assert_eq!(tags[0], PosTag::Determiner);
        assert_eq!(tags[2], PosTag::Verb);  // "running" ends with "ing"
    }

    #[test]
    fn test_verb_classification() {
        let parser = DeepParser::default();
        assert_eq!(parser.classify_verb("feel"), VerbClass::Experience);
        assert_eq!(parser.classify_verb("see"), VerbClass::Perception);
        assert_eq!(parser.classify_verb("go"), VerbClass::Motion);
        assert_eq!(parser.classify_verb("run"), VerbClass::Motion);  // run is motion, not action
        assert_eq!(parser.classify_verb("build"), VerbClass::Action);  // default action
    }

    #[test]
    fn test_semantic_roles() {
        let parser = DeepParser::default();
        let basic = make_basic_parse();
        let parse = parser.parse("I love you", basic);

        // Check that parsing completes without error
        assert!(!parse.text.is_empty());
        // Dependency tree should have the tokens
        assert_eq!(parse.dependencies.tokens.len(), 3);  // I, love, you
    }

    #[test]
    fn test_pragmatic_analysis() {
        let parser = DeepParser::default();
        let basic = make_basic_parse();

        // "the" triggers definite description presupposition
        let parse = parser.parse("Why is the sky blue?", basic);

        assert_eq!(parse.pragmatics.speech_act, SpeechAct::Directive);  // Questions are directives
        assert!(!parse.pragmatics.presuppositions.is_empty());  // "the" adds presupposition
    }

    #[test]
    fn test_politeness_detection() {
        let parser = DeepParser::default();
        let tokens = vec!["could".to_string(), "you".to_string(), "please".to_string()];
        let intent = parser.classify_intent("Could you please help?", &tokens);
        assert!(intent.is_polite);
    }

    #[test]
    fn test_negation_detection() {
        let parser = DeepParser::default();
        let tokens = vec!["i".to_string(), "do".to_string(), "not".to_string()];
        let intent = parser.classify_intent("I do not agree", &tokens);
        assert!(intent.is_negated);
    }

    #[test]
    fn test_formality_casual() {
        let parser = DeepParser::default();
        let tokens = vec!["hey".to_string()];
        let intent = parser.classify_intent("Hey what's up", &tokens);
        assert!(intent.formality < 0.5);
    }

    #[test]
    fn test_entity_extraction() {
        let parser = DeepParser::default();
        let tokens = vec!["i".to_string(), "have".to_string(), "3".to_string(), "cats".to_string()];
        let pos_tags = parser.tag_pos(&tokens);
        let entities = parser.extract_entities(&tokens, &pos_tags);

        let has_number = entities.iter().any(|e| e.entity_type == EntityType::Number);
        assert!(has_number);
    }

    #[test]
    fn test_summarize() {
        let parser = DeepParser::default();
        let basic = make_basic_parse();
        let parse = parser.parse("What is consciousness?", basic);
        let summary = parser.summarize(&parse);

        assert!(summary.contains("Intent:"));
        assert!(summary.contains("Speech Act:"));
    }
}
