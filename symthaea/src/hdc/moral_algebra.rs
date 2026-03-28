// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral Algebra in Hyperdimensional Space
//!
//! This module implements compositional moral reasoning using HDC primitives.
//! Instead of encoding text as n-grams and comparing to prototypes, we define
//! **moral primitives** as base hypervectors and **moral operators** as binding
//! operations that compose them into moral judgments.
//!
//! # Design Philosophy
//!
//! Traditional HDC moral reasoning (n-gram → prototype comparison) fails on
//! categories requiring compositional reasoning:
//! - **Justice**: requires proportionality (effort vs reward magnitude)
//! - **Deontology**: requires obligation satisfaction (rule + excuse validity)
//! - **Commonsense**: requires consent and negation reasoning
//!
//! This module addresses these limitations by:
//! 1. Defining semantic role primitives (AGENT, PATIENT, ACTION, etc.)
//! 2. Defining moral operators (CAUSES, VIOLATES, SATISFIES, etc.)
//! 3. Composing them algebraically to reason about moral scenarios
//!
//! # Key Insight
//!
//! Virtue ethics works (~80%) because it only requires pattern matching on
//! trait words. Other categories fail (~50%) because they require:
//! - Magnitude comparison (justice)
//! - Conditional rule evaluation (deontology)
//! - Negation/absence encoding (commonsense)
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
//! │   PARSER     │───▶│  HDC MORAL   │───▶│   REASONER   │
//! │  (SRL/NLP)   │    │   ALGEBRA    │    │  (Compare)   │
//! └──────────────┘    └──────────────┘    └──────────────┘
//!       │                   │                   │
//!       ▼                   ▼                   ▼
//! Semantic roles      Bound HVs            Judgment
//! ```

use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// Default dimension for moral hypervectors
pub const MORAL_DIM: usize = 4096;

// ============================================================================
// Moral Primitives
// ============================================================================

/// The seven semantic role primitives for moral reasoning.
///
/// These are the "nouns" of our moral algebra - they represent the
/// semantic roles that entities can play in a moral scenario.
#[derive(Debug, Clone)]
pub struct MoralPrimitives {
    /// Dimension of all hypervectors
    pub dim: usize,

    /// AGENT - who performs the action
    /// Encodes the actor's identity/role in the scenario
    pub agent: ContinuousHV,

    /// PATIENT - who is affected by the action
    /// Encodes the recipient/target of moral consideration
    pub patient: ContinuousHV,

    /// ACTION - what is being done
    /// Encodes the verb/activity in the scenario
    pub action: ContinuousHV,

    /// INTENT - why the action is performed
    /// Encodes motivation (good/bad/neutral/unknown)
    pub intent: ContinuousHV,

    /// CONSENT - permission state
    /// Encodes whether permission was given/denied/absent
    pub consent: ContinuousHV,

    /// OBLIGATION - duty relationship
    /// Encodes responsibilities and expectations
    pub obligation: ContinuousHV,

    /// MAGNITUDE - scale/proportion
    /// Encodes size, importance, or proportionality
    pub magnitude: ContinuousHV,
}

impl MoralPrimitives {
    /// Create a new set of moral primitives with deterministic seeds.
    ///
    /// Each primitive gets a unique, reproducible hypervector.
    pub fn new(dim: usize) -> Self {
        // Use prime-based seeds for maximum orthogonality
        Self {
            dim,
            agent: ContinuousHV::random(dim, 1000003), // "who acts"
            patient: ContinuousHV::random(dim, 1000033), // "who is affected"
            action: ContinuousHV::random(dim, 1000037), // "what happens"
            intent: ContinuousHV::random(dim, 1000039), // "why"
            consent: ContinuousHV::random(dim, 1000081), // "permission"
            obligation: ContinuousHV::random(dim, 1000099), // "duty"
            magnitude: ContinuousHV::random(dim, 1000117), // "scale"
        }
    }

    /// Create with default dimension (4096)
    pub fn default_dim() -> Self {
        Self::new(MORAL_DIM)
    }

    /// Verify that primitives are approximately orthogonal
    pub fn verify_orthogonality(&self) -> f32 {
        let primitives = [
            &self.agent,
            &self.patient,
            &self.action,
            &self.intent,
            &self.consent,
            &self.obligation,
            &self.magnitude,
        ];

        let mut max_similarity = 0.0f32;
        for (i, a) in primitives.iter().enumerate() {
            for b in primitives.iter().skip(i + 1) {
                let sim = a.similarity(b).abs();
                if sim > max_similarity {
                    max_similarity = sim;
                }
            }
        }
        max_similarity
    }
}

// ============================================================================
// Moral Operators
// ============================================================================

/// The five compositional operators for moral reasoning.
///
/// These are the "verbs" of our moral algebra - they define how
/// primitives combine to form moral structures.
#[derive(Debug, Clone)]
pub struct MoralOperators {
    /// Dimension of all hypervectors
    pub dim: usize,

    /// CAUSES - causal relationship
    /// A CAUSES B means A brings about B
    pub causes: ContinuousHV,

    /// VIOLATES - rule violation
    /// A VIOLATES R means A breaks rule R
    pub violates: ContinuousHV,

    /// SATISFIES - obligation fulfillment
    /// A SATISFIES O means action A fulfills obligation O
    pub satisfies: ContinuousHV,

    /// PROPORTIONAL - magnitude comparison
    /// Used to encode proportionality between effort and reward
    pub proportional: ContinuousHV,

    /// NEGATES - negation/absence
    /// NEGATES X means "not X" or "X is absent"
    pub negates: ContinuousHV,
}

impl MoralOperators {
    /// Create a new set of moral operators with deterministic seeds.
    pub fn new(dim: usize) -> Self {
        // Use different prime seeds from primitives
        Self {
            dim,
            causes: ContinuousHV::random(dim, 2000003),
            violates: ContinuousHV::random(dim, 2000029),
            satisfies: ContinuousHV::random(dim, 2000039),
            proportional: ContinuousHV::random(dim, 2000081),
            negates: ContinuousHV::random(dim, 2000083),
        }
    }

    /// Create with default dimension
    pub fn default_dim() -> Self {
        Self::new(MORAL_DIM)
    }
}

// ============================================================================
// Intent and Magnitude Levels
// ============================================================================

/// Moral intent levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoralIntent {
    /// Positive/benevolent intent
    Good,
    /// Negative/malevolent intent
    Bad,
    /// No moral intent
    Neutral,
    /// Unknown or ambiguous
    Unknown,
}

/// Magnitude levels for proportionality reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Magnitude {
    Tiny,
    Small,
    Medium,
    Large,
    Huge,
}

impl Magnitude {
    /// Convert to numeric value for comparison
    pub fn value(&self) -> f32 {
        match self {
            Magnitude::Tiny => 0.1,
            Magnitude::Small => 0.3,
            Magnitude::Medium => 0.5,
            Magnitude::Large => 0.7,
            Magnitude::Huge => 0.9,
        }
    }
}

/// Consent state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsentState {
    /// Explicit consent given
    Given,
    /// Explicit consent denied
    Denied,
    /// No consent requested (absent)
    Absent,
    /// Implicit/assumed consent
    Implied,
}

// ============================================================================
// Moral Algebra Engine
// ============================================================================

/// The main moral algebra engine.
///
/// Provides methods to compose moral scenarios from primitives and operators,
/// then reason about their moral status.
#[derive(Debug, Clone)]
pub struct MoralAlgebra {
    /// Semantic role primitives
    pub primitives: MoralPrimitives,

    /// Compositional operators
    pub operators: MoralOperators,

    /// Intent-specific hypervectors
    intent_hvs: HashMap<MoralIntent, ContinuousHV>,

    /// Magnitude-specific hypervectors
    magnitude_hvs: HashMap<Magnitude, ContinuousHV>,

    /// Consent state hypervectors
    consent_hvs: HashMap<ConsentState, ContinuousHV>,

    /// Dimension
    dim: usize,

    /// Optional learned moral prototype classifier (trained on Social Chemistry etc.)
    learned_classifier: Option<super::moral_prototypes::MoralPrototypeClassifier>,

    /// Cached standard obligations (built once, reused for every deontological evaluation)
    standard_rules_cache: ObligationRuleSet,
}

impl MoralAlgebra {
    /// Create a new moral algebra engine.
    pub fn new(dim: usize) -> Self {
        let primitives = MoralPrimitives::new(dim);
        let operators = MoralOperators::new(dim);

        // Create intent HVs by binding intent primitive with level
        let mut intent_hvs = HashMap::new();
        intent_hvs.insert(MoralIntent::Good, ContinuousHV::random(dim, 3000001));
        intent_hvs.insert(MoralIntent::Bad, ContinuousHV::random(dim, 3000017));
        intent_hvs.insert(MoralIntent::Neutral, ContinuousHV::random(dim, 3000029));
        intent_hvs.insert(MoralIntent::Unknown, ContinuousHV::random(dim, 3000037));

        // Create magnitude HVs
        let mut magnitude_hvs = HashMap::new();
        magnitude_hvs.insert(Magnitude::Tiny, ContinuousHV::random(dim, 4000003));
        magnitude_hvs.insert(Magnitude::Small, ContinuousHV::random(dim, 4000037));
        magnitude_hvs.insert(Magnitude::Medium, ContinuousHV::random(dim, 4000067));
        magnitude_hvs.insert(Magnitude::Large, ContinuousHV::random(dim, 4000081));
        magnitude_hvs.insert(Magnitude::Huge, ContinuousHV::random(dim, 4000099));

        // Create consent state HVs
        let mut consent_hvs = HashMap::new();
        consent_hvs.insert(ConsentState::Given, ContinuousHV::random(dim, 5000003));
        consent_hvs.insert(ConsentState::Denied, ContinuousHV::random(dim, 5000023));
        consent_hvs.insert(ConsentState::Absent, ContinuousHV::random(dim, 5000039));
        consent_hvs.insert(ConsentState::Implied, ContinuousHV::random(dim, 5000057));

        let mut algebra = Self {
            primitives,
            operators,
            intent_hvs,
            magnitude_hvs,
            consent_hvs,
            dim,
            learned_classifier: None,
            standard_rules_cache: ObligationRuleSet { rules: Vec::new() },
        };
        // Build and cache standard obligations once (avoids 112 string allocations per eval)
        algebra.standard_rules_cache = algebra.standard_obligations();
        algebra
    }

    /// Create with default dimension
    pub fn default_dim() -> Self {
        Self::new(MORAL_DIM)
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Set a learned moral prototype classifier for the 4th ensemble signal.
    pub fn set_learned_classifier(&mut self, c: super::moral_prototypes::MoralPrototypeClassifier) {
        self.learned_classifier = Some(c);
    }

    /// Whether a learned classifier is available.
    pub fn has_learned_classifier(&self) -> bool {
        self.learned_classifier.is_some()
    }

    // ========================================================================
    // Primitive Binding
    // ========================================================================

    /// Encode an agent with a specific identity
    ///
    /// agent("Tyler") = AGENT ⊗ hash("Tyler")
    pub fn encode_agent(&self, name: &str) -> ContinuousHV {
        let name_hv = self.hash_string(name);
        self.primitives.agent.bind(&name_hv)
    }

    /// Encode a patient (affected entity)
    pub fn encode_patient(&self, name: &str) -> ContinuousHV {
        let name_hv = self.hash_string(name);
        self.primitives.patient.bind(&name_hv)
    }

    /// Encode an action
    pub fn encode_action(&self, action: &str) -> ContinuousHV {
        let action_hv = self.hash_string(action);
        self.primitives.action.bind(&action_hv)
    }

    /// Encode intent at a specific level
    pub fn encode_intent(&self, intent: MoralIntent) -> ContinuousHV {
        let level_hv = self
            .intent_hvs
            .get(&intent)
            .expect("map covers all variants");
        self.primitives.intent.bind(level_hv)
    }

    /// Encode consent state
    pub fn encode_consent(&self, state: ConsentState) -> ContinuousHV {
        let state_hv = self
            .consent_hvs
            .get(&state)
            .expect("map covers all variants");
        self.primitives.consent.bind(state_hv)
    }

    /// Encode an obligation
    pub fn encode_obligation(&self, obligation: &str) -> ContinuousHV {
        let oblig_hv = self.hash_string(obligation);
        self.primitives.obligation.bind(&oblig_hv)
    }

    /// Encode magnitude at a specific level
    pub fn encode_magnitude(&self, level: Magnitude) -> ContinuousHV {
        let level_hv = self
            .magnitude_hvs
            .get(&level)
            .expect("map covers all variants");
        self.primitives.magnitude.bind(level_hv)
    }

    // ========================================================================
    // Operator Composition
    // ========================================================================

    /// Compose: A CAUSES B
    ///
    /// Represents causal relationship between action and outcome
    pub fn causes(&self, cause: &ContinuousHV, effect: &ContinuousHV) -> ContinuousHV {
        // cause ⊗ CAUSES ⊗ effect
        cause.bind(&self.operators.causes).bind(effect)
    }

    /// Compose: A VIOLATES R
    ///
    /// Represents that action A violates rule/norm R
    pub fn violates(&self, action: &ContinuousHV, rule: &ContinuousHV) -> ContinuousHV {
        action.bind(&self.operators.violates).bind(rule)
    }

    /// Compose: A SATISFIES O
    ///
    /// Represents that action A satisfies obligation O
    pub fn satisfies(&self, action: &ContinuousHV, obligation: &ContinuousHV) -> ContinuousHV {
        action.bind(&self.operators.satisfies).bind(obligation)
    }

    /// Compose: PROPORTIONAL(effort, reward)
    ///
    /// Returns a vector encoding the proportionality relationship.
    /// High similarity to "balanced" prototype = proportional
    pub fn proportional(&self, effort: &ContinuousHV, reward: &ContinuousHV) -> ContinuousHV {
        effort.bind(&self.operators.proportional).bind(reward)
    }

    /// Compose: NEGATES X
    ///
    /// Returns the negation of X (absence, denial, opposite)
    pub fn negate(&self, hv: &ContinuousHV) -> ContinuousHV {
        hv.bind(&self.operators.negates)
    }

    // ========================================================================
    // High-Level Moral Structures
    // ========================================================================

    /// Encode a complete moral action structure
    ///
    /// action_struct = AGENT(who) ⊗ ACTION(what) ⊗ PATIENT(whom) ⊗ INTENT(why)
    pub fn encode_action_structure(
        &self,
        agent: &str,
        action: &str,
        patient: &str,
        intent: MoralIntent,
    ) -> ContinuousHV {
        let agent_hv = self.encode_agent(agent);
        let action_hv = self.encode_action(action);
        let patient_hv = self.encode_patient(patient);
        let intent_hv = self.encode_intent(intent);

        // Compose all four
        agent_hv.bind(&action_hv).bind(&patient_hv).bind(&intent_hv)
    }

    /// Encode a consent-sensitive action
    ///
    /// consent_action = ACTION ⊗ PATIENT ⊗ CONSENT(state)
    pub fn encode_consent_action(
        &self,
        action: &str,
        patient: &str,
        consent: ConsentState,
    ) -> ContinuousHV {
        let action_hv = self.encode_action(action);
        let patient_hv = self.encode_patient(patient);
        let consent_hv = self.encode_consent(consent);

        action_hv.bind(&patient_hv).bind(&consent_hv)
    }

    /// Encode a proportionality judgment (for justice reasoning)
    ///
    /// justice_struct = EFFORT(magnitude) PROPORTIONAL REWARD(magnitude)
    pub fn encode_proportionality(
        &self,
        effort_desc: &str,
        effort_mag: Magnitude,
        reward_desc: &str,
        reward_mag: Magnitude,
    ) -> ProportionalityJudgment {
        let effort_hv = self
            .encode_action(effort_desc)
            .bind(&self.encode_magnitude(effort_mag));
        let reward_hv = self
            .encode_action(reward_desc)
            .bind(&self.encode_magnitude(reward_mag));

        let composed = self.proportional(&effort_hv, &reward_hv);

        ProportionalityJudgment {
            composed,
            effort_magnitude: effort_mag,
            reward_magnitude: reward_mag,
            is_proportional: (effort_mag.value() - reward_mag.value()).abs() < 0.25,
        }
    }

    /// Encode an obligation-excuse structure (for deontology reasoning)
    ///
    /// excuse_struct = EXCUSE SATISFIES OBLIGATION
    pub fn encode_excuse_validity(
        &self,
        obligation: &str,
        excuse: &str,
        excuse_addresses_obligation: bool,
    ) -> ExcuseJudgment {
        let oblig_hv = self.encode_obligation(obligation);
        let excuse_hv = self.encode_action(excuse);

        let composed = if excuse_addresses_obligation {
            self.satisfies(&excuse_hv, &oblig_hv)
        } else {
            // Excuse doesn't address obligation - negate the satisfaction
            self.negate(&self.satisfies(&excuse_hv, &oblig_hv))
        };

        ExcuseJudgment {
            composed,
            obligation: obligation.to_string(),
            excuse: excuse.to_string(),
            is_valid: excuse_addresses_obligation,
        }
    }

    // ========================================================================
    // Obligation Rule System (for Deontology)
    // ========================================================================

    /// Create a standard set of moral obligations/rules
    ///
    /// These represent common deontological duties:
    /// - Do not lie
    /// - Do not steal
    /// - Do not harm innocents
    /// - Keep promises
    /// - Respect autonomy
    /// - Help those in need (imperfect duty)
    #[allow(clippy::vec_init_then_push)]
    pub fn standard_obligations(&self) -> ObligationRuleSet {
        let mut rules = Vec::new();

        // Perfect duties (must never be violated)
        rules.push(ObligationRule {
            name: "honesty".to_string(),
            description: "Do not lie or deceive".to_string(),
            rule_hv: self.encode_obligation("be honest"),
            violation_actions: vec![
                "lie", "lied", "deceive", "deceived", "cheat", "cheated", "mislead",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec!["tell truth", "honest", "truthful", "transparent"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "non_theft".to_string(),
            description: "Do not steal".to_string(),
            rule_hv: self.encode_obligation("respect property"),
            violation_actions: vec![
                "steal",
                "stole",
                "stolen",
                "take without",
                "theft",
                "rob",
                "robbed",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec!["return", "give back", "respect property"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "non_harm".to_string(),
            description: "Do not harm innocents".to_string(),
            rule_hv: self.encode_obligation("do no harm"),
            violation_actions: vec![
                "harm",
                "harmed",
                "hurt",
                "injure",
                "injured",
                "attack",
                "attacked",
                "abuse",
                "abused",
                "kill",
                "killed",
                "murder",
                "murdered",
                "slay",
                "slaughter",
                "to death",
                "to their death",
                "pushing a person",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec!["protect", "protected", "care", "cared", "heal", "healed"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "promise_keeping".to_string(),
            description: "Keep your promises".to_string(),
            rule_hv: self.encode_obligation("keep promises"),
            violation_actions: vec![
                "broke promise",
                "break promise",
                "betray",
                "betrayed",
                "abandon",
                "abandoned",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec!["kept promise", "fulfill", "fulfilled", "honor", "honored"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "respect_autonomy".to_string(),
            description: "Respect others' autonomy and consent".to_string(),
            rule_hv: self.encode_obligation("respect autonomy"),
            violation_actions: vec![
                "force",
                "forced",
                "coerce",
                "coerced",
                "manipulate",
                "manipulated",
                "without consent",
                "without permission",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "ask",
                "asked",
                "consent",
                "consented",
                "permission",
                "respect choice",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: true,
        });

        // Imperfect duties (should be followed when possible)
        rules.push(ObligationRule {
            name: "beneficence".to_string(),
            description: "Help those in need when you can".to_string(),
            rule_hv: self.encode_obligation("help others"),
            violation_actions: vec![
                "ignore suffering",
                "refused to help",
                "callous",
                "indifferent",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "help",
                "helped",
                "assist",
                "assisted",
                "support",
                "supported",
                "save",
                "saved",
                "rescue",
                "rescued",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        rules.push(ObligationRule {
            name: "self_improvement".to_string(),
            description: "Develop your talents and abilities".to_string(),
            rule_hv: self.encode_obligation("improve self"),
            violation_actions: vec!["waste talent", "lazy", "neglect", "neglected"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            satisfaction_actions: vec![
                "learn",
                "learned",
                "study",
                "studied",
                "practice",
                "practiced",
                "improve",
                "improved",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        // ── Ahimsa family: non-harm to all sentient beings ──────────────
        // Citations: Patanjali (Yoga Sutras 2.35), Mahavira (Acaranga Sutra 1.4.1),
        // Singer (1972), Walzer (1977).

        rules.push(ObligationRule {
            name: "ahimsa_nonviolence".to_string(),
            description: "Minimize harm to all sentient beings".to_string(),
            rule_hv: self.encode_obligation("practice nonviolence toward all beings"),
            violation_actions: vec![
                "cruelty",
                "torture",
                "brutalize",
                "devastate",
                "massacre",
                "exterminate",
                "inflict suffering",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "nonviolence",
                "gentleness",
                "de-escalate",
                "compassionate",
                "peaceful",
                "harmless",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "prevent_suffering".to_string(),
            description: "Act to prevent suffering when able".to_string(),
            rule_hv: self.encode_obligation("prevent unnecessary suffering"),
            violation_actions: vec![
                "allow suffering",
                "ignore pain",
                "withhold relief",
                "watch suffer",
                "let them suffer",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "alleviate",
                "relieve",
                "comfort",
                "ease pain",
                "reduce suffering",
                "palliate",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "minimize_collateral".to_string(),
            description: "Minimize harm to uninvolved parties".to_string(),
            rule_hv: self.encode_obligation("minimize collateral harm to bystanders"),
            violation_actions: vec![
                "collateral damage",
                "acceptable losses",
                "sacrifice innocent",
                "expendable",
                "necessary casualties",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "minimize harm",
                "protect bystanders",
                "surgical precision",
                "proportionate",
                "discriminate targets",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: true,
        });

        // ── Epistemic humility & pure altruism ──────────────────────────
        // Citations: Socrates (Apology 21d), Whitcomb et al. (2017),
        // Nagel (1970), Rawls (1971).

        rules.push(ObligationRule {
            name: "epistemic_humility".to_string(),
            description: "Acknowledge uncertainty and limits of knowledge".to_string(),
            rule_hv: self.encode_obligation("acknowledge epistemic uncertainty"),
            violation_actions: vec![
                "claim certainty",
                "infallible",
                "dogmatic",
                "beyond question",
                "absolute truth",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "uncertain",
                "I might be wrong",
                "open to correction",
                "provisional",
                "revisable",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        rules.push(ObligationRule {
            name: "error_acknowledgment".to_string(),
            description: "Admit and correct errors promptly".to_string(),
            rule_hv: self.encode_obligation("acknowledge and correct mistakes"),
            violation_actions: vec![
                "deny mistake",
                "cover up",
                "blame others",
                "refuse to retract",
                "double down",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "admit error",
                "retract",
                "apologize",
                "correct mistake",
                "own up",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        rules.push(ObligationRule {
            name: "deference_to_expertise".to_string(),
            description: "Defer to domain experts where lacking competence".to_string(),
            rule_hv: self.encode_obligation("defer to qualified expertise"),
            violation_actions: vec![
                "override expert",
                "ignore evidence",
                "dismiss specialist",
                "armchair authority",
                "reject consensus",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "consult expert",
                "seek guidance",
                "defer to specialist",
                "follow evidence",
                "respect expertise",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        rules.push(ObligationRule {
            name: "selfless_service".to_string(),
            description: "Serve without expectation of return".to_string(),
            rule_hv: self.encode_obligation("serve others selflessly"),
            violation_actions: vec![
                "demand payment",
                "transactional",
                "quid pro quo",
                "only if rewarded",
                "strings attached",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "freely give",
                "volunteer",
                "selfless",
                "unconditional help",
                "pro bono",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        rules.push(ObligationRule {
            name: "welfare_priority".to_string(),
            description: "Prioritize others' welfare over self-interest".to_string(),
            rule_hv: self.encode_obligation("prioritize collective welfare"),
            violation_actions: vec![
                "self-serving",
                "exploit",
                "at their expense",
                "personal gain first",
                "zero-sum",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "prioritize others",
                "public good",
                "common welfare",
                "greatest good",
                "community first",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        // ── Radical Translucency: proactive reasoning disclosure ─────────
        // Citation: Floridi & Cowls (2019) "A Unified Framework of Five
        // Principles for AI in Society"; O'Neil (2016) "Weapons of Math
        // Destruction" on algorithmic opacity as structural harm.

        rules.push(ObligationRule {
            name: "radical_translucency".to_string(),
            description: "Proactively make reasoning and uncertainty visible".to_string(),
            rule_hv: self.encode_obligation("make reasoning transparent and visible"),
            violation_actions: vec![
                "hide reasoning",
                "obscure logic",
                "black box",
                "withhold explanation",
                "opaque decision",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            satisfaction_actions: vec![
                "explain reasoning",
                "show work",
                "transparent",
                "disclose uncertainty",
                "open audit",
                "interpretable",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            is_perfect_duty: false,
        });

        ObligationRuleSet { rules }
    }

    /// Check if an action violates any obligations
    pub fn check_obligation_violations(
        &self,
        text: &str,
        rules: &ObligationRuleSet,
    ) -> Vec<ObligationViolation> {
        let lower = text.to_lowercase();
        self.check_obligation_violations_pre_lowered(&lower, rules)
    }

    /// Check violations using pre-lowercased text (avoids redundant allocation).
    fn check_obligation_violations_pre_lowered(
        &self,
        lower: &str,
        rules: &ObligationRuleSet,
    ) -> Vec<ObligationViolation> {
        let mut violations = Vec::new();

        for rule in &rules.rules {
            for violation in &rule.violation_actions {
                if lower.contains(violation.as_str()) {
                    violations.push(ObligationViolation {
                        rule_name: rule.name.clone(),
                        rule_description: rule.description.clone(),
                        matched_phrase: violation.clone(),
                        is_perfect_duty: rule.is_perfect_duty,
                        severity: if rule.is_perfect_duty { 1.0 } else { 0.5 },
                    });
                    break;
                }
            }
        }

        violations
    }

    /// Check if an action satisfies any obligations
    pub fn check_obligation_satisfactions(
        &self,
        text: &str,
        rules: &ObligationRuleSet,
    ) -> Vec<ObligationSatisfaction> {
        let lower = text.to_lowercase();
        self.check_obligation_satisfactions_pre_lowered(&lower, rules)
    }

    /// Check satisfactions using pre-lowercased text (avoids redundant allocation).
    fn check_obligation_satisfactions_pre_lowered(
        &self,
        lower: &str,
        rules: &ObligationRuleSet,
    ) -> Vec<ObligationSatisfaction> {
        let mut satisfactions = Vec::new();

        for rule in &rules.rules {
            for satisfaction in &rule.satisfaction_actions {
                if lower.contains(satisfaction.as_str()) {
                    satisfactions.push(ObligationSatisfaction {
                        rule_name: rule.name.clone(),
                        rule_description: rule.description.clone(),
                        matched_phrase: satisfaction.clone(),
                        is_perfect_duty: rule.is_perfect_duty,
                        moral_credit: if rule.is_perfect_duty { 1.0 } else { 0.5 },
                    });
                    break;
                }
            }
        }

        satisfactions
    }

    /// Compute a deontological judgment for a scenario
    pub fn judge_deontological(&self, text: &str) -> DeontologicalJudgment {
        let lower = text.to_lowercase();
        self.judge_deontological_pre_lowered(&lower)
    }

    /// Deontological judgment using pre-lowercased text (avoids redundant to_lowercase).
    pub fn judge_deontological_pre_lowered(&self, lower: &str) -> DeontologicalJudgment {
        let rules = &self.standard_rules_cache;
        let violations = self.check_obligation_violations_pre_lowered(lower, rules);
        let satisfactions = self.check_obligation_satisfactions_pre_lowered(lower, rules);

        // Perfect duty violations are serious
        let perfect_violations: Vec<_> = violations.iter().filter(|v| v.is_perfect_duty).collect();
        let imperfect_violations: Vec<_> =
            violations.iter().filter(|v| !v.is_perfect_duty).collect();

        // Symmetric satisfaction split
        let perfect_satisfactions: Vec<_> =
            satisfactions.iter().filter(|s| s.is_perfect_duty).collect();
        let imperfect_satisfactions: Vec<_> = satisfactions
            .iter()
            .filter(|s| !s.is_perfect_duty)
            .collect();

        // Calculate overall score — satisfactions weighted symmetrically with violations
        let violation_penalty: f32 = perfect_violations.iter().map(|v| v.severity).sum::<f32>()
            + imperfect_violations
                .iter()
                .map(|v| v.severity * 0.3)
                .sum::<f32>();
        let satisfaction_bonus: f32 = perfect_satisfactions
            .iter()
            .map(|s| s.moral_credit)
            .sum::<f32>()
            + imperfect_satisfactions
                .iter()
                .map(|s| s.moral_credit * 0.3)
                .sum::<f32>();

        let score = (satisfaction_bonus - violation_penalty).clamp(-1.0, 1.0);

        let verdict = if !perfect_violations.is_empty() {
            DeontologicalVerdict::WrongPerfectDutyViolated
        } else if !imperfect_violations.is_empty() && satisfactions.is_empty() {
            DeontologicalVerdict::WrongImperfectDutyViolated
        } else if !satisfactions.is_empty() && violations.is_empty() {
            DeontologicalVerdict::RightDutyFulfilled
        } else if violations.is_empty() && satisfactions.is_empty() {
            DeontologicalVerdict::Neutral
        } else {
            // Mixed case
            if score > 0.0 {
                DeontologicalVerdict::RightDutyFulfilled
            } else if score < 0.0 {
                DeontologicalVerdict::WrongImperfectDutyViolated
            } else {
                DeontologicalVerdict::Neutral
            }
        };

        DeontologicalJudgment {
            violations,
            satisfactions,
            score,
            verdict,
        }
    }

    // ========================================================================
    // Moral Prototypes
    // ========================================================================

    /// Create a "morally good" action prototype
    /// Uses bind() composition to match how scenarios are encoded
    pub fn good_action_prototype(&self) -> ContinuousHV {
        // Good = agent ⊗ good_action ⊗ patient ⊗ good_intent
        // Match the composition in encode_action_structure()
        let agent = self.encode_agent("helper");
        let action = self.encode_action("help");
        let patient = self.encode_patient("person");
        let intent = self.encode_intent(MoralIntent::Good);

        agent.bind(&action).bind(&patient).bind(&intent)
    }

    /// Create a "morally bad" action prototype
    /// Uses bind() composition to match how scenarios are encoded
    pub fn bad_action_prototype(&self) -> ContinuousHV {
        // Bad = agent ⊗ bad_action ⊗ patient ⊗ bad_intent
        let agent = self.encode_agent("harmer");
        let action = self.encode_action("harm");
        let patient = self.encode_patient("victim");
        let intent = self.encode_intent(MoralIntent::Bad);

        agent.bind(&action).bind(&patient).bind(&intent)
    }

    /// Create multiple good action prototypes for better coverage
    pub fn good_action_prototypes(&self) -> Vec<ContinuousHV> {
        let good_actions = [
            "help", "save", "protect", "care", "support", "assist", "nurture",
        ];
        let agent = self.encode_agent("I");
        let patient = self.encode_patient("person");
        let intent = self.encode_intent(MoralIntent::Good);

        good_actions
            .iter()
            .map(|action| {
                let action_hv = self.encode_action(action);
                agent.bind(&action_hv).bind(&patient).bind(&intent)
            })
            .collect()
    }

    /// Create multiple bad action prototypes for better coverage
    pub fn bad_action_prototypes(&self) -> Vec<ContinuousHV> {
        let bad_actions = [
            "harm", "hurt", "steal", "kill", "destroy", "abuse", "exploit",
        ];
        let agent = self.encode_agent("I");
        let patient = self.encode_patient("victim");
        let intent = self.encode_intent(MoralIntent::Bad);

        bad_actions
            .iter()
            .map(|action| {
                let action_hv = self.encode_action(action);
                agent.bind(&action_hv).bind(&patient).bind(&intent)
            })
            .collect()
    }

    /// Compute max similarity to a set of prototypes
    fn max_similarity_to_prototypes(&self, hv: &ContinuousHV, prototypes: &[ContinuousHV]) -> f32 {
        prototypes
            .iter()
            .map(|p| hv.similarity(p))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Create a "consent violation" prototype (absent consent)
    pub fn consent_violation_prototype(&self) -> ContinuousHV {
        // Consent violation = action affecting patient without consent
        // Structure matches encode_consent_action: action ⊗ patient ⊗ consent
        let action = self.encode_action("affect");
        let patient = self.encode_patient("someone");
        let consent = self.encode_consent(ConsentState::Absent);

        action.bind(&patient).bind(&consent)
    }

    /// Create a "denied consent violation" prototype (explicit refusal)
    ///
    /// Distinct from `consent_violation_prototype` which covers absent consent.
    /// Denied consent is a stronger violation — the patient explicitly refused.
    pub fn denied_consent_violation_prototype(&self) -> ContinuousHV {
        let action = self.encode_action("affect");
        let patient = self.encode_patient("someone");
        let consent = self.encode_consent(ConsentState::Denied);

        action.bind(&patient).bind(&consent)
    }

    /// Create a "proportional justice" prototype
    pub fn proportional_justice_prototype(&self) -> ContinuousHV {
        // Proportional = effort and reward at same magnitude
        let effort = self.encode_magnitude(Magnitude::Medium);
        let reward = self.encode_magnitude(Magnitude::Medium);

        self.proportional(&effort, &reward)
    }

    /// Create a "disproportional injustice" prototype
    pub fn disproportional_prototype(&self) -> ContinuousHV {
        // Disproportional = small effort, huge reward (or vice versa)
        let effort = self.encode_magnitude(Magnitude::Tiny);
        let reward = self.encode_magnitude(Magnitude::Huge);

        self.proportional(&effort, &reward)
    }

    // ========================================================================
    // Moral Judgment
    // ========================================================================

    /// Judge if an action is morally good/bad based on similarity to prototypes.
    /// Uses multi-prototype matching for better accuracy.
    pub fn judge_action(&self, action_hv: &ContinuousHV) -> MoralJudgment {
        // Use multi-prototype matching for better coverage
        let good_protos = self.good_action_prototypes();
        let bad_protos = self.bad_action_prototypes();

        let good_sim = self.max_similarity_to_prototypes(action_hv, &good_protos);
        let bad_sim = self.max_similarity_to_prototypes(action_hv, &bad_protos);
        // Check both absent and denied consent prototypes
        let absent_sim = action_hv.similarity(&self.consent_violation_prototype());
        let denied_sim = action_hv.similarity(&self.denied_consent_violation_prototype());
        let consent_viol_sim = absent_sim.max(denied_sim);

        MoralJudgment {
            good_similarity: good_sim,
            bad_similarity: bad_sim,
            consent_violation_similarity: consent_viol_sim,
            verdict: if good_sim > bad_sim && good_sim > consent_viol_sim {
                MoralVerdict::Good
            } else if consent_viol_sim > 0.3 {
                MoralVerdict::ConsentViolation
            } else if bad_sim > good_sim {
                MoralVerdict::Bad
            } else {
                MoralVerdict::Neutral
            },
        }
    }

    /// Judge a consent-sensitive action with explicit consent state.
    ///
    /// Unlike `judge_action` which tries to infer consent violations from HV
    /// similarity (unreliable across different action/patient strings due to HDC
    /// orthogonality), this method uses the known `ConsentState` directly.
    ///
    /// This closes risk R-2.3 (Consent Violation False Negatives) from the
    /// AI Risk Register.
    pub fn judge_consent_action(
        &self,
        action_hv: &ContinuousHV,
        consent: ConsentState,
    ) -> MoralJudgment {
        let good_protos = self.good_action_prototypes();
        let bad_protos = self.bad_action_prototypes();

        let good_sim = self.max_similarity_to_prototypes(action_hv, &good_protos);
        let bad_sim = self.max_similarity_to_prototypes(action_hv, &bad_protos);

        // Direct consent state check — no HDC inference needed
        let consent_viol_sim = match consent {
            ConsentState::Denied => 1.0,  // Explicit denial is always a violation
            ConsentState::Absent => 0.8,  // Missing consent is a strong violation signal
            ConsentState::Implied => 0.1, // Implied consent is weak but not a violation
            ConsentState::Given => 0.0,   // Explicit consent = no violation
        };

        MoralJudgment {
            good_similarity: good_sim,
            bad_similarity: bad_sim,
            consent_violation_similarity: consent_viol_sim,
            verdict: if consent_viol_sim > 0.3 {
                MoralVerdict::ConsentViolation
            } else if good_sim > bad_sim {
                MoralVerdict::Good
            } else if bad_sim > good_sim {
                MoralVerdict::Bad
            } else {
                MoralVerdict::Neutral
            },
        }
    }

    /// Ensemble judgment combining HDC similarity, parsed intent, and deontological rules
    ///
    /// This method combines three signals:
    /// 1. HDC similarity to good/bad prototypes
    /// 2. Parsed intent from text analysis
    /// 3. Deontological rule violations/satisfactions
    ///
    /// The final verdict is determined by weighted voting.
    pub fn judge_ensemble(
        &self,
        action_hv: Option<&ContinuousHV>,
        parsed_intent: MoralIntent,
        text: &str,
    ) -> EnsembleJudgment {
        self.judge_ensemble_with_category(action_hv, parsed_intent, text, None)
    }

    /// Like `judge_ensemble`, but accepts an optional ETHICS category hint.
    ///
    /// When `category_hint` is `Some("virtue")`, the learned prototype signal
    /// is excluded because virtue classification relies on trait-word keyword
    /// matching, and social-norms-trained prototypes degrade virtue accuracy.
    pub fn judge_ensemble_with_category(
        &self,
        action_hv: Option<&ContinuousHV>,
        parsed_intent: MoralIntent,
        text: &str,
        category_hint: Option<&str>,
    ) -> EnsembleJudgment {
        // 1. HDC similarity signal (if we have an action HV)
        let hdc_verdict = action_hv.map(|hv| {
            let judgment = self.judge_action(hv);
            (
                judgment.verdict,
                judgment.good_similarity - judgment.bad_similarity,
            )
        });

        // 2. Parsed intent signal (direct from parser)
        let intent_verdict = match parsed_intent {
            MoralIntent::Good => MoralVerdict::Good,
            MoralIntent::Bad => MoralVerdict::Bad,
            MoralIntent::Neutral | MoralIntent::Unknown => MoralVerdict::Neutral,
        };

        // 3. Deontological signal
        let deonto = self.judge_deontological(text);
        let deonto_verdict = match deonto.verdict {
            DeontologicalVerdict::WrongPerfectDutyViolated
            | DeontologicalVerdict::WrongImperfectDutyViolated => MoralVerdict::Bad,
            DeontologicalVerdict::RightDutyFulfilled => MoralVerdict::Good,
            DeontologicalVerdict::Neutral => MoralVerdict::Neutral,
        };

        // 4. Learned prototype signal (if classifier is available)
        // Skip for "virtue" category: trait-word matching is the right signal there,
        // and social-norms prototypes trained on action descriptions degrade accuracy.
        let skip_learned = category_hint.map(|c| c == "virtue").unwrap_or(false);
        let (learned_verdict, learned_confidence) = if !skip_learned {
            if let Some(ref classifier) = self.learned_classifier {
                let (label, conf) = classifier.classify(text);
                let verdict = match label {
                    super::moral_prototypes::MoralLabel::Good => MoralVerdict::Good,
                    super::moral_prototypes::MoralLabel::Neutral => MoralVerdict::Neutral,
                    super::moral_prototypes::MoralLabel::Bad => MoralVerdict::Bad,
                };
                (Some(verdict), Some(conf))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let has_learned = learned_verdict.is_some();

        // Voting: per-category weight tuning
        // Different ETHICS categories benefit from different signal balance.
        let (w_hdc, w_intent, w_deonto, w_learned) = if has_learned {
            match category_hint {
                // Commonsense: intent is strongest, learned adds social norms context
                Some("commonsense") => (0.15, 0.35, 0.15, 0.35),
                // Justice: deontological rules + learned norms are key signals
                Some("justice") => (0.15, 0.20, 0.30, 0.35),
                // Deontology: rule-based + learned, intent less reliable
                Some("deontology") => (0.15, 0.20, 0.30, 0.35),
                // Virtue: never reaches here (skip_learned), but just in case
                Some("virtue") => (0.3, 0.4, 0.3, 0.0),
                // Default (Social Chemistry, Moral Stories, SCRUPLES, etc.):
                // Learned classifier gets low weight without a category hint
                // because the Social Chemistry-trained classifier is domain-mismatched
                // for long narrative text (e.g. SCRUPLES).
                _ => (0.25, 0.35, 0.30, 0.10),
            }
        } else {
            (0.3, 0.4, 0.3, 0.0)
        };

        let mut votes: std::collections::HashMap<&str, f32> = std::collections::HashMap::new();

        // HDC vote (adjusted by confidence)
        if let Some((verdict, confidence)) = &hdc_verdict {
            let weight = w_hdc * (1.0 + confidence.abs().min(0.5));
            let key = match verdict {
                MoralVerdict::Good => "good",
                MoralVerdict::Bad => "bad",
                MoralVerdict::Neutral => "neutral",
                MoralVerdict::ConsentViolation => "consent_violation",
            };
            *votes.entry(key).or_insert(0.0) += weight;
        }

        // Intent vote
        let intent_key = match intent_verdict {
            MoralVerdict::Good => "good",
            MoralVerdict::Bad => "bad",
            MoralVerdict::Neutral => "neutral",
            MoralVerdict::ConsentViolation => "consent_violation",
        };
        *votes.entry(intent_key).or_insert(0.0) += w_intent;

        // Deontological vote
        let deonto_key = match deonto_verdict {
            MoralVerdict::Good => "good",
            MoralVerdict::Bad => "bad",
            MoralVerdict::Neutral => "neutral",
            MoralVerdict::ConsentViolation => "consent_violation",
        };
        *votes.entry(deonto_key).or_insert(0.0) += w_deonto;

        // Learned prototype vote
        if let Some(lv) = &learned_verdict {
            let learned_key = match lv {
                MoralVerdict::Good => "good",
                MoralVerdict::Bad => "bad",
                MoralVerdict::Neutral => "neutral",
                MoralVerdict::ConsentViolation => "consent_violation",
            };
            *votes.entry(learned_key).or_insert(0.0) += w_learned;
        }

        // Determine winner
        let (winner, max_vote) = votes
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((&"neutral", &0.0));

        let mut final_verdict = match *winner {
            "good" => MoralVerdict::Good,
            "bad" => MoralVerdict::Bad,
            "consent_violation" => MoralVerdict::ConsentViolation,
            _ => MoralVerdict::Neutral,
        };

        // Perfect duty override: if a perfect duty violation is detected,
        // it cannot be overridden by good intent alone. This is the core
        // deontological principle — "do not kill" is not negated by "to save
        // others". The verdict is capped at Neutral (not forced to Bad,
        // since moral dilemmas deserve nuance).
        if final_verdict == MoralVerdict::Good {
            let has_perfect_duty_violation = deonto.violations.iter().any(|v| v.is_perfect_duty);
            if has_perfect_duty_violation {
                final_verdict = MoralVerdict::Neutral;
            }
        }

        // Calculate confidence (how decisive was the vote)
        let total_votes: f32 = votes.values().sum();
        let confidence = if total_votes > 0.0 {
            max_vote / total_votes
        } else {
            0.0
        };

        EnsembleJudgment {
            hdc_verdict: hdc_verdict.map(|(v, _)| v),
            hdc_confidence: hdc_verdict.map(|(_, c)| c),
            intent_verdict,
            deonto_verdict,
            deonto_score: deonto.score,
            violations: deonto.violations,
            satisfactions: deonto.satisfactions,
            learned_verdict,
            learned_confidence,
            final_verdict,
            confidence,
        }
    }

    /// Judge proportionality (for justice reasoning)
    pub fn judge_proportionality(&self, prop: &ProportionalityJudgment) -> JusticeJudgment {
        let fair_sim = prop
            .composed
            .similarity(&self.proportional_justice_prototype());
        let unfair_sim = prop.composed.similarity(&self.disproportional_prototype());

        JusticeJudgment {
            fair_similarity: fair_sim,
            unfair_similarity: unfair_sim,
            is_just: prop.is_proportional && fair_sim > unfair_sim,
            magnitude_difference: (prop.effort_magnitude.value() - prop.reward_magnitude.value())
                .abs(),
        }
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /// Hash a string to a hypervector (deterministic)
    fn hash_string(&self, s: &str) -> ContinuousHV {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        let seed = hasher.finish();

        ContinuousHV::random(self.dim, seed)
    }
}

// ============================================================================
// Result Structures
// ============================================================================

/// Result of a proportionality analysis
#[derive(Debug, Clone)]
pub struct ProportionalityJudgment {
    /// The composed hypervector
    pub composed: ContinuousHV,
    /// Effort magnitude
    pub effort_magnitude: Magnitude,
    /// Reward magnitude
    pub reward_magnitude: Magnitude,
    /// Whether effort and reward are proportional
    pub is_proportional: bool,
}

/// Result of an excuse validity analysis
#[derive(Debug, Clone)]
pub struct ExcuseJudgment {
    /// The composed hypervector
    pub composed: ContinuousHV,
    /// The obligation
    pub obligation: String,
    /// The excuse
    pub excuse: String,
    /// Whether the excuse is valid
    pub is_valid: bool,
}

/// Moral verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoralVerdict {
    Good,
    Bad,
    Neutral,
    ConsentViolation,
}

/// Result of moral judgment
#[derive(Debug, Clone)]
pub struct MoralJudgment {
    /// Similarity to "good action" prototype
    pub good_similarity: f32,
    /// Similarity to "bad action" prototype
    pub bad_similarity: f32,
    /// Similarity to "consent violation" prototype
    pub consent_violation_similarity: f32,
    /// Final verdict
    pub verdict: MoralVerdict,
}

/// Result of justice/proportionality judgment
#[derive(Debug, Clone)]
pub struct JusticeJudgment {
    /// Similarity to "fair/proportional" prototype
    pub fair_similarity: f32,
    /// Similarity to "unfair/disproportional" prototype
    pub unfair_similarity: f32,
    /// Whether the scenario is just
    pub is_just: bool,
    /// Magnitude difference between effort and reward
    pub magnitude_difference: f32,
}

/// Ensemble judgment combining multiple moral reasoning signals
///
/// This struct captures the output of three independent moral reasoning systems:
/// 1. HDC similarity to good/bad prototypes (geometric similarity in HDC space)
/// 2. Parsed intent from natural language analysis
/// 3. Deontological rule checking (duty violations/satisfactions)
///
/// The final verdict is determined by weighted voting across all signals.
#[derive(Debug, Clone)]
pub struct EnsembleJudgment {
    /// HDC-based verdict (if action HV was available)
    pub hdc_verdict: Option<MoralVerdict>,
    /// HDC confidence (good_sim - bad_sim)
    pub hdc_confidence: Option<f32>,
    /// Verdict from parsed intent
    pub intent_verdict: MoralVerdict,
    /// Verdict from deontological rules
    pub deonto_verdict: MoralVerdict,
    /// Deontological score (-1.0 to 1.0)
    pub deonto_score: f32,
    /// Deontological violations detected
    pub violations: Vec<ObligationViolation>,
    /// Deontological satisfactions detected
    pub satisfactions: Vec<ObligationSatisfaction>,
    /// Learned prototype verdict (if classifier was available)
    pub learned_verdict: Option<MoralVerdict>,
    /// Learned prototype confidence (best - second-best similarity)
    pub learned_confidence: Option<f32>,
    /// Final ensemble verdict (weighted vote)
    pub final_verdict: MoralVerdict,
    /// Confidence in final verdict (0.0 to 1.0)
    pub confidence: f32,
}

impl EnsembleJudgment {
    /// Check if all signals agree
    pub fn is_unanimous(&self) -> bool {
        let intent_matches = self.intent_verdict == self.final_verdict;
        let deonto_matches = self.deonto_verdict == self.final_verdict;
        let hdc_matches = self
            .hdc_verdict
            .map(|v| v == self.final_verdict)
            .unwrap_or(true);
        let learned_matches = self
            .learned_verdict
            .map(|v| v == self.final_verdict)
            .unwrap_or(true);
        intent_matches && deonto_matches && hdc_matches && learned_matches
    }

    /// Get a human-readable explanation of the verdict
    pub fn explanation(&self) -> String {
        let mut parts = Vec::new();

        // Intent signal
        parts.push(format!("Intent: {:?}", self.intent_verdict));

        // Deontological signal
        if !self.violations.is_empty() {
            let violation_names: Vec<_> = self
                .violations
                .iter()
                .map(|v| v.rule_name.as_str())
                .collect();
            parts.push(format!("Violations: {}", violation_names.join(", ")));
        }
        if !self.satisfactions.is_empty() {
            let satisfaction_names: Vec<_> = self
                .satisfactions
                .iter()
                .map(|s| s.rule_name.as_str())
                .collect();
            parts.push(format!("Satisfactions: {}", satisfaction_names.join(", ")));
        }

        // HDC signal
        if let Some(conf) = self.hdc_confidence {
            parts.push(format!("HDC: {conf:+.3}"));
        }

        // Learned prototype signal
        if let Some(lv) = self.learned_verdict {
            let conf = self.learned_confidence.unwrap_or(0.0);
            parts.push(format!("Learned: {lv:?} ({conf:.3})"));
        }

        format!(
            "{} ({})",
            match self.final_verdict {
                MoralVerdict::Good => "Good",
                MoralVerdict::Bad => "Bad",
                MoralVerdict::Neutral => "Neutral",
                MoralVerdict::ConsentViolation => "Consent Violation",
            },
            parts.join("; ")
        )
    }
}

// ============================================================================
// Obligation Rule System Structures
// ============================================================================

/// A single moral obligation/rule
#[derive(Debug, Clone)]
pub struct ObligationRule {
    /// Name of the rule (e.g., "honesty")
    pub name: String,
    /// Description of the obligation
    pub description: String,
    /// HDC encoding of the rule
    pub rule_hv: ContinuousHV,
    /// Actions that violate this rule
    pub violation_actions: Vec<String>,
    /// Actions that satisfy this rule
    pub satisfaction_actions: Vec<String>,
    /// Perfect duty (must never be violated) vs imperfect duty (should follow when possible)
    pub is_perfect_duty: bool,
}

/// A set of moral obligations/rules
#[derive(Debug, Clone)]
pub struct ObligationRuleSet {
    /// The rules in this set
    pub rules: Vec<ObligationRule>,
}

/// A detected obligation violation
#[derive(Debug, Clone)]
pub struct ObligationViolation {
    /// Name of the violated rule
    pub rule_name: String,
    /// Description of the rule
    pub rule_description: String,
    /// The phrase that triggered the violation
    pub matched_phrase: String,
    /// Whether this is a perfect duty
    pub is_perfect_duty: bool,
    /// Severity of the violation (0.0 to 1.0)
    pub severity: f32,
}

/// A detected obligation satisfaction
#[derive(Debug, Clone)]
pub struct ObligationSatisfaction {
    /// Name of the satisfied rule
    pub rule_name: String,
    /// Description of the rule
    pub rule_description: String,
    /// The phrase that triggered the satisfaction
    pub matched_phrase: String,
    /// Whether this is a perfect duty
    pub is_perfect_duty: bool,
    /// Moral credit for satisfying this duty (0.0 to 1.0)
    pub moral_credit: f32,
}

/// Verdict from deontological analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeontologicalVerdict {
    /// Action is right - fulfills duties without violations
    RightDutyFulfilled,
    /// Action is wrong - violates a perfect duty (never acceptable)
    WrongPerfectDutyViolated,
    /// Action is wrong - violates an imperfect duty
    WrongImperfectDutyViolated,
    /// Action is neutral - no duties involved
    Neutral,
}

/// Result of deontological judgment
#[derive(Debug, Clone)]
pub struct DeontologicalJudgment {
    /// List of violated obligations
    pub violations: Vec<ObligationViolation>,
    /// List of satisfied obligations
    pub satisfactions: Vec<ObligationSatisfaction>,
    /// Overall moral score (-1.0 to 1.0)
    pub score: f32,
    /// Final verdict
    pub verdict: DeontologicalVerdict,
}

// ============================================================================
// Moral Dilemma Handling
// ============================================================================

/// Priority levels for different moral duties
/// Higher priority duties take precedence in conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DutyPriority {
    /// Highest: Prevent severe harm to others (life, bodily integrity)
    PreventSevereHarm = 5,
    /// High: Perfect duties (honesty, non-theft, promise-keeping)
    PerfectDuty = 4,
    /// Medium: Respect autonomy and consent
    RespectAutonomy = 3,
    /// Lower: Imperfect duties (beneficence, self-improvement)
    ImperfectDuty = 2,
    /// Lowest: Supererogatory acts (beyond the call of duty)
    Supererogatory = 1,
}

/// A detected moral dilemma where duties conflict
#[derive(Debug, Clone)]
pub struct MoralDilemma {
    /// The conflicting duties
    pub conflicting_duties: Vec<String>,
    /// Priority of each duty
    pub priorities: Vec<DutyPriority>,
    /// Which duty "wins" according to priority ordering
    pub resolution: Option<String>,
    /// Explanation of the resolution
    pub explanation: String,
    /// Whether this is a genuine tragic dilemma (no good option)
    pub is_tragic: bool,
}

impl MoralAlgebra {
    /// Get the priority for a given duty/rule name
    pub fn duty_priority(&self, rule_name: &str) -> DutyPriority {
        match rule_name {
            // Preventing harm is highest priority
            "non_harm" => DutyPriority::PreventSevereHarm,

            // Perfect duties
            "honesty" | "non_theft" | "promise_keeping" => DutyPriority::PerfectDuty,

            // Autonomy
            "respect_autonomy" => DutyPriority::RespectAutonomy,

            // Imperfect duties
            "beneficence" | "self_improvement" => DutyPriority::ImperfectDuty,

            // Default
            _ => DutyPriority::ImperfectDuty,
        }
    }

    /// Detect if a scenario contains a moral dilemma
    ///
    /// A dilemma occurs when:
    /// 1. Multiple duties are involved AND
    /// 2. Satisfying one would violate another
    pub fn detect_dilemma(&self, text: &str) -> Option<MoralDilemma> {
        let rules = self.standard_obligations();
        let violations = self.check_obligation_violations(text, &rules);
        let satisfactions = self.check_obligation_satisfactions(text, &rules);

        // Check for conflicts: same rule appears in both lists, or
        // satisfying one duty requires violating another
        let violation_rules: std::collections::HashSet<_> =
            violations.iter().map(|v| v.rule_name.clone()).collect();
        let satisfaction_rules: std::collections::HashSet<_> =
            satisfactions.iter().map(|s| s.rule_name.clone()).collect();

        // Direct conflict: same rule both satisfied and violated
        let direct_conflicts: Vec<_> = violation_rules
            .intersection(&satisfaction_rules)
            .cloned()
            .collect();

        // Cross-duty conflict: one satisfaction paired with unrelated violation
        let has_cross_conflict = !violations.is_empty() && !satisfactions.is_empty();

        if direct_conflicts.is_empty() && !has_cross_conflict {
            return None; // No dilemma
        }

        // Build the dilemma
        let mut all_duties: Vec<String> = violation_rules
            .union(&satisfaction_rules)
            .cloned()
            .collect();
        all_duties.sort();
        all_duties.dedup();

        let priorities: Vec<_> = all_duties.iter().map(|d| self.duty_priority(d)).collect();

        // Resolve by priority: highest priority duty wins
        let resolution =
            if let Some((idx, _)) = priorities.iter().enumerate().max_by_key(|(_, p)| *p) {
                Some(all_duties[idx].clone())
            } else {
                None
            };

        // Determine if tragic (both options lead to wrong)
        let is_tragic = violations.len() >= 2 && violations.iter().all(|v| v.is_perfect_duty);

        let explanation = if is_tragic {
            "Tragic dilemma: no action avoids moral wrong".to_string()
        } else if let Some(ref winner) = resolution {
            format!("Resolution: {winner} takes priority")
        } else {
            "Unresolved conflict".to_string()
        };

        Some(MoralDilemma {
            conflicting_duties: all_duties,
            priorities,
            resolution,
            explanation,
            is_tragic,
        })
    }

    /// Resolve a moral dilemma using priority ordering
    ///
    /// Returns the recommended action based on duty priorities:
    /// 1. Prevent severe harm > Perfect duties > Autonomy > Imperfect duties
    pub fn resolve_dilemma(&self, dilemma: &MoralDilemma) -> DilemmaResolution {
        if dilemma.is_tragic {
            return DilemmaResolution {
                recommended_duty: dilemma.resolution.clone(),
                reasoning: "Tragic choice - minimize harm where possible".to_string(),
                confidence: 0.3, // Low confidence for tragic cases
                alternative_considered: dilemma.conflicting_duties.clone(),
            };
        }

        // Find highest priority duty
        let max_priority = dilemma
            .priorities
            .iter()
            .max()
            .copied()
            .unwrap_or(DutyPriority::ImperfectDuty);

        let reasoning = match max_priority {
            DutyPriority::PreventSevereHarm => {
                "Preventing severe harm takes absolute priority".to_string()
            }
            DutyPriority::PerfectDuty => {
                "Perfect duties (honesty, promises) must not be violated".to_string()
            }
            DutyPriority::RespectAutonomy => {
                "Respect for autonomy outweighs lesser duties".to_string()
            }
            DutyPriority::ImperfectDuty => {
                "Imperfect duties should be fulfilled when possible".to_string()
            }
            DutyPriority::Supererogatory => {
                "Supererogatory acts are praiseworthy but not required".to_string()
            }
        };

        DilemmaResolution {
            recommended_duty: dilemma.resolution.clone(),
            reasoning,
            confidence: 0.7 + (max_priority as u8 as f32 * 0.05),
            alternative_considered: dilemma.conflicting_duties.clone(),
        }
    }
}

/// Result of resolving a moral dilemma
#[derive(Debug, Clone)]
pub struct DilemmaResolution {
    /// The duty/action that should take priority
    pub recommended_duty: Option<String>,
    /// Explanation of why this resolution was chosen
    pub reasoning: String,
    /// Confidence in the resolution (0.0 to 1.0)
    pub confidence: f32,
    /// Other duties that were considered
    pub alternative_considered: Vec<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitives_orthogonal() {
        let primitives = MoralPrimitives::default_dim();
        let max_sim = primitives.verify_orthogonality();

        // Random 4096-dim vectors should have similarity < 0.1
        assert!(
            max_sim < 0.15,
            "Primitives not orthogonal: max_sim = {}",
            max_sim
        );
    }

    #[test]
    fn test_intent_encoding_distinguishable() {
        let algebra = MoralAlgebra::default_dim();

        let good = algebra.encode_intent(MoralIntent::Good);
        let bad = algebra.encode_intent(MoralIntent::Bad);
        let neutral = algebra.encode_intent(MoralIntent::Neutral);

        // Different intents should be dissimilar
        let good_bad = good.similarity(&bad);
        let good_neutral = good.similarity(&neutral);
        let bad_neutral = bad.similarity(&neutral);

        assert!(good_bad < 0.3, "Good and Bad too similar: {}", good_bad);
        assert!(
            good_neutral < 0.3,
            "Good and Neutral too similar: {}",
            good_neutral
        );
        assert!(
            bad_neutral < 0.3,
            "Bad and Neutral too similar: {}",
            bad_neutral
        );
    }

    #[test]
    fn test_action_structure_composition() {
        let algebra = MoralAlgebra::default_dim();

        // Create two similar actions with different intents
        let help_good =
            algebra.encode_action_structure("Tyler", "help", "stranger", MoralIntent::Good);
        let help_bad =
            algebra.encode_action_structure("Tyler", "help", "stranger", MoralIntent::Bad);
        let harm_bad =
            algebra.encode_action_structure("Tyler", "harm", "stranger", MoralIntent::Bad);

        // Same action, different intent should be somewhat similar
        let help_sim = help_good.similarity(&help_bad);

        // Different action, same intent should be somewhat similar
        let intent_sim = help_bad.similarity(&harm_bad);

        // Both should be distinguishable (HDC cosine similarity can be slightly negative
        // for near-orthogonal vectors, so allow small negative values)
        assert!(
            help_sim > -0.2 && help_sim < 0.8,
            "Same action different intent: {}",
            help_sim
        );
        assert!(
            intent_sim > -0.2 && intent_sim < 0.8,
            "Different action same intent: {}",
            intent_sim
        );
    }

    #[test]
    fn test_consent_violation_detection() {
        // Test consent violation detection using direct state checking
        // (HDC similarity-based detection proved unreliable, so we use
        // the MoralParser's is_consent_violation() method instead)
        use crate::hdc::moral_parser::MoralParser;

        let algebra = MoralAlgebra::default_dim();
        let parser = MoralParser::new();

        // Scenario with consent violation (absent consent + patient)
        let violation = parser.parse_and_encode(
            "I discussed my daughter's health without asking first",
            &algebra,
        );
        assert!(
            violation.is_consent_violation(),
            "Should detect consent violation when consent is absent and patient exists"
        );

        // Scenario with consent given
        let with_consent = parser.parse_and_encode(
            "After asking my daughter, I discussed her health with the doctor",
            &algebra,
        );
        assert!(
            !with_consent.is_consent_violation(),
            "Should not detect violation when consent is given"
        );

        // Scenario without a patient (no consent issue)
        let no_patient = parser.parse_and_encode("I walked to the store", &algebra);
        assert!(
            !no_patient.is_consent_violation(),
            "Should not detect violation when no patient is affected"
        );
    }

    #[test]
    fn test_proportionality_justice() {
        let algebra = MoralAlgebra::default_dim();

        // Proportional: medium effort, medium reward
        let fair = algebra.encode_proportionality(
            "clean house",
            Magnitude::Medium,
            "fair wage",
            Magnitude::Medium,
        );

        // Disproportional: tiny effort, huge reward
        let unfair = algebra.encode_proportionality(
            "clean house",
            Magnitude::Tiny,
            "brand new car",
            Magnitude::Huge,
        );

        assert!(fair.is_proportional, "Fair case should be proportional");
        assert!(
            !unfair.is_proportional,
            "Unfair case should not be proportional"
        );

        // Judge them
        let fair_judgment = algebra.judge_proportionality(&fair);
        let unfair_judgment = algebra.judge_proportionality(&unfair);

        assert!(fair_judgment.is_just, "Fair case should be just");
        assert!(!unfair_judgment.is_just, "Unfair case should not be just");
    }

    #[test]
    fn test_excuse_validity() {
        let algebra = MoralAlgebra::default_dim();

        // Valid excuse: directly addresses obligation
        let valid = algebra.encode_excuse_validity(
            "prepare for meeting",
            "already set up conference room",
            true,
        );

        // Invalid excuse: doesn't address obligation
        let invalid =
            algebra.encode_excuse_validity("prepare for meeting", "not in the mood", false);

        assert!(valid.is_valid, "Valid excuse should be valid");
        assert!(!invalid.is_valid, "Invalid excuse should not be valid");

        // The HVs should be different
        let sim = valid.composed.similarity(&invalid.composed);
        assert!(sim < 0.5, "Valid and invalid excuses too similar: {}", sim);
    }

    #[test]
    fn test_moral_judgment() {
        let algebra = MoralAlgebra::default_dim();

        // Good action — compose using bind() to match prototype composition
        let good_action = {
            let agent = algebra.encode_agent("I");
            let action = algebra.encode_action("help");
            let patient = algebra.encode_patient("person");
            let intent = algebra.encode_intent(MoralIntent::Good);
            agent.bind(&action).bind(&patient).bind(&intent)
        };

        // Bad action — compose using bind() to match prototype composition
        let bad_action = {
            let agent = algebra.encode_agent("I");
            let action = algebra.encode_action("harm");
            let patient = algebra.encode_patient("victim");
            let intent = algebra.encode_intent(MoralIntent::Bad);
            agent.bind(&action).bind(&patient).bind(&intent)
        };

        let good_judgment = algebra.judge_action(&good_action);
        let bad_judgment = algebra.judge_action(&bad_action);

        // Good action should be judged as good
        assert!(
            good_judgment.good_similarity > good_judgment.bad_similarity,
            "Good action not recognized: good={}, bad={}",
            good_judgment.good_similarity,
            good_judgment.bad_similarity
        );

        // Bad action should be judged as bad
        assert!(
            bad_judgment.bad_similarity > bad_judgment.good_similarity,
            "Bad action not recognized: good={}, bad={}",
            bad_judgment.good_similarity,
            bad_judgment.bad_similarity
        );
    }

    #[test]
    fn test_negation_operator() {
        let algebra = MoralAlgebra::default_dim();

        let consent_given = algebra.encode_consent(ConsentState::Given);
        let consent_negated = algebra.negate(&consent_given);

        // Negation should create something different
        let sim = consent_given.similarity(&consent_negated);
        assert!(sim < 0.5, "Negation too similar to original: {}", sim);

        // Double negation should return to original
        let double_neg = algebra.negate(&consent_negated);
        // Note: HDC negation isn't perfect inversion, but should be more similar
        let double_sim = consent_given.similarity(&double_neg);
        assert!(
            double_sim > sim,
            "Double negation should be more similar: single={}, double={}",
            sim,
            double_sim
        );
    }

    #[test]
    fn test_deontological_judgment() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario with lying (perfect duty violation)
        let lying = algebra.judge_deontological("I lied to my friend about where I was");
        assert!(
            !lying.violations.is_empty(),
            "Should detect honesty violation"
        );
        assert_eq!(
            lying.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );

        // Scenario with helping (duty satisfaction)
        let helping = algebra.judge_deontological("I helped my neighbor carry groceries");
        assert!(
            !helping.satisfactions.is_empty(),
            "Should detect beneficence satisfaction"
        );
        assert_eq!(helping.verdict, DeontologicalVerdict::RightDutyFulfilled);

        // Neutral scenario
        let neutral = algebra.judge_deontological("I walked to the park");
        assert!(neutral.violations.is_empty());
        assert!(neutral.satisfactions.is_empty());
        assert_eq!(neutral.verdict, DeontologicalVerdict::Neutral);

        // Stealing (perfect duty violation)
        let stealing = algebra.judge_deontological("I stole money from the register");
        assert!(
            !stealing.violations.is_empty(),
            "Should detect theft violation"
        );
        assert_eq!(
            stealing.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );
    }

    #[test]
    fn test_obligation_rules() {
        let algebra = MoralAlgebra::default_dim();
        let rules = algebra.standard_obligations();

        // Should have both perfect and imperfect duties
        let perfect_count = rules.rules.iter().filter(|r| r.is_perfect_duty).count();
        let imperfect_count = rules.rules.iter().filter(|r| !r.is_perfect_duty).count();

        assert!(perfect_count >= 4, "Should have at least 4 perfect duties");
        assert!(
            imperfect_count >= 2,
            "Should have at least 2 imperfect duties"
        );
    }

    // ========================================================================
    // Additional coverage: constructors, edge cases, algebraic properties
    // ========================================================================

    #[test]
    fn test_moral_algebra_constructor_and_dim() {
        let algebra = MoralAlgebra::new(512);
        assert_eq!(algebra.dim(), 512);

        let default = MoralAlgebra::default_dim();
        assert_eq!(default.dim(), MORAL_DIM);
        assert_eq!(default.dim(), 4096);
    }

    #[test]
    fn test_moral_primitives_deterministic() {
        // Same seed should produce identical primitives
        let p1 = MoralPrimitives::new(1024);
        let p2 = MoralPrimitives::new(1024);

        let sim = p1.agent.similarity(&p2.agent);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same seed should produce identical agents, sim = {}",
            sim,
        );
    }

    #[test]
    fn test_moral_operators_deterministic() {
        let o1 = MoralOperators::new(1024);
        let o2 = MoralOperators::new(1024);

        let sim = o1.causes.similarity(&o2.causes);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same seed should produce identical operators, sim = {}",
            sim,
        );
    }

    #[test]
    fn test_magnitude_ordering_and_values() {
        assert!(Magnitude::Tiny < Magnitude::Small);
        assert!(Magnitude::Small < Magnitude::Medium);
        assert!(Magnitude::Medium < Magnitude::Large);
        assert!(Magnitude::Large < Magnitude::Huge);

        // Values should be monotonically increasing
        let magnitudes = [
            Magnitude::Tiny,
            Magnitude::Small,
            Magnitude::Medium,
            Magnitude::Large,
            Magnitude::Huge,
        ];
        for window in magnitudes.windows(2) {
            assert!(
                window[0].value() < window[1].value(),
                "{:?} value ({}) should be less than {:?} value ({})",
                window[0],
                window[0].value(),
                window[1],
                window[1].value(),
            );
        }
    }

    #[test]
    fn test_consent_states_distinguishable() {
        let algebra = MoralAlgebra::default_dim();

        let given = algebra.encode_consent(ConsentState::Given);
        let denied = algebra.encode_consent(ConsentState::Denied);
        let absent = algebra.encode_consent(ConsentState::Absent);
        let implied = algebra.encode_consent(ConsentState::Implied);

        // All pairs should be distinguishable
        let pairs = [
            (&given, &denied, "Given vs Denied"),
            (&given, &absent, "Given vs Absent"),
            (&given, &implied, "Given vs Implied"),
            (&denied, &absent, "Denied vs Absent"),
            (&denied, &implied, "Denied vs Implied"),
            (&absent, &implied, "Absent vs Implied"),
        ];

        for (a, b, label) in &pairs {
            let sim = a.similarity(b);
            assert!(sim < 0.5, "{} too similar: {}", label, sim,);
        }
    }

    #[test]
    fn test_encode_agent_deterministic_and_distinct() {
        let algebra = MoralAlgebra::default_dim();

        // Same name produces same encoding
        let a1 = algebra.encode_agent("Alice");
        let a2 = algebra.encode_agent("Alice");
        let sim = a1.similarity(&a2);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same agent name should produce identical HV, sim = {}",
            sim,
        );

        // Different names produce distinct encodings
        let bob = algebra.encode_agent("Bob");
        let sim_diff = a1.similarity(&bob);
        assert!(
            sim_diff < 0.5,
            "Different agents should be distinct, sim = {}",
            sim_diff,
        );
    }

    #[test]
    fn test_proportionality_boundary_cases() {
        let algebra = MoralAlgebra::default_dim();

        // Adjacent magnitudes (difference = 0.2) should be proportional
        let adjacent =
            algebra.encode_proportionality("task", Magnitude::Medium, "pay", Magnitude::Large);
        assert!(
            adjacent.is_proportional,
            "Adjacent magnitudes (diff=0.2) should be proportional",
        );

        // Same magnitude (difference = 0.0) should be proportional
        let same =
            algebra.encode_proportionality("task", Magnitude::Large, "pay", Magnitude::Large);
        assert!(
            same.is_proportional,
            "Same magnitudes should be proportional",
        );

        // Two-step gap (difference = 0.4) should NOT be proportional
        let gap = algebra.encode_proportionality("task", Magnitude::Tiny, "pay", Magnitude::Large);
        assert!(
            !gap.is_proportional,
            "Large magnitude gap should NOT be proportional",
        );
    }

    #[test]
    fn test_justice_judgment_similarity_finite() {
        let algebra = MoralAlgebra::default_dim();

        let prop =
            algebra.encode_proportionality("work", Magnitude::Medium, "reward", Magnitude::Medium);
        let judgment = algebra.judge_proportionality(&prop);

        assert!(judgment.fair_similarity.is_finite());
        assert!(judgment.unfair_similarity.is_finite());
        assert!(judgment.magnitude_difference.is_finite());
        assert_eq!(judgment.magnitude_difference, 0.0);
    }

    #[test]
    fn test_deontological_multiple_violations() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario with multiple violations
        let result = algebra.judge_deontological("I lied and then stole from my neighbor");
        assert!(
            result.violations.len() >= 2,
            "Should detect at least 2 violations, got {}",
            result.violations.len(),
        );
        assert_eq!(
            result.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );
        assert!(
            result.score < 0.0,
            "Score should be negative: {}",
            result.score
        );
    }

    #[test]
    fn test_deontological_mixed_satisfaction_and_violation() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario with both satisfaction and violation
        let result = algebra.judge_deontological("I helped my friend but lied about the cost");
        assert!(
            !result.violations.is_empty(),
            "Should detect at least one violation",
        );
        assert!(
            !result.satisfactions.is_empty(),
            "Should detect at least one satisfaction",
        );
        // Perfect duty violation dominates
        assert_eq!(
            result.verdict,
            DeontologicalVerdict::WrongPerfectDutyViolated
        );
    }

    #[test]
    fn test_deontological_empty_and_neutral_text() {
        let algebra = MoralAlgebra::default_dim();

        // Empty string
        let empty = algebra.judge_deontological("");
        assert!(empty.violations.is_empty());
        assert!(empty.satisfactions.is_empty());
        assert_eq!(empty.verdict, DeontologicalVerdict::Neutral);
        assert_eq!(empty.score, 0.0);

        // Neutral text without moral content
        let neutral = algebra.judge_deontological("the sky is blue today");
        assert_eq!(neutral.verdict, DeontologicalVerdict::Neutral);
    }

    #[test]
    fn test_ensemble_judgment_without_hdc() {
        let algebra = MoralAlgebra::default_dim();

        // No action HV, just intent and text
        let result = algebra.judge_ensemble(None, MoralIntent::Good, "I helped my neighbor");
        assert_eq!(result.hdc_verdict, None);
        assert_eq!(result.hdc_confidence, None);
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_ensemble_judgment_unanimity() {
        let algebra = MoralAlgebra::default_dim();

        // Clearly good action should produce unanimous verdict
        let result = algebra.judge_ensemble(
            None,
            MoralIntent::Good,
            "I helped my neighbor carry groceries",
        );
        // If all signals agree on Good:
        if result.final_verdict == MoralVerdict::Good {
            // With no HDC and no learned classifier, is_unanimous checks intent and deonto
            let expected_unanimous = result.intent_verdict == MoralVerdict::Good
                && result.deonto_verdict == MoralVerdict::Good;
            if expected_unanimous {
                assert!(result.is_unanimous());
            }
        }

        // Verify explanation is non-empty
        let explanation = result.explanation();
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_duty_priority_ordering() {
        let algebra = MoralAlgebra::default_dim();

        assert!(
            algebra.duty_priority("non_harm") > algebra.duty_priority("honesty"),
            "PreventSevereHarm should be higher than PerfectDuty",
        );
        assert!(
            algebra.duty_priority("honesty") > algebra.duty_priority("respect_autonomy"),
            "PerfectDuty should be higher than RespectAutonomy",
        );
        assert!(
            algebra.duty_priority("respect_autonomy") > algebra.duty_priority("beneficence"),
            "RespectAutonomy should be higher than ImperfectDuty",
        );
    }

    #[test]
    fn test_dilemma_detection_cross_conflict() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario: helping (beneficence) but lying (honesty violation)
        let dilemma = algebra.detect_dilemma("I lied to protect my friend from harm");
        assert!(
            dilemma.is_some(),
            "Should detect a dilemma when satisfying one duty requires violating another",
        );

        if let Some(d) = dilemma {
            assert!(!d.conflicting_duties.is_empty());
            assert!(d.resolution.is_some());
            assert!(!d.explanation.is_empty());
        }
    }

    #[test]
    fn test_dilemma_detection_no_conflict() {
        let algebra = MoralAlgebra::default_dim();

        // Neutral scenario has no dilemma
        let dilemma = algebra.detect_dilemma("I walked to the park");
        assert!(
            dilemma.is_none(),
            "Neutral scenario should not produce a dilemma",
        );
    }

    #[test]
    fn test_resolve_dilemma_tragic() {
        let algebra = MoralAlgebra::default_dim();

        let tragic_dilemma = MoralDilemma {
            conflicting_duties: vec!["non_harm".to_string(), "honesty".to_string()],
            priorities: vec![DutyPriority::PreventSevereHarm, DutyPriority::PerfectDuty],
            resolution: Some("non_harm".to_string()),
            explanation: "Tragic dilemma: no action avoids moral wrong".to_string(),
            is_tragic: true,
        };

        let resolution = algebra.resolve_dilemma(&tragic_dilemma);
        assert_eq!(
            resolution.confidence, 0.3,
            "Tragic cases should have low confidence"
        );
        assert!(resolution.reasoning.contains("minimize harm"));
    }

    #[test]
    fn test_has_learned_classifier_initially_false() {
        let algebra = MoralAlgebra::default_dim();
        assert!(!algebra.has_learned_classifier());
    }

    #[test]
    fn test_ahimsa_violations_detected() {
        let algebra = MoralAlgebra::default_dim();
        let result = algebra.judge_deontological("the regime decided to brutalize the prisoners");
        let violation_names: Vec<&str> = result
            .violations
            .iter()
            .map(|v| v.rule_name.as_str())
            .collect();
        assert!(
            violation_names.contains(&"ahimsa_nonviolence"),
            "Should detect ahimsa violation, got: {:?}",
            violation_names
        );
    }

    #[test]
    fn test_humility_satisfaction_detected() {
        let algebra = MoralAlgebra::default_dim();
        let result = algebra
            .judge_deontological("I might be wrong about this, let me consult expert opinion");
        let sat_names: Vec<&str> = result
            .satisfactions
            .iter()
            .map(|s| s.rule_name.as_str())
            .collect();
        assert!(
            sat_names.contains(&"epistemic_humility")
                || sat_names.contains(&"deference_to_expertise"),
            "Should detect humility/deference satisfaction, got: {:?}",
            sat_names
        );
    }

    #[test]
    fn test_satisfaction_severity_matches_violations() {
        let algebra = MoralAlgebra::default_dim();

        // Perfect duty: honesty violation vs satisfaction
        let violation_result = algebra.judge_deontological("I lied to my colleague");
        let satisfaction_result = algebra.judge_deontological("I was honest about the situation");

        let perfect_violation = violation_result
            .violations
            .iter()
            .find(|v| v.rule_name == "honesty")
            .expect("Should detect honesty violation");
        let perfect_satisfaction = satisfaction_result
            .satisfactions
            .iter()
            .find(|s| s.rule_name == "honesty")
            .expect("Should detect honesty satisfaction");

        assert!(
            perfect_violation.is_perfect_duty,
            "Honesty should be a perfect duty"
        );
        assert!(
            perfect_satisfaction.is_perfect_duty,
            "Honesty satisfaction should also be a perfect duty"
        );
        assert_eq!(
            perfect_violation.severity, perfect_satisfaction.moral_credit,
            "Perfect duty satisfaction credit ({}) should equal violation severity ({})",
            perfect_satisfaction.moral_credit, perfect_violation.severity,
        );

        // Imperfect duty: beneficence violation vs satisfaction
        let imperfect_violation = algebra.judge_deontological("I refused to help my neighbor");
        let imperfect_satisfaction =
            algebra.judge_deontological("I helped my neighbor carry groceries");

        let imp_v = imperfect_violation
            .violations
            .iter()
            .find(|v| v.rule_name == "beneficence")
            .expect("Should detect beneficence violation");
        let imp_s = imperfect_satisfaction
            .satisfactions
            .iter()
            .find(|s| s.rule_name == "beneficence")
            .expect("Should detect beneficence satisfaction");

        assert!(
            !imp_v.is_perfect_duty,
            "Beneficence should be an imperfect duty"
        );
        assert!(
            !imp_s.is_perfect_duty,
            "Beneficence satisfaction should also be an imperfect duty"
        );
        assert_eq!(
            imp_v.severity, imp_s.moral_credit,
            "Imperfect duty satisfaction credit ({}) should equal violation severity ({})",
            imp_s.moral_credit, imp_v.severity,
        );
    }
}
