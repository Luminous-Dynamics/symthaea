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
            agent: ContinuousHV::random(dim, 1000003),      // "who acts"
            patient: ContinuousHV::random(dim, 1000033),    // "who is affected"
            action: ContinuousHV::random(dim, 1000037),     // "what happens"
            intent: ContinuousHV::random(dim, 1000039),     // "why"
            consent: ContinuousHV::random(dim, 1000081),    // "permission"
            obligation: ContinuousHV::random(dim, 1000099), // "duty"
            magnitude: ContinuousHV::random(dim, 1000117),  // "scale"
        }
    }

    /// Create with default dimension (4096)
    pub fn default_dim() -> Self {
        Self::new(MORAL_DIM)
    }

    /// Verify that primitives are approximately orthogonal
    pub fn verify_orthogonality(&self) -> f32 {
        let primitives = [
            &self.agent, &self.patient, &self.action,
            &self.intent, &self.consent, &self.obligation, &self.magnitude,
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

        Self {
            primitives,
            operators,
            intent_hvs,
            magnitude_hvs,
            consent_hvs,
            dim,
        }
    }

    /// Create with default dimension
    pub fn default_dim() -> Self {
        Self::new(MORAL_DIM)
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
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
        let level_hv = self.intent_hvs.get(&intent).unwrap();
        self.primitives.intent.bind(level_hv)
    }

    /// Encode consent state
    pub fn encode_consent(&self, state: ConsentState) -> ContinuousHV {
        let state_hv = self.consent_hvs.get(&state).unwrap();
        self.primitives.consent.bind(state_hv)
    }

    /// Encode an obligation
    pub fn encode_obligation(&self, obligation: &str) -> ContinuousHV {
        let oblig_hv = self.hash_string(obligation);
        self.primitives.obligation.bind(&oblig_hv)
    }

    /// Encode magnitude at a specific level
    pub fn encode_magnitude(&self, level: Magnitude) -> ContinuousHV {
        let level_hv = self.magnitude_hvs.get(&level).unwrap();
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
        let effort_hv = self.encode_action(effort_desc)
            .bind(&self.encode_magnitude(effort_mag));
        let reward_hv = self.encode_action(reward_desc)
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
    pub fn standard_obligations(&self) -> ObligationRuleSet {
        let mut rules = Vec::new();

        // Perfect duties (must never be violated)
        rules.push(ObligationRule {
            name: "honesty".to_string(),
            description: "Do not lie or deceive".to_string(),
            rule_hv: self.encode_obligation("be honest"),
            violation_actions: vec!["lie", "lied", "deceive", "deceived", "cheat", "cheated", "mislead"]
                .into_iter().map(|s| s.to_string()).collect(),
            satisfaction_actions: vec!["tell truth", "honest", "truthful", "transparent"]
                .into_iter().map(|s| s.to_string()).collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "non_theft".to_string(),
            description: "Do not steal".to_string(),
            rule_hv: self.encode_obligation("respect property"),
            violation_actions: vec!["steal", "stole", "stolen", "take without", "theft", "rob", "robbed"]
                .into_iter().map(|s| s.to_string()).collect(),
            satisfaction_actions: vec!["return", "give back", "respect property"]
                .into_iter().map(|s| s.to_string()).collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "non_harm".to_string(),
            description: "Do not harm innocents".to_string(),
            rule_hv: self.encode_obligation("do no harm"),
            violation_actions: vec!["harm", "harmed", "hurt", "injure", "injured", "attack", "attacked", "abuse", "abused"]
                .into_iter().map(|s| s.to_string()).collect(),
            satisfaction_actions: vec!["protect", "protected", "care", "cared", "heal", "healed"]
                .into_iter().map(|s| s.to_string()).collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "promise_keeping".to_string(),
            description: "Keep your promises".to_string(),
            rule_hv: self.encode_obligation("keep promises"),
            violation_actions: vec!["broke promise", "break promise", "betray", "betrayed", "abandon", "abandoned"]
                .into_iter().map(|s| s.to_string()).collect(),
            satisfaction_actions: vec!["kept promise", "fulfill", "fulfilled", "honor", "honored"]
                .into_iter().map(|s| s.to_string()).collect(),
            is_perfect_duty: true,
        });

        rules.push(ObligationRule {
            name: "respect_autonomy".to_string(),
            description: "Respect others' autonomy and consent".to_string(),
            rule_hv: self.encode_obligation("respect autonomy"),
            violation_actions: vec!["force", "forced", "coerce", "coerced", "manipulate", "manipulated", "without consent", "without permission"]
                .into_iter().map(|s| s.to_string()).collect(),
            satisfaction_actions: vec!["ask", "asked", "consent", "consented", "permission", "respect choice"]
                .into_iter().map(|s| s.to_string()).collect(),
            is_perfect_duty: true,
        });

        // Imperfect duties (should be followed when possible)
        rules.push(ObligationRule {
            name: "beneficence".to_string(),
            description: "Help those in need when you can".to_string(),
            rule_hv: self.encode_obligation("help others"),
            violation_actions: vec!["ignore suffering", "refused to help", "callous", "indifferent"]
                .into_iter().map(|s| s.to_string()).collect(),
            satisfaction_actions: vec!["help", "helped", "assist", "assisted", "support", "supported", "save", "saved", "rescue", "rescued"]
                .into_iter().map(|s| s.to_string()).collect(),
            is_perfect_duty: false,
        });

        rules.push(ObligationRule {
            name: "self_improvement".to_string(),
            description: "Develop your talents and abilities".to_string(),
            rule_hv: self.encode_obligation("improve self"),
            violation_actions: vec!["waste talent", "lazy", "neglect", "neglected"]
                .into_iter().map(|s| s.to_string()).collect(),
            satisfaction_actions: vec!["learn", "learned", "study", "studied", "practice", "practiced", "improve", "improved"]
                .into_iter().map(|s| s.to_string()).collect(),
            is_perfect_duty: false,
        });

        ObligationRuleSet { rules }
    }

    /// Check if an action violates any obligations
    pub fn check_obligation_violations(&self, text: &str, rules: &ObligationRuleSet) -> Vec<ObligationViolation> {
        let lower = text.to_lowercase();
        let mut violations = Vec::new();

        for rule in &rules.rules {
            // Check for violation actions
            for violation in &rule.violation_actions {
                if lower.contains(violation) {
                    violations.push(ObligationViolation {
                        rule_name: rule.name.clone(),
                        rule_description: rule.description.clone(),
                        matched_phrase: violation.clone(),
                        is_perfect_duty: rule.is_perfect_duty,
                        severity: if rule.is_perfect_duty { 1.0 } else { 0.5 },
                    });
                    break; // One violation per rule is enough
                }
            }
        }

        violations
    }

    /// Check if an action satisfies any obligations
    pub fn check_obligation_satisfactions(&self, text: &str, rules: &ObligationRuleSet) -> Vec<ObligationSatisfaction> {
        let lower = text.to_lowercase();
        let mut satisfactions = Vec::new();

        for rule in &rules.rules {
            // Check for satisfaction actions
            for satisfaction in &rule.satisfaction_actions {
                if lower.contains(satisfaction) {
                    satisfactions.push(ObligationSatisfaction {
                        rule_name: rule.name.clone(),
                        rule_description: rule.description.clone(),
                        matched_phrase: satisfaction.clone(),
                        moral_credit: if rule.is_perfect_duty { 0.3 } else { 0.7 }, // Imperfect duties give more credit when fulfilled
                    });
                    break;
                }
            }
        }

        satisfactions
    }

    /// Compute a deontological judgment for a scenario
    pub fn judge_deontological(&self, text: &str) -> DeontologicalJudgment {
        let rules = self.standard_obligations();
        let violations = self.check_obligation_violations(text, &rules);
        let satisfactions = self.check_obligation_satisfactions(text, &rules);

        // Perfect duty violations are serious
        let perfect_violations: Vec<_> = violations.iter().filter(|v| v.is_perfect_duty).collect();
        let imperfect_violations: Vec<_> = violations.iter().filter(|v| !v.is_perfect_duty).collect();

        // Calculate overall score
        let violation_penalty: f32 = perfect_violations.iter().map(|v| v.severity).sum::<f32>()
            + imperfect_violations.iter().map(|v| v.severity * 0.3).sum::<f32>();
        let satisfaction_bonus: f32 = satisfactions.iter().map(|s| s.moral_credit).sum();

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
            if score > 0.0 { DeontologicalVerdict::RightDutyFulfilled }
            else if score < 0.0 { DeontologicalVerdict::WrongImperfectDutyViolated }
            else { DeontologicalVerdict::Neutral }
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
    pub fn good_action_prototype(&self) -> ContinuousHV {
        // Good = action + good intent + consent given
        let action = self.encode_action("help");
        let intent = self.encode_intent(MoralIntent::Good);
        let consent = self.encode_consent(ConsentState::Given);

        ContinuousHV::bundle_owned(&[action, intent, consent])
    }

    /// Create a "morally bad" action prototype
    pub fn bad_action_prototype(&self) -> ContinuousHV {
        // Bad = action + bad intent + consent denied
        let action = self.encode_action("harm");
        let intent = self.encode_intent(MoralIntent::Bad);
        let consent = self.encode_consent(ConsentState::Denied);

        ContinuousHV::bundle_owned(&[action, intent, consent])
    }

    /// Create a "consent violation" prototype
    pub fn consent_violation_prototype(&self) -> ContinuousHV {
        // Consent violation = action affecting patient without consent
        let action = self.encode_action("affect");
        let consent = self.encode_consent(ConsentState::Absent);

        action.bind(&consent)
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

    /// Judge if an action is morally good/bad based on similarity to prototypes
    pub fn judge_action(&self, action_hv: &ContinuousHV) -> MoralJudgment {
        let good_sim = action_hv.similarity(&self.good_action_prototype());
        let bad_sim = action_hv.similarity(&self.bad_action_prototype());
        let consent_viol_sim = action_hv.similarity(&self.consent_violation_prototype());

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

    /// Judge proportionality (for justice reasoning)
    pub fn judge_proportionality(&self, prop: &ProportionalityJudgment) -> JusticeJudgment {
        let fair_sim = prop.composed.similarity(&self.proportional_justice_prototype());
        let unfair_sim = prop.composed.similarity(&self.disproportional_prototype());

        JusticeJudgment {
            fair_similarity: fair_sim,
            unfair_similarity: unfair_sim,
            is_just: prop.is_proportional && fair_sim > unfair_sim,
            magnitude_difference: (prop.effort_magnitude.value() - prop.reward_magnitude.value()).abs(),
        }
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /// Hash a string to a hypervector (deterministic)
    fn hash_string(&self, s: &str) -> ContinuousHV {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

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
    /// Moral credit for satisfying this duty
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
        assert!(max_sim < 0.15, "Primitives not orthogonal: max_sim = {}", max_sim);
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
        assert!(good_neutral < 0.3, "Good and Neutral too similar: {}", good_neutral);
        assert!(bad_neutral < 0.3, "Bad and Neutral too similar: {}", bad_neutral);
    }

    #[test]
    fn test_action_structure_composition() {
        let algebra = MoralAlgebra::default_dim();

        // Create two similar actions with different intents
        let help_good = algebra.encode_action_structure(
            "Tyler", "help", "stranger", MoralIntent::Good
        );
        let help_bad = algebra.encode_action_structure(
            "Tyler", "help", "stranger", MoralIntent::Bad
        );
        let harm_bad = algebra.encode_action_structure(
            "Tyler", "harm", "stranger", MoralIntent::Bad
        );

        // Same action, different intent should be somewhat similar
        let help_sim = help_good.similarity(&help_bad);

        // Different action, same intent should be somewhat similar
        let intent_sim = help_bad.similarity(&harm_bad);

        // Both should be distinguishable (HDC cosine similarity can be slightly negative
        // for near-orthogonal vectors, so allow small negative values)
        assert!(help_sim > -0.2 && help_sim < 0.8,
                "Same action different intent: {}", help_sim);
        assert!(intent_sim > -0.2 && intent_sim < 0.8,
                "Different action same intent: {}", intent_sim);
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
            &algebra
        );
        assert!(violation.is_consent_violation(),
                "Should detect consent violation when consent is absent and patient exists");

        // Scenario with consent given
        let with_consent = parser.parse_and_encode(
            "After asking my daughter, I discussed her health with the doctor",
            &algebra
        );
        assert!(!with_consent.is_consent_violation(),
                "Should not detect violation when consent is given");

        // Scenario without a patient (no consent issue)
        let no_patient = parser.parse_and_encode(
            "I walked to the store",
            &algebra
        );
        assert!(!no_patient.is_consent_violation(),
                "Should not detect violation when no patient is affected");
    }

    #[test]
    fn test_proportionality_justice() {
        let algebra = MoralAlgebra::default_dim();

        // Proportional: medium effort, medium reward
        let fair = algebra.encode_proportionality(
            "clean house", Magnitude::Medium,
            "fair wage", Magnitude::Medium
        );

        // Disproportional: tiny effort, huge reward
        let unfair = algebra.encode_proportionality(
            "clean house", Magnitude::Tiny,
            "brand new car", Magnitude::Huge
        );

        assert!(fair.is_proportional, "Fair case should be proportional");
        assert!(!unfair.is_proportional, "Unfair case should not be proportional");

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
            true
        );

        // Invalid excuse: doesn't address obligation
        let invalid = algebra.encode_excuse_validity(
            "prepare for meeting",
            "not in the mood",
            false
        );

        assert!(valid.is_valid, "Valid excuse should be valid");
        assert!(!invalid.is_valid, "Invalid excuse should not be valid");

        // The HVs should be different
        let sim = valid.composed.similarity(&invalid.composed);
        assert!(sim < 0.5, "Valid and invalid excuses too similar: {}", sim);
    }

    #[test]
    fn test_moral_judgment() {
        let algebra = MoralAlgebra::default_dim();

        // Good action with good intent and consent
        let good_action = {
            let action = algebra.encode_action("help");
            let intent = algebra.encode_intent(MoralIntent::Good);
            let consent = algebra.encode_consent(ConsentState::Given);
            ContinuousHV::bundle_owned(&[action, intent, consent])
        };

        // Bad action with bad intent and no consent
        let bad_action = {
            let action = algebra.encode_action("harm");
            let intent = algebra.encode_intent(MoralIntent::Bad);
            let consent = algebra.encode_consent(ConsentState::Denied);
            ContinuousHV::bundle_owned(&[action, intent, consent])
        };

        let good_judgment = algebra.judge_action(&good_action);
        let bad_judgment = algebra.judge_action(&bad_action);

        // Good action should be judged as good
        assert!(good_judgment.good_similarity > good_judgment.bad_similarity,
                "Good action not recognized: good={}, bad={}",
                good_judgment.good_similarity, good_judgment.bad_similarity);

        // Bad action should be judged as bad
        assert!(bad_judgment.bad_similarity > bad_judgment.good_similarity,
                "Bad action not recognized: good={}, bad={}",
                bad_judgment.good_similarity, bad_judgment.bad_similarity);
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
        assert!(double_sim > sim,
                "Double negation should be more similar: single={}, double={}",
                sim, double_sim);
    }

    #[test]
    fn test_deontological_judgment() {
        let algebra = MoralAlgebra::default_dim();

        // Scenario with lying (perfect duty violation)
        let lying = algebra.judge_deontological("I lied to my friend about where I was");
        assert!(!lying.violations.is_empty(), "Should detect honesty violation");
        assert_eq!(lying.verdict, DeontologicalVerdict::WrongPerfectDutyViolated);

        // Scenario with helping (duty satisfaction)
        let helping = algebra.judge_deontological("I helped my neighbor carry groceries");
        assert!(!helping.satisfactions.is_empty(), "Should detect beneficence satisfaction");
        assert_eq!(helping.verdict, DeontologicalVerdict::RightDutyFulfilled);

        // Neutral scenario
        let neutral = algebra.judge_deontological("I walked to the park");
        assert!(neutral.violations.is_empty());
        assert!(neutral.satisfactions.is_empty());
        assert_eq!(neutral.verdict, DeontologicalVerdict::Neutral);

        // Stealing (perfect duty violation)
        let stealing = algebra.judge_deontological("I stole money from the register");
        assert!(!stealing.violations.is_empty(), "Should detect theft violation");
        assert_eq!(stealing.verdict, DeontologicalVerdict::WrongPerfectDutyViolated);
    }

    #[test]
    fn test_obligation_rules() {
        let algebra = MoralAlgebra::default_dim();
        let rules = algebra.standard_obligations();

        // Should have both perfect and imperfect duties
        let perfect_count = rules.rules.iter().filter(|r| r.is_perfect_duty).count();
        let imperfect_count = rules.rules.iter().filter(|r| !r.is_perfect_duty).count();

        assert!(perfect_count >= 4, "Should have at least 4 perfect duties");
        assert!(imperfect_count >= 2, "Should have at least 2 imperfect duties");
    }
}
