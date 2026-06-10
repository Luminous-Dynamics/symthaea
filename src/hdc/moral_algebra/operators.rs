// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core `MoralAlgebra` struct and first impl block (encoding, composition,
//! obligation checking, deontological judgment, prototypes, ensemble judgment).

use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

use super::judgment::*;
use super::primitives::*;

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
    learned_classifier: Option<super::super::moral_prototypes::MoralPrototypeClassifier>,

    /// Optional Spinozist classifier (77.2% on Social Chemistry with 18D affect interpretability).
    /// Arc-wrapped: SpinozistClassifier contains ExemplarStore (5000 HVs), too expensive to clone.
    spinozist: Option<std::sync::Arc<super::super::spinozist_geometry::SpinozistClassifier>>,

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
            spinozist: None,
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

    /// Set the Spinozist classifier (5th ensemble signal).
    pub fn set_spinozist(&mut self, s: super::super::spinozist_geometry::SpinozistClassifier) {
        self.spinozist = Some(std::sync::Arc::new(s));
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Set a learned moral prototype classifier for the 4th ensemble signal.
    pub fn set_learned_classifier(
        &mut self,
        c: super::super::moral_prototypes::MoralPrototypeClassifier,
    ) {
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
                    super::super::moral_prototypes::MoralLabel::Good => MoralVerdict::Good,
                    super::super::moral_prototypes::MoralLabel::Neutral => MoralVerdict::Neutral,
                    super::super::moral_prototypes::MoralLabel::Bad => MoralVerdict::Bad,
                };
                (Some(verdict), Some(conf))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let has_learned = learned_verdict.is_some();

        // 5. Spinozist signal (if classifier available)
        let (spinozist_verdict, spinozist_confidence) = if let Some(ref spin) = self.spinozist {
            let (v, c) = spin.classify(text);
            (Some(v), Some(c))
        } else {
            (None, None)
        };
        let has_spinozist = spinozist_verdict.is_some();

        // Voting: per-category weight tuning (5 signals)
        // Different ETHICS categories benefit from different signal balance.
        let (w_hdc, w_intent, w_deonto, w_learned, w_spinozist) = if has_spinozist && has_learned {
            match category_hint {
                Some("commonsense") => (0.10, 0.30, 0.15, 0.25, 0.20),
                Some("justice") => (0.10, 0.15, 0.25, 0.25, 0.25),
                Some("deontology") => (0.10, 0.15, 0.25, 0.25, 0.25),
                Some("virtue") => (0.20, 0.30, 0.20, 0.00, 0.30),
                _ => (0.15, 0.25, 0.20, 0.10, 0.30),
            }
        } else if has_spinozist {
            match category_hint {
                Some("virtue") => (0.20, 0.30, 0.20, 0.00, 0.30),
                _ => (0.20, 0.30, 0.20, 0.00, 0.30),
            }
        } else if has_learned {
            match category_hint {
                Some("commonsense") => (0.15, 0.35, 0.15, 0.35, 0.0),
                Some("justice") => (0.15, 0.20, 0.30, 0.35, 0.0),
                Some("deontology") => (0.15, 0.20, 0.30, 0.35, 0.0),
                Some("virtue") => (0.3, 0.4, 0.3, 0.0, 0.0),
                _ => (0.25, 0.35, 0.30, 0.10, 0.0),
            }
        } else {
            (0.3, 0.4, 0.3, 0.0, 0.0)
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

        // Spinozist vote (confidence-weighted)
        if let (Some(sv), Some(sc)) = (&spinozist_verdict, &spinozist_confidence) {
            let spin_key = match sv {
                MoralVerdict::Good => "good",
                MoralVerdict::Bad => "bad",
                MoralVerdict::Neutral => "neutral",
                MoralVerdict::ConsentViolation => "consent_violation",
            };
            *votes.entry(spin_key).or_insert(0.0) += w_spinozist * (1.0 + sc.min(0.5));
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
            spinozist_verdict,
            spinozist_confidence,
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
