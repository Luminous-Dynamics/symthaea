// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Judgment structs, obligation types, dilemma handling, and the second
//! `impl MoralAlgebra` block (dilemma resolution methods).

use symthaea_core::hdc::ContinuousHV;

use super::operators::MoralAlgebra;
use super::primitives::*;

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
    /// Spinozist classifier verdict (if available)
    pub spinozist_verdict: Option<MoralVerdict>,
    /// Spinozist confidence
    pub spinozist_confidence: Option<f32>,
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
        let spinozist_matches = self
            .spinozist_verdict
            .map(|v| v == self.final_verdict)
            .unwrap_or(true);
        intent_matches && deonto_matches && hdc_matches && learned_matches && spinozist_matches
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
