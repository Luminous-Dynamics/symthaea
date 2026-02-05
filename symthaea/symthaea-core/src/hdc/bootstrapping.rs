//! # Cognitive Bootstrapping - Connecting Ontological Primitives to Reasoning
//!
//! This module bridges the 186+ ontological primitives in `primitive_system.rs`
//! with the TaskState domain for MMLU-style semantic reasoning.
//!
//! ## Purpose
//!
//! For the Φ-Accuracy correlation experiment (Phase 2 validation), we need to:
//! 1. Load relevant primitives for a given reasoning task
//! 2. Compose primitives into reasoning chains
//! 3. Measure Φ during reasoning to correlate with accuracy
//!
//! ## Architecture
//!
//! ```text
//! MMLU Question → TaskState → PrimitiveBootstrapper → Reasoning Chain → Answer
//!                                     ↓
//!                           ConsciousnessGraph (Φ measurement)
//! ```

use crate::hdc::binary_hv::HV16;
use crate::hdc::primitive_system::PrimitiveSystem;
use std::collections::HashMap;

/// Categories of reasoning required for MMLU-style questions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningCategory {
    /// Logical deduction (AND, OR, IMPLIES, NOT)
    LogicalDeduction,
    /// Mathematical reasoning (arithmetic, set theory)
    MathematicalReasoning,
    /// Causal inference (CAUSES, ENABLES, PREVENTS)
    CausalInference,
    /// Analogical reasoning (SIMILAR_TO, IS_A, PART_OF)
    AnalogicalReasoning,
    /// Temporal reasoning (BEFORE, AFTER, DURING)
    TemporalReasoning,
    /// Modal reasoning (POSSIBLE, NECESSARY, OBLIGATORY)
    ModalReasoning,
    /// Quantitative reasoning (ALL, SOME, NONE, MOST)
    QuantitativeReasoning,
    /// Theory of mind (KNOWS, BELIEVES, WANTS, INTENDS)
    TheoryOfMind,
    /// Scientific reasoning (hypotheses, evidence, causality)
    ScientificReasoning,
    /// Ethical reasoning (GOOD, BAD, OBLIGATORY, PERMITTED)
    EthicalReasoning,
}

/// Maps MMLU subjects to reasoning categories
impl ReasoningCategory {
    /// Get reasoning categories relevant to an MMLU subject
    pub fn for_subject(subject: &str) -> Vec<Self> {
        let subject_lower = subject.to_lowercase();

        match subject_lower.as_str() {
            // STEM subjects
            s if s.contains("math") || s.contains("algebra") || s.contains("geometry") => {
                vec![Self::MathematicalReasoning, Self::LogicalDeduction]
            }
            s if s.contains("physics") || s.contains("chemistry") => {
                vec![Self::ScientificReasoning, Self::CausalInference, Self::MathematicalReasoning]
            }
            s if s.contains("biology") => {
                vec![Self::ScientificReasoning, Self::CausalInference, Self::AnalogicalReasoning]
            }
            s if s.contains("computer") || s.contains("programming") => {
                vec![Self::LogicalDeduction, Self::MathematicalReasoning]
            }

            // Humanities
            s if s.contains("philosophy") || s.contains("logic") => {
                vec![Self::LogicalDeduction, Self::ModalReasoning, Self::EthicalReasoning]
            }
            s if s.contains("history") => {
                vec![Self::TemporalReasoning, Self::CausalInference]
            }
            s if s.contains("psychology") => {
                vec![Self::TheoryOfMind, Self::CausalInference, Self::AnalogicalReasoning]
            }
            s if s.contains("ethics") || s.contains("moral") => {
                vec![Self::EthicalReasoning, Self::ModalReasoning, Self::TheoryOfMind]
            }

            // Social sciences
            s if s.contains("economics") => {
                vec![Self::MathematicalReasoning, Self::CausalInference, Self::QuantitativeReasoning]
            }
            s if s.contains("sociology") || s.contains("political") => {
                vec![Self::CausalInference, Self::QuantitativeReasoning, Self::TheoryOfMind]
            }
            s if s.contains("law") => {
                vec![Self::LogicalDeduction, Self::ModalReasoning, Self::EthicalReasoning]
            }

            // Default: general reasoning
            _ => vec![Self::LogicalDeduction, Self::CausalInference, Self::AnalogicalReasoning]
        }
    }
}

/// Bootstrapper that loads relevant primitives for reasoning tasks
pub struct PrimitiveBootstrapper {
    /// Reference to the global primitive system
    system: &'static PrimitiveSystem,
    /// Cached primitive HVs by category
    category_cache: HashMap<ReasoningCategory, Vec<(String, HV16)>>,
}

impl PrimitiveBootstrapper {
    /// Create a new bootstrapper using the global primitive system
    pub fn new() -> Self {
        let system = PrimitiveSystem::global();
        let mut bootstrapper = Self {
            system,
            category_cache: HashMap::new(),
        };
        bootstrapper.build_category_cache();
        bootstrapper
    }

    /// Build cache of primitives organized by reasoning category
    fn build_category_cache(&mut self) {
        // Logical deduction primitives
        self.category_cache.insert(ReasoningCategory::LogicalDeduction, vec![
            self.get_primitive("AND"),
            self.get_primitive("OR"),
            self.get_primitive("NOT"),
            self.get_primitive("IMPLIES"),
            self.get_primitive("IFF"),
            self.get_primitive("TRUE"),
            self.get_primitive("FALSE"),
            self.get_primitive("EQUALS"),
        ].into_iter().flatten().collect());

        // Mathematical reasoning primitives
        self.category_cache.insert(ReasoningCategory::MathematicalReasoning, vec![
            self.get_primitive("ZERO"),
            self.get_primitive("ONE"),
            self.get_primitive("SUCCESSOR"),
            self.get_primitive("ADDITION"),
            self.get_primitive("MULTIPLICATION"),
            self.get_primitive("SET"),
            self.get_primitive("MEMBERSHIP"),
            self.get_primitive("UNION"),
            self.get_primitive("INTERSECTION"),
            self.get_primitive("RATIO"),
            self.get_primitive("PROBABILITY"),
        ].into_iter().flatten().collect());

        // Causal inference primitives
        self.category_cache.insert(ReasoningCategory::CausalInference, vec![
            self.get_primitive("CAUSES"),
            self.get_primitive("ENABLES"),
            self.get_primitive("PREVENTS"),
            self.get_primitive("STATE_CHANGE"),
            self.get_primitive("EFFECT"),
            self.get_primitive("FORCE"),
            self.get_primitive("IF_THEN"),
        ].into_iter().flatten().collect());

        // Analogical reasoning primitives
        self.category_cache.insert(ReasoningCategory::AnalogicalReasoning, vec![
            self.get_primitive("IS_A"),
            self.get_primitive("PART_OF"),
            self.get_primitive("SIMILAR_TO"),
            self.get_primitive("ABOVE"),
            self.get_primitive("BELOW"),
            self.get_primitive("NEAR"),
            self.get_primitive("INSIDE"),
        ].into_iter().flatten().collect());

        // Temporal reasoning primitives
        self.category_cache.insert(ReasoningCategory::TemporalReasoning, vec![
            self.get_primitive("BEFORE"),
            self.get_primitive("AFTER"),
            self.get_primitive("DURING"),
            self.get_primitive("BEGINS"),
            self.get_primitive("ENDS"),
            self.get_primitive("TIME"),
            self.get_primitive("SEQUENCE"),
        ].into_iter().flatten().collect());

        // Modal reasoning primitives
        self.category_cache.insert(ReasoningCategory::ModalReasoning, vec![
            self.get_primitive("POSSIBLE"),
            self.get_primitive("NECESSARY"),
            self.get_primitive("MAYBE"),
            self.get_primitive("OBLIGATORY"),
            self.get_primitive("PERMITTED"),
            self.get_primitive("FORBIDDEN"),
        ].into_iter().flatten().collect());

        // Quantitative reasoning primitives
        self.category_cache.insert(ReasoningCategory::QuantitativeReasoning, vec![
            self.get_primitive("ALL"),
            self.get_primitive("SOME"),
            self.get_primitive("NONE"),
            self.get_primitive("MOST"),
            self.get_primitive("FEW"),
            self.get_primitive("MORE"),
            self.get_primitive("LESS"),
        ].into_iter().flatten().collect());

        // Theory of mind primitives
        self.category_cache.insert(ReasoningCategory::TheoryOfMind, vec![
            self.get_primitive("KNOW"),
            self.get_primitive("BELIEVE"),
            self.get_primitive("WANT"),
            self.get_primitive("INTEND"),
            self.get_primitive("FEEL"),
            self.get_primitive("THINK"),
            self.get_primitive("SELF"),
        ].into_iter().flatten().collect());

        // Scientific reasoning primitives
        self.category_cache.insert(ReasoningCategory::ScientificReasoning, vec![
            self.get_primitive("CAUSES"),
            self.get_primitive("EVIDENCE"),
            self.get_primitive("HYPOTHESIS"),
            self.get_primitive("OBSERVE"),
            self.get_primitive("PREDICT"),
            self.get_primitive("ENTROPY"),
            self.get_primitive("CONSERVATION"),
        ].into_iter().flatten().collect());

        // Ethical reasoning primitives
        self.category_cache.insert(ReasoningCategory::EthicalReasoning, vec![
            self.get_primitive("GOOD"),
            self.get_primitive("BAD"),
            self.get_primitive("OBLIGATORY"),
            self.get_primitive("PERMITTED"),
            self.get_primitive("FORBIDDEN"),
            self.get_primitive("FAIR"),
            self.get_primitive("HARM"),
        ].into_iter().flatten().collect());
    }

    /// Get primitive by name as (name, hv) tuple
    fn get_primitive(&self, name: &str) -> Option<(String, HV16)> {
        self.system.get(name).map(|p| (p.name.clone(), p.encoding.clone()))
    }

    /// Get primitives for a reasoning category
    pub fn primitives_for_category(&self, category: ReasoningCategory) -> &[(String, HV16)] {
        self.category_cache.get(&category).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all primitives relevant to an MMLU subject
    pub fn primitives_for_subject(&self, subject: &str) -> Vec<(String, HV16)> {
        let categories = ReasoningCategory::for_subject(subject);
        let mut primitives = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for category in categories {
            for (name, hv) in self.primitives_for_category(category) {
                if !seen.contains(name) {
                    seen.insert(name.clone());
                    primitives.push((name.clone(), hv.clone()));
                }
            }
        }

        primitives
    }

    /// Create an initial working memory HV for a reasoning task
    /// by bundling relevant primitives
    pub fn bootstrap_working_memory(&self, subject: &str, question_hv: &HV16) -> HV16 {
        let primitives = self.primitives_for_subject(subject);

        if primitives.is_empty() {
            return question_hv.clone();
        }

        // Bundle question with relevant primitive encodings
        let mut hvs: Vec<HV16> = vec![question_hv.clone()];
        for (_, hv) in primitives.iter().take(10) { // Limit to top 10 most relevant
            hvs.push(hv.clone());
        }

        HV16::bundle(&hvs)
    }

    /// Get the total number of primitives available
    pub fn total_primitives(&self) -> usize {
        self.system.count()
    }

    /// Get a summary of the primitive ecology
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Total primitives: {}\n", self.total_primitives()));
        s.push_str(&format!("Reasoning categories: {}\n\n", self.category_cache.len()));

        for (category, primitives) in &self.category_cache {
            s.push_str(&format!("{:?}: {} primitives\n", category, primitives.len()));
        }

        s
    }
}

impl Default for PrimitiveBootstrapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrapper_creation() {
        let bootstrapper = PrimitiveBootstrapper::new();
        assert!(bootstrapper.total_primitives() >= 180, "Should have 180+ primitives");
    }

    #[test]
    fn test_category_mapping() {
        let categories = ReasoningCategory::for_subject("philosophy");
        assert!(categories.contains(&ReasoningCategory::LogicalDeduction));
        assert!(categories.contains(&ReasoningCategory::EthicalReasoning));

        let categories = ReasoningCategory::for_subject("mathematics");
        assert!(categories.contains(&ReasoningCategory::MathematicalReasoning));
    }

    #[test]
    fn test_primitives_for_category() {
        let bootstrapper = PrimitiveBootstrapper::new();

        let logical = bootstrapper.primitives_for_category(ReasoningCategory::LogicalDeduction);
        assert!(!logical.is_empty(), "Should have logical primitives");

        // Check that we have the core logical operators
        let names: Vec<_> = logical.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"AND"), "Should have AND primitive");
        assert!(names.contains(&"OR"), "Should have OR primitive");
        assert!(names.contains(&"IMPLIES"), "Should have IMPLIES primitive");
    }

    #[test]
    fn test_primitives_for_subject() {
        let bootstrapper = PrimitiveBootstrapper::new();

        let physics_prims = bootstrapper.primitives_for_subject("physics");
        assert!(!physics_prims.is_empty(), "Should have physics primitives");

        // Physics should include causal and mathematical primitives
        let names: Vec<_> = physics_prims.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"CAUSES") || names.contains(&"PROBABILITY"),
            "Should have relevant physics primitives");
    }

    #[test]
    fn test_bootstrap_working_memory() {
        let bootstrapper = PrimitiveBootstrapper::new();
        let question_hv = HV16::random(42);

        let bootstrapped = bootstrapper.bootstrap_working_memory("mathematics", &question_hv);

        // Bootstrapped HV should be different from original (bundled with primitives)
        assert_ne!(bootstrapped.popcount(), question_hv.popcount() + 0, // Allow some variance
            "Bootstrapped HV should differ from original");
    }

    #[test]
    fn test_summary() {
        let bootstrapper = PrimitiveBootstrapper::new();
        let summary = bootstrapper.summary();

        assert!(summary.contains("Total primitives:"));
        assert!(summary.contains("LogicalDeduction"));
        assert!(summary.contains("MathematicalReasoning"));
    }
}
