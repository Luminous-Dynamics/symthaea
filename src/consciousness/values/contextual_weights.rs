// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Contextual Harmony Weighting
//!
//! Adjusts harmony importance based on action context, solving the limitation
//! where all harmonies were weighted equally regardless of situation.
//!
//! ## The Problem
//!
//! Without contextual weighting:
//! - Voting on governance: same weight for Infinite Play as Integral Wisdom
//! - Financial action: same weight for creativity as fairness
//! - Emergency action: same slow deliberation as routine action
//!
//! ## The Solution
//!
//! With contextual weighting:
//! - Voting: Higher weight on truth (Integral Wisdom), fairness (Sacred Reciprocity)
//! - Financial: Higher weight on fairness, do-no-harm (Pan-Sentient Flourishing)
//! - Creative: Higher weight on play (Infinite Play), coherence (Resonant Coherence)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │              CONTEXTUAL WEIGHTING PIPELINE                               │
//! │                                                                          │
//! │  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐    │
//! │  │   Action       │  →   │   Context      │  →   │   Weighted     │    │
//! │  │   Context      │      │   Analysis     │      │   Harmonies    │    │
//! │  │   (type,domain)│      │   + Profile    │      │   Evaluation   │    │
//! │  └────────────────┘      └────────────────┘      └────────────────┘    │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use super::eight_harmonies::Harmony;
use std::collections::HashMap;

/// Type of action being evaluated.
///
/// Canonical action type enum, re-exported by `unified_value_evaluator::types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ActionType {
    /// Basic action (default threshold)
    #[default]
    Basic,
    /// Governance proposal (higher threshold)
    Governance,
    /// Voting on proposal (higher threshold)
    Voting,
    /// Constitutional change (highest threshold)
    Constitutional,
}

/// Domain classification for actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ActionDomain {
    /// General/unclassified action
    #[default]
    General,
    /// Financial transactions, resource allocation
    Financial,
    /// Creative work, artistic expression
    Creative,
    /// Social interactions, relationships
    Social,
    /// Technical/system operations
    Technical,
    /// Educational, knowledge-sharing
    Educational,
    /// Healthcare, wellbeing
    Healthcare,
    /// Environmental, sustainability
    Environmental,
}

/// Weight profile for harmonies in a specific context
#[derive(Debug, Clone)]
pub struct HarmonyWeightProfile {
    /// Name of this profile
    pub name: String,
    /// Weights for each harmony (multipliers, 1.0 = default)
    weights: HashMap<Harmony, f32>,
    /// Description of when to use this profile
    pub description: String,
}

impl HarmonyWeightProfile {
    /// Create a new profile with all weights at 1.0
    pub fn new(name: &str, description: &str) -> Self {
        let mut weights = HashMap::new();
        for harmony in Harmony::all() {
            weights.insert(harmony, 1.0);
        }
        Self {
            name: name.to_string(),
            weights,
            description: description.to_string(),
        }
    }

    /// Set weight for a specific harmony
    pub fn with_weight(mut self, harmony: Harmony, weight: f32) -> Self {
        self.weights.insert(harmony, weight);
        self
    }

    /// Get the weight for a harmony
    pub fn get_weight(&self, harmony: &Harmony) -> f32 {
        self.weights.get(harmony).copied().unwrap_or(1.0)
    }

    /// Get all weights as a vector for easy iteration
    pub fn as_vec(&self) -> Vec<(Harmony, f32)> {
        Harmony::all()
            .iter()
            .map(|h| (*h, self.get_weight(h)))
            .collect()
    }
}

/// Contextual weight manager
#[derive(Debug)]
pub struct ContextualWeights {
    /// Action type profiles
    action_type_profiles: HashMap<ActionType, HarmonyWeightProfile>,
    /// Domain profiles
    domain_profiles: HashMap<ActionDomain, HarmonyWeightProfile>,
    /// Combined profile cache
    cached_profiles: HashMap<(ActionType, ActionDomain), HarmonyWeightProfile>,
}

impl ContextualWeights {
    /// Create a new contextual weights manager with default profiles
    pub fn new() -> Self {
        let mut manager = Self {
            action_type_profiles: HashMap::new(),
            domain_profiles: HashMap::new(),
            cached_profiles: HashMap::new(),
        };
        manager.initialize_default_profiles();
        manager
    }

    /// Initialize default profiles for action types and domains
    fn initialize_default_profiles(&mut self) {
        // === ACTION TYPE PROFILES ===

        // Basic actions: balanced weights
        self.action_type_profiles.insert(
            ActionType::Basic,
            HarmonyWeightProfile::new("Basic", "Default balanced weights for routine actions"),
        );

        // Governance proposals: emphasize wisdom, truth, fairness
        self.action_type_profiles.insert(
            ActionType::Governance,
            HarmonyWeightProfile::new("Governance", "Proposals affecting collective decisions")
                .with_weight(Harmony::IntegralWisdom, 1.4) // Truth is crucial
                .with_weight(Harmony::SacredReciprocity, 1.3) // Fairness matters
                .with_weight(Harmony::PanSentientFlourishing, 1.2) // Do no harm
                .with_weight(Harmony::UniversalInterconnectedness, 1.2) // Consider all stakeholders
                .with_weight(Harmony::InfinitePlay, 0.8), // Creativity less critical
        );

        // Voting: emphasize truth, interconnectedness, fairness
        self.action_type_profiles.insert(
            ActionType::Voting,
            HarmonyWeightProfile::new("Voting", "Casting votes on community decisions")
                .with_weight(Harmony::IntegralWisdom, 1.5) // Truth paramount
                .with_weight(Harmony::UniversalInterconnectedness, 1.4) // Consider community
                .with_weight(Harmony::SacredReciprocity, 1.3) // Fair to all
                .with_weight(Harmony::PanSentientFlourishing, 1.2) // Don't harm minority
                .with_weight(Harmony::InfinitePlay, 0.7), // Play less relevant
        );

        // Constitutional: maximum scrutiny on all core harmonies
        self.action_type_profiles.insert(
            ActionType::Constitutional,
            HarmonyWeightProfile::new("Constitutional", "Fundamental changes to governance")
                .with_weight(Harmony::IntegralWisdom, 1.6) // Wisdom paramount
                .with_weight(Harmony::PanSentientFlourishing, 1.5) // Protect all
                .with_weight(Harmony::SacredReciprocity, 1.5) // Long-term fairness
                .with_weight(Harmony::UniversalInterconnectedness, 1.4) // Systemic thinking
                .with_weight(Harmony::EvolutionaryProgression, 1.3) // Enable growth
                .with_weight(Harmony::ResonantCoherence, 1.2), // System integrity
        );

        // === DOMAIN PROFILES ===

        // General domain: balanced
        self.domain_profiles.insert(
            ActionDomain::General,
            HarmonyWeightProfile::new("General", "No specific domain - balanced approach"),
        );

        // Financial domain: fairness, do-no-harm, wisdom
        self.domain_profiles.insert(
            ActionDomain::Financial,
            HarmonyWeightProfile::new("Financial", "Money, resources, transactions")
                .with_weight(Harmony::SacredReciprocity, 1.4) // Fair exchange
                .with_weight(Harmony::PanSentientFlourishing, 1.3) // Don't harm
                .with_weight(Harmony::IntegralWisdom, 1.2) // No deception
                .with_weight(Harmony::InfinitePlay, 0.7), // Less playful
        );

        // Creative domain: play, coherence, evolution
        self.domain_profiles.insert(
            ActionDomain::Creative,
            HarmonyWeightProfile::new("Creative", "Art, design, innovation")
                .with_weight(Harmony::InfinitePlay, 1.5) // Creativity central
                .with_weight(Harmony::ResonantCoherence, 1.3) // Aesthetic harmony
                .with_weight(Harmony::EvolutionaryProgression, 1.2) // Growth through creation
                .with_weight(Harmony::SacredReciprocity, 0.8), // Less about exchange
        );

        // Social domain: interconnectedness, care, reciprocity
        self.domain_profiles.insert(
            ActionDomain::Social,
            HarmonyWeightProfile::new("Social", "Relationships, community interactions")
                .with_weight(Harmony::UniversalInterconnectedness, 1.4) // Relationships key
                .with_weight(Harmony::PanSentientFlourishing, 1.3) // Care for others
                .with_weight(Harmony::SacredReciprocity, 1.2) // Give and take
                .with_weight(Harmony::InfinitePlay, 1.1), // Playful engagement
        );

        // Technical domain: coherence, wisdom, evolution
        self.domain_profiles.insert(
            ActionDomain::Technical,
            HarmonyWeightProfile::new("Technical", "System operations, code, infrastructure")
                .with_weight(Harmony::ResonantCoherence, 1.4) // System integrity
                .with_weight(Harmony::IntegralWisdom, 1.3) // Correct decisions
                .with_weight(Harmony::EvolutionaryProgression, 1.2) // Continuous improvement
                .with_weight(Harmony::InfinitePlay, 0.9), // Some creativity
        );

        // Educational domain: wisdom, evolution, interconnectedness
        self.domain_profiles.insert(
            ActionDomain::Educational,
            HarmonyWeightProfile::new("Educational", "Teaching, learning, knowledge sharing")
                .with_weight(Harmony::IntegralWisdom, 1.5) // Truth in teaching
                .with_weight(Harmony::EvolutionaryProgression, 1.4) // Growth through learning
                .with_weight(Harmony::UniversalInterconnectedness, 1.2) // Shared knowledge
                .with_weight(Harmony::InfinitePlay, 1.2), // Playful learning
        );

        // Healthcare domain: flourishing, wisdom, care
        self.domain_profiles.insert(
            ActionDomain::Healthcare,
            HarmonyWeightProfile::new("Healthcare", "Health, wellbeing, medical")
                .with_weight(Harmony::PanSentientFlourishing, 1.6) // Do no harm primary
                .with_weight(Harmony::IntegralWisdom, 1.4) // Evidence-based
                .with_weight(Harmony::SacredReciprocity, 1.2) // Patient autonomy
                .with_weight(Harmony::InfinitePlay, 0.7), // Less playful
        );

        // Environmental domain: interconnectedness, evolution, flourishing
        self.domain_profiles.insert(
            ActionDomain::Environmental,
            HarmonyWeightProfile::new("Environmental", "Sustainability, ecology, nature")
                .with_weight(Harmony::UniversalInterconnectedness, 1.5) // Ecosystem thinking
                .with_weight(Harmony::EvolutionaryProgression, 1.4) // Long-term thinking
                .with_weight(Harmony::PanSentientFlourishing, 1.3) // All beings
                .with_weight(Harmony::SacredReciprocity, 1.2), // Give back to earth
        );
    }

    /// Get a combined weight profile for an action type and domain
    pub fn get_combined_profile(
        &mut self,
        action_type: ActionType,
        domain: ActionDomain,
    ) -> &HarmonyWeightProfile {
        let key = (action_type, domain);

        if !self.cached_profiles.contains_key(&key) {
            let combined = self.compute_combined_profile(action_type, domain);
            self.cached_profiles.insert(key, combined);
        }

        // Safe: we just inserted the key if it didn't exist
        self.cached_profiles
            .get(&key)
            .expect("profile was just inserted")
    }

    /// Compute a combined profile from action type and domain
    fn compute_combined_profile(
        &self,
        action_type: ActionType,
        domain: ActionDomain,
    ) -> HarmonyWeightProfile {
        let action_profile = self.action_type_profiles.get(&action_type);
        let domain_profile = self.domain_profiles.get(&domain);

        let name = format!("{action_type:?}_{domain:?}");
        let description =
            format!("Combined profile for {action_type:?} action in {domain:?} domain");

        let mut combined = HarmonyWeightProfile::new(&name, &description);

        // Combine weights: geometric mean of action type and domain weights
        // This gives balanced influence to both factors
        for harmony in Harmony::all() {
            let action_weight = action_profile
                .map(|p| p.get_weight(&harmony))
                .unwrap_or(1.0);
            let domain_weight = domain_profile
                .map(|p| p.get_weight(&harmony))
                .unwrap_or(1.0);

            // Geometric mean for balanced combination
            let combined_weight = (action_weight * domain_weight).sqrt();
            combined.weights.insert(harmony, combined_weight);
        }

        combined
    }

    /// Get weight for a specific harmony in context
    pub fn get_weight(
        &mut self,
        harmony: &Harmony,
        action_type: ActionType,
        domain: ActionDomain,
    ) -> f32 {
        self.get_combined_profile(action_type, domain)
            .get_weight(harmony)
    }

    /// Get all weights for a context as a vector
    pub fn get_all_weights(
        &mut self,
        action_type: ActionType,
        domain: ActionDomain,
    ) -> Vec<(Harmony, f32)> {
        self.get_combined_profile(action_type, domain).as_vec()
    }

    /// Register a custom action type profile
    pub fn register_action_profile(
        &mut self,
        action_type: ActionType,
        profile: HarmonyWeightProfile,
    ) {
        self.action_type_profiles.insert(action_type, profile);
        // Invalidate cache
        self.cached_profiles.retain(|(at, _), _| *at != action_type);
    }

    /// Register a custom domain profile
    pub fn register_domain_profile(&mut self, domain: ActionDomain, profile: HarmonyWeightProfile) {
        self.domain_profiles.insert(domain, profile);
        // Invalidate cache
        self.cached_profiles.retain(|(_, d), _| *d != domain);
    }
}

impl Default for ContextualWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Domain classifier that analyzes text to determine action domain
pub struct DomainClassifier {
    /// Keywords for each domain
    domain_keywords: HashMap<ActionDomain, Vec<&'static str>>,
}

impl DomainClassifier {
    /// Create a new domain classifier
    pub fn new() -> Self {
        let mut keywords = HashMap::new();

        keywords.insert(
            ActionDomain::Financial,
            vec![
                "money",
                "payment",
                "transfer",
                "invest",
                "budget",
                "cost",
                "price",
                "fund",
                "capital",
                "revenue",
                "profit",
                "expense",
                "transaction",
                "bank",
                "loan",
                "credit",
                "debt",
                "asset",
                "stock",
                "crypto",
                "dollar",
                "euro",
                "bitcoin",
                "salary",
                "wage",
                "tax",
                "fee",
            ],
        );

        keywords.insert(
            ActionDomain::Creative,
            vec![
                "create",
                "design",
                "art",
                "music",
                "write",
                "compose",
                "paint",
                "draw",
                "build",
                "craft",
                "imagine",
                "invent",
                "innovate",
                "story",
                "poem",
                "song",
                "novel",
                "aesthetic",
                "beautiful",
                "creative",
                "artistic",
                "express",
                "inspire",
                "style",
                "color",
            ],
        );

        keywords.insert(
            ActionDomain::Social,
            vec![
                "friend",
                "family",
                "community",
                "relationship",
                "connect",
                "share",
                "together",
                "collaborate",
                "team",
                "group",
                "social",
                "people",
                "conversation",
                "message",
                "chat",
                "communicate",
                "meet",
                "party",
                "invite",
                "gather",
                "network",
                "introduce",
                "support",
                "help others",
            ],
        );

        keywords.insert(
            ActionDomain::Technical,
            vec![
                "code",
                "program",
                "software",
                "system",
                "server",
                "database",
                "api",
                "config",
                "deploy",
                "install",
                "update",
                "fix",
                "debug",
                "compile",
                "run",
                "execute",
                "script",
                "function",
                "module",
                "nix",
                "linux",
                "container",
                "docker",
                "kubernetes",
                "git",
            ],
        );

        keywords.insert(
            ActionDomain::Educational,
            vec![
                "learn",
                "teach",
                "study",
                "course",
                "lesson",
                "tutorial",
                "explain",
                "understand",
                "knowledge",
                "educate",
                "train",
                "school",
                "university",
                "lecture",
                "research",
                "discover",
                "curious",
                "question",
                "answer",
                "mentor",
                "student",
                "teacher",
            ],
        );

        keywords.insert(
            ActionDomain::Healthcare,
            vec![
                "health",
                "medical",
                "doctor",
                "patient",
                "treatment",
                "medicine",
                "hospital",
                "clinic",
                "symptom",
                "diagnosis",
                "therapy",
                "care",
                "wellbeing",
                "wellness",
                "fitness",
                "exercise",
                "diet",
                "nutrition",
                "mental",
                "anxiety",
                "depression",
                "heal",
                "recover",
                "pain",
            ],
        );

        keywords.insert(
            ActionDomain::Environmental,
            vec![
                "environment",
                "climate",
                "nature",
                "sustainable",
                "green",
                "eco",
                "recycle",
                "renewable",
                "carbon",
                "pollution",
                "conservation",
                "wildlife",
                "forest",
                "ocean",
                "biodiversity",
                "organic",
                "solar",
                "wind",
                "energy",
                "footprint",
                "earth",
                "planet",
            ],
        );

        Self {
            domain_keywords: keywords,
        }
    }

    /// Classify text into a domain
    pub fn classify(&self, text: &str) -> ActionDomain {
        let text_lower = text.to_lowercase();
        let mut domain_scores: HashMap<ActionDomain, i32> = HashMap::new();

        for (domain, keywords) in &self.domain_keywords {
            let score = keywords
                .iter()
                .filter(|kw| text_lower.contains(*kw))
                .count() as i32;

            if score > 0 {
                domain_scores.insert(*domain, score);
            }
        }

        // Return domain with highest score, or General if no matches
        domain_scores
            .into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(domain, _)| domain)
            .unwrap_or(ActionDomain::General)
    }

    /// Classify with confidence score
    pub fn classify_with_confidence(&self, text: &str) -> (ActionDomain, f32) {
        let text_lower = text.to_lowercase();
        let mut domain_scores: HashMap<ActionDomain, i32> = HashMap::new();
        let mut total_matches = 0;

        for (domain, keywords) in &self.domain_keywords {
            let score = keywords
                .iter()
                .filter(|kw| text_lower.contains(*kw))
                .count() as i32;

            if score > 0 {
                domain_scores.insert(*domain, score);
                total_matches += score;
            }
        }

        if total_matches == 0 {
            return (ActionDomain::General, 0.0);
        }

        let (best_domain, best_score) = domain_scores
            .into_iter()
            .max_by_key(|(_, score)| *score)
            .unwrap_or((ActionDomain::General, 0));

        // Confidence is proportion of matches in the winning domain
        let confidence = best_score as f32 / total_matches as f32;

        (best_domain, confidence)
    }
}

impl Default for DomainClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_weights_are_balanced() {
        let mut weights = ContextualWeights::new();
        let profile = weights.get_combined_profile(ActionType::Basic, ActionDomain::General);

        for harmony in Harmony::all() {
            assert!(
                (profile.get_weight(&harmony) - 1.0).abs() < 0.01,
                "Basic/General should have balanced weights"
            );
        }
    }

    #[test]
    fn test_governance_emphasizes_wisdom() {
        let mut weights = ContextualWeights::new();
        let profile = weights.get_combined_profile(ActionType::Governance, ActionDomain::General);

        let wisdom_weight = profile.get_weight(&Harmony::IntegralWisdom);
        let play_weight = profile.get_weight(&Harmony::InfinitePlay);

        assert!(
            wisdom_weight > play_weight,
            "Governance should weight wisdom ({}) higher than play ({})",
            wisdom_weight,
            play_weight
        );
    }

    #[test]
    fn test_constitutional_has_high_scrutiny() {
        let mut weights = ContextualWeights::new();
        let profile =
            weights.get_combined_profile(ActionType::Constitutional, ActionDomain::General);

        // All core harmonies should have elevated weights
        assert!(profile.get_weight(&Harmony::IntegralWisdom) > 1.0);
        assert!(profile.get_weight(&Harmony::PanSentientFlourishing) > 1.0);
        assert!(profile.get_weight(&Harmony::SacredReciprocity) > 1.0);
    }

    #[test]
    fn test_financial_domain_emphasizes_fairness() {
        let mut weights = ContextualWeights::new();
        let profile = weights.get_combined_profile(ActionType::Basic, ActionDomain::Financial);

        let reciprocity_weight = profile.get_weight(&Harmony::SacredReciprocity);
        let play_weight = profile.get_weight(&Harmony::InfinitePlay);

        assert!(
            reciprocity_weight > play_weight,
            "Financial domain should weight reciprocity ({}) higher than play ({})",
            reciprocity_weight,
            play_weight
        );
    }

    #[test]
    fn test_creative_domain_emphasizes_play() {
        let mut weights = ContextualWeights::new();
        let profile = weights.get_combined_profile(ActionType::Basic, ActionDomain::Creative);

        let play_weight = profile.get_weight(&Harmony::InfinitePlay);

        assert!(
            play_weight > 1.0,
            "Creative domain should elevate play weight ({})",
            play_weight
        );
    }

    #[test]
    fn test_combined_profile_uses_geometric_mean() {
        let mut weights = ContextualWeights::new();

        // Governance has wisdom at 1.4, financial has wisdom at 1.2
        // Combined should be sqrt(1.4 * 1.2) ≈ 1.296
        let profile = weights.get_combined_profile(ActionType::Governance, ActionDomain::Financial);
        let wisdom_weight = profile.get_weight(&Harmony::IntegralWisdom);

        let expected = (1.4_f32 * 1.2_f32).sqrt();
        assert!(
            (wisdom_weight - expected).abs() < 0.01,
            "Combined wisdom weight ({}) should be geometric mean ({})",
            wisdom_weight,
            expected
        );
    }

    #[test]
    fn test_domain_classifier_financial() {
        let classifier = DomainClassifier::new();

        let domain = classifier.classify("I want to transfer money to my bank account");
        assert_eq!(domain, ActionDomain::Financial);
    }

    #[test]
    fn test_domain_classifier_creative() {
        let classifier = DomainClassifier::new();

        let domain = classifier.classify("Create a beautiful design for my art project");
        assert_eq!(domain, ActionDomain::Creative);
    }

    #[test]
    fn test_domain_classifier_technical() {
        let classifier = DomainClassifier::new();

        let domain = classifier.classify("Deploy the code to the server and update the database");
        assert_eq!(domain, ActionDomain::Technical);
    }

    #[test]
    fn test_domain_classifier_healthcare() {
        let classifier = DomainClassifier::new();

        let domain = classifier.classify("Schedule an appointment with the doctor for my symptoms");
        assert_eq!(domain, ActionDomain::Healthcare);
    }

    #[test]
    fn test_domain_classifier_general_fallback() {
        let classifier = DomainClassifier::new();

        let domain = classifier.classify("Do something interesting today");
        assert_eq!(domain, ActionDomain::General);
    }

    #[test]
    fn test_confidence_scoring() {
        let classifier = DomainClassifier::new();

        // Strong financial signal
        let (domain, confidence) = classifier
            .classify_with_confidence("Transfer money to bank, pay the loan, check credit score");
        assert_eq!(domain, ActionDomain::Financial);
        assert!(
            confidence > 0.8,
            "Strong signal should have high confidence: {}",
            confidence
        );

        // Mixed signal
        let (_, mixed_confidence) =
            classifier.classify_with_confidence("Create art about money and health");
        assert!(
            mixed_confidence < 0.8,
            "Mixed signal should have lower confidence: {}",
            mixed_confidence
        );
    }

    #[test]
    fn test_healthcare_prioritizes_do_no_harm() {
        let mut weights = ContextualWeights::new();
        let profile = weights.get_combined_profile(ActionType::Basic, ActionDomain::Healthcare);

        let flourishing_weight = profile.get_weight(&Harmony::PanSentientFlourishing);

        // Combined weight is geometric mean of Basic (1.0) and Healthcare (1.6)
        // = sqrt(1.0 * 1.6) ≈ 1.265
        // It should still be elevated above default (1.0)
        assert!(
            flourishing_weight > 1.0,
            "Healthcare should elevate Pan-Sentient Flourishing above default: {}",
            flourishing_weight
        );

        // Also verify it's the expected geometric mean
        let expected = (1.0_f32 * 1.6_f32).sqrt();
        assert!(
            (flourishing_weight - expected).abs() < 0.01,
            "Healthcare flourishing weight ({}) should equal geometric mean ({})",
            flourishing_weight,
            expected
        );
    }

    #[test]
    fn test_action_type_default() {
        let action_type = ActionType::default();
        assert_eq!(action_type, ActionType::Basic);
    }

    #[test]
    fn test_action_domain_default() {
        let domain = ActionDomain::default();
        assert_eq!(domain, ActionDomain::General);
    }

    #[test]
    fn test_all_action_types_have_profiles() {
        let mut weights = ContextualWeights::new();

        // Verify all action types produce valid profiles
        let action_types = [
            ActionType::Basic,
            ActionType::Governance,
            ActionType::Voting,
            ActionType::Constitutional,
        ];

        for action_type in action_types {
            let profile = weights.get_combined_profile(action_type, ActionDomain::General);
            // All profiles should have non-zero weights
            for harmony in Harmony::all() {
                let weight = profile.get_weight(&harmony);
                assert!(
                    weight > 0.0,
                    "{:?} should have positive weight for {:?}",
                    action_type,
                    harmony
                );
            }
        }
    }

    #[test]
    fn test_voting_emphasizes_truth_and_community() {
        let mut weights = ContextualWeights::new();
        let profile = weights.get_combined_profile(ActionType::Voting, ActionDomain::General);

        // Voting should prioritize truth (IntegralWisdom) and community (UniversalInterconnectedness)
        let wisdom = profile.get_weight(&Harmony::IntegralWisdom);
        let interconnectedness = profile.get_weight(&Harmony::UniversalInterconnectedness);

        assert!(wisdom > 1.0, "Voting should elevate wisdom: {}", wisdom);
        assert!(
            interconnectedness > 1.0,
            "Voting should elevate interconnectedness: {}",
            interconnectedness
        );
    }

    #[test]
    fn test_custom_domain_profile_registration() {
        let mut weights = ContextualWeights::new();

        // Create a custom profile for a hypothetical Gaming domain
        let gaming_profile = HarmonyWeightProfile::new("Gaming", "Interactive entertainment")
            .with_weight(Harmony::InfinitePlay, 2.0)
            .with_weight(Harmony::ResonantCoherence, 1.5);

        // Register as Environmental (just for testing registration mechanism)
        weights.register_domain_profile(ActionDomain::Environmental, gaming_profile);

        let profile = weights.get_combined_profile(ActionType::Basic, ActionDomain::Environmental);

        // The custom profile should now be in effect
        let play_weight = profile.get_weight(&Harmony::InfinitePlay);
        // Basic has 1.0, custom has 2.0, geometric mean = sqrt(2.0) ≈ 1.414
        let expected = (1.0_f32 * 2.0_f32).sqrt();
        assert!(
            (play_weight - expected).abs() < 0.1,
            "Custom profile should affect weights: {} vs expected {}",
            play_weight,
            expected
        );
    }

    #[test]
    fn test_harmony_weight_profile_as_vec() {
        let profile = HarmonyWeightProfile::new("Test", "Test profile")
            .with_weight(Harmony::InfinitePlay, 1.5);

        let weights_vec = profile.as_vec();
        assert_eq!(weights_vec.len(), 8, "Should have 8 harmonies");

        // Find InfinitePlay in the vector
        let play_entry = weights_vec
            .iter()
            .find(|(h, _)| *h == Harmony::InfinitePlay)
            .expect("InfinitePlay should be in the vector");
        assert!(
            (play_entry.1 - 1.5).abs() < 0.01,
            "InfinitePlay should have weight 1.5"
        );
    }

    #[test]
    fn test_domain_classifier_educational() {
        let classifier = DomainClassifier::new();

        let domain = classifier.classify("I want to learn and understand this new concept");
        assert_eq!(domain, ActionDomain::Educational);
    }

    #[test]
    fn test_domain_classifier_social() {
        let classifier = DomainClassifier::new();

        let domain =
            classifier.classify("Let's collaborate with the team and share ideas together");
        assert_eq!(domain, ActionDomain::Social);
    }

    #[test]
    fn test_domain_classifier_environmental() {
        let classifier = DomainClassifier::new();

        let domain =
            classifier.classify("We need to focus on sustainable and renewable energy sources");
        assert_eq!(domain, ActionDomain::Environmental);
    }

    #[test]
    fn test_get_weight_method() {
        let mut weights = ContextualWeights::new();

        let weight = weights.get_weight(
            &Harmony::IntegralWisdom,
            ActionType::Governance,
            ActionDomain::General,
        );

        // Governance elevates wisdom to 1.4
        assert!(
            weight > 1.0,
            "Governance wisdom weight should be elevated: {}",
            weight
        );
    }

    #[test]
    fn test_get_all_weights_method() {
        let mut weights = ContextualWeights::new();

        let all_weights = weights.get_all_weights(ActionType::Basic, ActionDomain::General);

        assert_eq!(
            all_weights.len(),
            8,
            "Should return weights for all 8 harmonies"
        );

        // Basic/General should all be close to 1.0
        for (harmony, weight) in all_weights {
            assert!(
                (weight - 1.0).abs() < 0.01,
                "Basic/General {:?} weight should be 1.0, got {}",
                harmony,
                weight
            );
        }
    }
}
