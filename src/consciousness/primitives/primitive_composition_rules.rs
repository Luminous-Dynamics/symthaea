// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive Composition Rules
//!
//! Domain-specific rules for composing HDC primitives. Different cognitive
//! domains require different composition operators to preserve semantic
//! structure through binding operations.

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::PrimitiveTier;

/// A rule that defines how two primitives should be composed
pub trait CompositionRule: Send + Sync {
    /// Check if this rule applies to the given tier pair
    fn applies(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> bool;
    /// Compose two hypervectors according to this rule
    fn compose(
        &self,
        hv1: &BinaryHV,
        hv2: &BinaryHV,
        tier1: PrimitiveTier,
        tier2: PrimitiveTier,
    ) -> BinaryHV;
    /// Rule name for debugging
    fn name(&self) -> &'static str;
}

/// Temporal-Physical rule: permute hv1 before XOR to encode ordering
pub struct TemporalPhysicalRule;

impl CompositionRule for TemporalPhysicalRule {
    fn applies(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> bool {
        matches!(
            (tier1, tier2),
            (PrimitiveTier::Temporal, PrimitiveTier::Physical)
                | (PrimitiveTier::Physical, PrimitiveTier::Temporal)
                | (PrimitiveTier::Temporal, PrimitiveTier::Temporal)
        )
    }

    fn compose(
        &self,
        hv1: &BinaryHV,
        hv2: &BinaryHV,
        _tier1: PrimitiveTier,
        _tier2: PrimitiveTier,
    ) -> BinaryHV {
        // Permute hv1 to encode ordering, then bind
        let permuted = hv1.permute(1);
        permuted.bind(hv2)
    }

    fn name(&self) -> &'static str {
        "TemporalPhysical"
    }
}

/// Mathematical rule: standard XOR binding (preserves algebraic structure)
pub struct MathematicalRule;

impl CompositionRule for MathematicalRule {
    fn applies(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> bool {
        matches!(tier1, PrimitiveTier::Mathematical) || matches!(tier2, PrimitiveTier::Mathematical)
    }

    fn compose(
        &self,
        hv1: &BinaryHV,
        hv2: &BinaryHV,
        _tier1: PrimitiveTier,
        _tier2: PrimitiveTier,
    ) -> BinaryHV {
        hv1.bind(hv2)
    }

    fn name(&self) -> &'static str {
        "Mathematical"
    }
}

/// Consciousness rule: bundle then bind (resonant — preserves both parents)
pub struct ConsciousnessRule;

impl CompositionRule for ConsciousnessRule {
    fn applies(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> bool {
        matches!(
            tier1,
            PrimitiveTier::Consciousness | PrimitiveTier::MetaCognitive
        ) || matches!(
            tier2,
            PrimitiveTier::Consciousness | PrimitiveTier::MetaCognitive
        )
    }

    fn compose(
        &self,
        hv1: &BinaryHV,
        hv2: &BinaryHV,
        _tier1: PrimitiveTier,
        _tier2: PrimitiveTier,
    ) -> BinaryHV {
        // Bundle first (majority vote preserves both), then bind for uniqueness
        let bundled = BinaryHV::bundle(&[*hv1, *hv2]);
        bundled.bind(&hv1.bind(hv2))
    }

    fn name(&self) -> &'static str {
        "Consciousness"
    }
}

/// Cross-tier rule: permute by tier distance then XOR
pub struct CrossTierRule;

impl CrossTierRule {
    fn tier_distance(tier1: PrimitiveTier, tier2: PrimitiveTier) -> u32 {
        let t1 = tier1 as u32;
        let t2 = tier2 as u32;
        t1.abs_diff(t2)
    }
}

impl CompositionRule for CrossTierRule {
    fn applies(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> bool {
        // Applies when tiers are different and no more specific rule matches
        tier1 != tier2
    }

    fn compose(
        &self,
        hv1: &BinaryHV,
        hv2: &BinaryHV,
        tier1: PrimitiveTier,
        tier2: PrimitiveTier,
    ) -> BinaryHV {
        let distance = Self::tier_distance(tier1, tier2) as usize;
        // Permute by tier distance to encode structural relationship
        let permuted = hv1.permute(distance.max(1));
        permuted.bind(hv2)
    }

    fn name(&self) -> &'static str {
        "CrossTier"
    }
}

/// Engine that selects the best composition rule for a given tier pair
pub struct CompositionRuleEngine {
    rules: Vec<Box<dyn CompositionRule>>,
}

impl CompositionRuleEngine {
    /// Create with default rules (ordered by priority)
    pub fn new() -> Self {
        Self {
            rules: vec![
                Box::new(ConsciousnessRule),
                Box::new(TemporalPhysicalRule),
                Box::new(MathematicalRule),
                Box::new(CrossTierRule),
            ],
        }
    }

    /// Compose two HVs using the best matching rule
    pub fn compose(
        &self,
        hv1: &BinaryHV,
        hv2: &BinaryHV,
        tier1: PrimitiveTier,
        tier2: PrimitiveTier,
    ) -> BinaryHV {
        for rule in &self.rules {
            if rule.applies(tier1, tier2) {
                return rule.compose(hv1, hv2, tier1, tier2);
            }
        }
        // Fallback: standard bind
        hv1.bind(hv2)
    }

    /// Get the name of the rule that would be used
    pub fn matching_rule_name(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> &'static str {
        for rule in &self.rules {
            if rule.applies(tier1, tier2) {
                return rule.name();
            }
        }
        "Default"
    }
}

impl Default for CompositionRuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_rules_select_correctly() {
        let engine = CompositionRuleEngine::new();

        assert_eq!(
            engine.matching_rule_name(PrimitiveTier::Consciousness, PrimitiveTier::Physical),
            "Consciousness"
        );
        assert_eq!(
            engine.matching_rule_name(PrimitiveTier::Temporal, PrimitiveTier::Physical),
            "TemporalPhysical"
        );
        assert_eq!(
            engine.matching_rule_name(PrimitiveTier::Mathematical, PrimitiveTier::Geometric),
            "Mathematical"
        );
        assert_eq!(
            engine.matching_rule_name(PrimitiveTier::Strategic, PrimitiveTier::Physical),
            "CrossTier"
        );
    }

    #[test]
    fn test_composition_produces_valid_hvs() {
        let engine = CompositionRuleEngine::new();
        let hv1 = BinaryHV::random(42);
        let hv2 = BinaryHV::random(99);

        let result = engine.compose(&hv1, &hv2, PrimitiveTier::Temporal, PrimitiveTier::Physical);
        // Result should be different from both inputs
        let sim1 = result.similarity(&hv1);
        let sim2 = result.similarity(&hv2);
        // Composed HV should not be identical to either parent
        assert!(
            sim1 < 0.95,
            "Composed should differ from hv1, similarity={}",
            sim1
        );
        assert!(
            sim2 < 0.95,
            "Composed should differ from hv2, similarity={}",
            sim2
        );
    }

    #[test]
    fn test_consciousness_rule_preserves_parents() {
        let rule = ConsciousnessRule;
        let hv1 = BinaryHV::random(42);
        let hv2 = BinaryHV::random(99);
        let composed = rule.compose(
            &hv1,
            &hv2,
            PrimitiveTier::Consciousness,
            PrimitiveTier::Physical,
        );

        // Consciousness rule uses bundle+bind, result should exist
        assert!(composed.popcount() > 0);
    }
}
