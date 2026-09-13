// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Collective/context-aware/meta-reasoning integration.
//!
//! The historical API name `UnifiedIntelligence` is retained for compatibility, but the
//! diagnostics in this module are **measurement-only telemetry**. In particular, the current
//! `MetaCognitiveReasoner::meta_confidence` is not yet qualified as a probability of answer
//! correctness, self-awareness, consciousness, or general intelligence. RQ-006 owns any future
//! authority upgrade.
//!
//! The module therefore distinguishes mechanism observations from capability claims:
//!
//! `integration observed != intelligence established != self-awareness established`.

use crate::consciousness::meta_reasoning::{
    MetaCognitiveReasoner, MetaReasoningConfig, MetaReasoningResult,
};
use crate::consciousness::primitive_evolution::{
    CandidatePrimitive, EvolutionConfig, PrimitiveEvolution, PrimitiveEvolutionResult,
};
use crate::consciousness::primitive_reasoning::ReasoningChain;
use crate::hdc::BinaryHV;
use crate::hdc::primitive_system::PrimitiveTier;
use crate::physiology::social_coherence::CollectivePrimitiveEvolution;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Authority carried by the diagnostics emitted from this module.
///
/// There is intentionally no stronger variant yet. A future variant must be introduced only by
/// an evidence-backed qualification change, not by a threshold crossing inside this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ObservationAuthority {
    #[default]
    MeasurementOnly,
}

/// A reasoning instance in the integrated multi-agent system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningInstance {
    pub id: String,
    pub local_primitives: Vec<CandidatePrimitive>,
    pub contribution_count: usize,
    /// Heuristic meta-confidence telemetry. Not a qualified correctness probability.
    pub meta_confidence: f64,
    pub episodes_completed: usize,
}

impl ReasoningInstance {
    pub fn new(id: String) -> Self {
        Self {
            id,
            local_primitives: Vec::new(),
            contribution_count: 0,
            meta_confidence: 0.5,
            episodes_completed: 0,
        }
    }
}

/// Result of one integrated reasoning episode.
#[derive(Debug, Clone)]
pub struct UnifiedReasoningResult {
    pub instance_id: String,
    pub meta_result: MetaReasoningResult,
    pub primitives_used: Vec<CandidatePrimitive>,
    pub collective_primitives_count: usize,
    pub local_primitives_count: usize,
    /// Mechanism-level observations. Every entry is explicitly measurement-only.
    pub emergent_properties: Vec<EmergentProperty>,
    /// Historical compatibility field.
    ///
    /// This value is an integration heuristic assembled from unqualified internal signals. It
    /// must not be interpreted as an intelligence, AGI, consciousness, or self-awareness score.
    pub unified_intelligence: f64,
    /// Explicit authority boundary for all diagnostics above.
    pub authority: ObservationAuthority,
}

/// One observation emitted by the integration layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentProperty {
    pub property: String,
    pub strength: f64,
    pub evidence: String,
    #[serde(default)]
    pub authority: ObservationAuthority,
}

/// Statistics for the integrated system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSystemStats {
    pub total_instances: usize,
    pub total_collective_primitives: usize,
    pub total_episodes: usize,
    /// Average of the current heuristic meta-confidence telemetry.
    pub avg_meta_confidence: f64,
    pub emergent_properties: Vec<EmergentProperty>,
    /// Historical compatibility field: collective/local primitive-count ratio.
    /// It is not a qualified intelligence comparison.
    pub collective_intelligence: f64,
    #[serde(default)]
    pub authority: ObservationAuthority,
}

/// Integrated collective primitive selection, context optimization, and meta-reasoning.
///
/// The type name is historical. Its outputs remain `MeasurementOnly` until separately qualified.
pub struct UnifiedIntelligence {
    instances: HashMap<String, ReasoningInstance>,
    collective: CollectivePrimitiveEvolution,
    evolution_config: EvolutionConfig,
    meta_config: MetaReasoningConfig,
    stats: UnifiedSystemStats,
}

impl UnifiedIntelligence {
    pub fn new(
        system_id: String,
        evolution_config: EvolutionConfig,
        meta_config: MetaReasoningConfig,
    ) -> Self {
        Self {
            instances: HashMap::new(),
            collective: CollectivePrimitiveEvolution::new(system_id),
            evolution_config,
            meta_config,
            stats: UnifiedSystemStats {
                total_instances: 0,
                total_collective_primitives: 0,
                total_episodes: 0,
                avg_meta_confidence: 0.0,
                emergent_properties: Vec::new(),
                collective_intelligence: 0.0,
                authority: ObservationAuthority::MeasurementOnly,
            },
        }
    }

    pub fn add_instance(&mut self, instance_id: String) -> Result<()> {
        let instance = ReasoningInstance::new(instance_id.clone());
        self.instances.insert(instance_id, instance);
        self.stats.total_instances += 1;
        Ok(())
    }

    pub fn evolve_local_primitives(
        &mut self,
        instance_id: &str,
        _initial_primitives: Vec<CandidatePrimitive>,
    ) -> Result<PrimitiveEvolutionResult> {
        let mut evolution = PrimitiveEvolution::new(self.evolution_config.clone())?;
        evolution.initialize_population();
        let result = evolution.evolve()?;
        if let Some(instance) = self.instances.get_mut(instance_id) {
            instance.local_primitives = result.final_primitives.clone();
        }
        Ok(result)
    }

    pub fn contribute_to_collective(
        &mut self,
        instance_id: &str,
        primitive: CandidatePrimitive,
        success: bool,
        phi_improvement: f32,
        harmonic_score: f32,
        epistemic_score: f32,
    ) {
        self.collective.contribute_primitive(
            primitive,
            success,
            phi_improvement,
            harmonic_score,
            epistemic_score,
        );
        if let Some(instance) = self.instances.get_mut(instance_id) {
            instance.contribution_count += 1;
        }
        let (_, prims, _) = self.collective.get_stats();
        self.stats.total_collective_primitives = prims;
    }

    pub fn get_reasoning_primitives(
        &self,
        instance_id: &str,
        tier: PrimitiveTier,
        count: usize,
    ) -> (Vec<CandidatePrimitive>, usize, usize) {
        let collective_primitives = self.collective.query_top_primitives(tier, count);
        let collective_count = collective_primitives.len();
        let local_primitives = if let Some(instance) = self.instances.get(instance_id) {
            instance
                .local_primitives
                .iter()
                .filter(|p| p.tier == tier)
                .take(count.saturating_sub(collective_count))
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let local_count = local_primitives.len();
        let mut all_primitives = collective_primitives;
        all_primitives.extend(local_primitives);
        (all_primitives, collective_count, local_count)
    }

    pub fn unified_reason(
        &mut self,
        instance_id: &str,
        query: &str,
        tier: PrimitiveTier,
    ) -> Result<UnifiedReasoningResult> {
        let (primitives, collective_count, local_count) =
            self.get_reasoning_primitives(instance_id, tier, 20);
        if primitives.is_empty() {
            anyhow::bail!("No primitives available for reasoning");
        }

        let mut meta_reasoner =
            MetaCognitiveReasoner::new(self.evolution_config.clone(), self.meta_config.clone())?;
        let mut chain = ReasoningChain::new(BinaryHV::random(self.stats.total_episodes as u64));
        let meta_result = meta_reasoner.meta_reason(query, primitives.clone(), &mut chain)?;

        let chosen_primitive = &meta_result.optimization_result.primitive;
        let fitness = chosen_primitive.fitness as f32;
        let harmonic = chosen_primitive.harmonic_alignment as f32;
        let epistemic = chosen_primitive.epistemic_coordinate.quality_score() as f32;
        self.contribute_to_collective(
            instance_id,
            chosen_primitive.clone(),
            fitness > 0.3,
            fitness,
            harmonic,
            epistemic,
        );

        let emergent_properties =
            self.detect_observations(&meta_result, collective_count, &primitives);

        if let Some(instance) = self.instances.get_mut(instance_id) {
            instance.meta_confidence = meta_result.meta_confidence;
            instance.episodes_completed += 1;
        }

        self.stats.total_episodes += 1;
        self.update_system_stats(&emergent_properties);

        // Historical field name retained. The computation is explicitly telemetry-only.
        let unified_intelligence = self.compute_unified_integration_heuristic(&meta_result);

        Ok(UnifiedReasoningResult {
            instance_id: instance_id.to_string(),
            meta_result,
            primitives_used: primitives,
            collective_primitives_count: collective_count,
            local_primitives_count: local_count,
            emergent_properties,
            unified_intelligence,
            authority: ObservationAuthority::MeasurementOnly,
        })
    }

    fn detect_observations(
        &self,
        meta_result: &MetaReasoningResult,
        collective_count: usize,
        primitives: &[CandidatePrimitive],
    ) -> Vec<EmergentProperty> {
        let mut properties = Vec::new();

        if collective_count > 0 {
            let collective_ratio = collective_count as f64 / primitives.len() as f64;
            properties.push(EmergentProperty {
                property: "Collective Primitive Utilization".to_string(),
                strength: collective_ratio,
                evidence: format!(
                    "Used {} collective primitives ({:.1}%) out of {} total",
                    collective_count,
                    collective_ratio * 100.0,
                    primitives.len()
                ),
                authority: ObservationAuthority::MeasurementOnly,
            });
        }

        // This threshold describes the heuristic itself; it establishes neither correctness nor
        // self-awareness. RQ-006 must qualify those semantics independently.
        if meta_result.meta_confidence > 0.7 {
            properties.push(EmergentProperty {
                property: "High Heuristic Meta-Confidence".to_string(),
                strength: meta_result.meta_confidence,
                evidence: format!(
                    "Heuristic meta-confidence {:.2}; RQ-006 qualification pending",
                    meta_result.meta_confidence
                ),
                authority: ObservationAuthority::MeasurementOnly,
            });
        }

        if meta_result.context_reflection.reconsider_context
            || meta_result.strategy_reflection.adjust_strategy
        {
            properties.push(EmergentProperty {
                property: "Context Or Strategy Adjustment Triggered".to_string(),
                strength: 1.0,
                evidence: "The meta-reasoning mechanism requested context reconsideration or strategy adjustment"
                    .to_string(),
                authority: ObservationAuthority::MeasurementOnly,
            });
        }

        let fitness = meta_result
            .optimization_result
            .tradeoff_point
            .weighted_fitness(&meta_result.optimization_result.weights);
        if fitness > 0.6 {
            properties.push(EmergentProperty {
                property: "High Weighted Objective Fitness".to_string(),
                strength: fitness,
                evidence: format!(
                    "Observed weighted objective fitness {:.2} under the current configured weights",
                    fitness
                ),
                authority: ObservationAuthority::MeasurementOnly,
            });
        }

        if !meta_result.meta_insights.is_empty() {
            let avg_reliability = meta_result
                .meta_insights
                .iter()
                .map(|insight| insight.reliability)
                .sum::<f64>()
                / meta_result.meta_insights.len() as f64;
            properties.push(EmergentProperty {
                property: "Meta-Learning Insight Emitted".to_string(),
                strength: avg_reliability,
                evidence: format!(
                    "Emitted {} heuristic meta-insights with mean internal reliability {:.2}",
                    meta_result.meta_insights.len(),
                    avg_reliability
                ),
                authority: ObservationAuthority::MeasurementOnly,
            });
        }

        properties
    }

    fn update_system_stats(&mut self, observations: &[EmergentProperty]) {
        for observation in observations {
            if !self
                .stats
                .emergent_properties
                .iter()
                .any(|existing| existing.property == observation.property)
            {
                self.stats.emergent_properties.push(observation.clone());
            }
        }

        let total_confidence: f64 = self.instances.values().map(|i| i.meta_confidence).sum();
        self.stats.avg_meta_confidence = if self.instances.is_empty() {
            0.0
        } else {
            total_confidence / self.instances.len() as f64
        };
        self.stats.collective_intelligence = self.compute_collective_sharing_ratio();
        self.stats.authority = ObservationAuthority::MeasurementOnly;
    }

    /// Compatibility heuristic over internal integration signals.
    ///
    /// The numerical formula is retained to avoid silently changing historical telemetry while
    /// RQ-006 is in progress. The result has no capability authority.
    fn compute_unified_integration_heuristic(&self, meta_result: &MetaReasoningResult) -> f64 {
        let meta_component = 0.3 * meta_result.meta_confidence;
        let fitness_component = 0.3
            * meta_result
                .optimization_result
                .tradeoff_point
                .weighted_fitness(&meta_result.optimization_result.weights);
        let collective_component =
            0.2 * (self.stats.total_collective_primitives as f64 / 100.0).min(1.0);
        let frontier_size = meta_result.optimization_result.frontier.size() as f64;
        let frontier_component = 0.2 * (frontier_size / 10.0).min(1.0);
        meta_component + fitness_component + collective_component + frontier_component
    }

    /// Historical `collective_intelligence` telemetry: collective/local primitive-count ratio.
    fn compute_collective_sharing_ratio(&self) -> f64 {
        if self.instances.is_empty() {
            return 0.0;
        }
        let avg_local_primitives = self
            .instances
            .values()
            .map(|instance| instance.local_primitives.len())
            .sum::<usize>() as f64
            / self.instances.len() as f64;
        let collective_primitives = self.stats.total_collective_primitives as f64;
        if avg_local_primitives > 0.0 {
            collective_primitives / avg_local_primitives
        } else {
            1.0
        }
    }

    pub fn merge_instances(&mut self, other_collective: &CollectivePrimitiveEvolution) {
        self.collective.merge_knowledge(other_collective);
        let (_, prims, _) = self.collective.get_stats();
        self.stats.total_collective_primitives = prims;
    }

    pub fn stats(&self) -> &UnifiedSystemStats {
        &self.stats
    }

    pub fn instance(&self, instance_id: &str) -> Option<&ReasoningInstance> {
        self.instances.get(instance_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrated_telemetry_starts_measurement_only() {
        let system = UnifiedIntelligence::new(
            "test_system".to_string(),
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        );
        assert_eq!(system.stats().total_instances, 0);
        assert_eq!(system.stats().total_episodes, 0);
        assert_eq!(
            system.stats().authority,
            ObservationAuthority::MeasurementOnly
        );
    }

    #[test]
    fn add_instance_preserves_counting() {
        let mut system = UnifiedIntelligence::new(
            "test_system".to_string(),
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        );
        system.add_instance("instance_a".to_string()).unwrap();
        system.add_instance("instance_b".to_string()).unwrap();
        assert_eq!(system.stats().total_instances, 2);
    }

    #[test]
    fn empty_collective_sharing_ratio_is_zero() {
        let system = UnifiedIntelligence::new(
            "test_system".to_string(),
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        );
        assert_eq!(system.compute_collective_sharing_ratio(), 0.0);
    }

    #[test]
    fn authority_has_no_implicit_upgrade_path() {
        assert_eq!(
            ObservationAuthority::default(),
            ObservationAuthority::MeasurementOnly
        );
    }
}
