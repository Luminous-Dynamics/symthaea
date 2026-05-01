// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi Code Quality — Use IIT-inspired metrics for code quality
//!
//! Measures "information integration" of code structure as a quality metric.
//! Highly integrated code (tight cohesion, no dead code) has high Phi.
//! Disconnected modules and dead code have low Phi.
//!
//! # Phi as Code Quality
//!
//! - **High Phi**: Tightly integrated, every entity references others, strong cohesion
//! - **Low Phi**: Disconnected modules, dead code, poor cohesion
//!
//! This is an analogy to IIT's Phi — not an exact computation (which is
//! intractable for large systems), but a useful proxy.

use std::collections::HashMap;

use symthaea_core::hdc::ContinuousHV;

use crate::hdc::code_encoder::CodeHDEncoder;
use crate::language::code_parser::{Entity, ParsedCode};

/// Phi-inspired code quality score
#[derive(Debug, Clone)]
pub struct PhiCodeScore {
    /// Overall integration score (0.0 - 1.0)
    pub integration: f32,
    /// Cohesion: how related are entities within the module?
    pub cohesion: f32,
    /// Connectivity: fraction of entities that reference at least one other
    pub connectivity: f32,
    /// Dead code ratio: entities that are never referenced
    pub dead_code_ratio: f32,
    /// Number of connected components in the dependency graph
    pub component_count: usize,
    /// Total entities analyzed
    pub entity_count: usize,
}

impl PhiCodeScore {
    /// Compute a single combined quality score
    pub fn quality_score(&self) -> f32 {
        // Weighted combination
        let score = self.integration * 0.3
            + self.cohesion * 0.3
            + self.connectivity * 0.2
            + (1.0 - self.dead_code_ratio) * 0.2;
        score.clamp(0.0, 1.0)
    }

    /// Human-readable quality level
    pub fn quality_level(&self) -> &'static str {
        let score = self.quality_score();
        match score {
            s if s >= 0.8 => "Excellent — highly integrated",
            s if s >= 0.6 => "Good — well-connected",
            s if s >= 0.4 => "Fair — some disconnected parts",
            s if s >= 0.2 => "Poor — significant dead code or isolation",
            _ => "Critical — largely disconnected",
        }
    }
}

/// Change in Phi between two code versions
#[derive(Debug, Clone)]
pub struct PhiDelta {
    /// Phi before the change
    pub before: PhiCodeScore,
    /// Phi after the change
    pub after: PhiCodeScore,
    /// Delta in overall quality score
    pub delta: f32,
    /// Whether the change improved integration
    pub improved: bool,
}

/// Phi-based code quality analyzer
pub struct PhiCodeAnalyzer {
    encoder: CodeHDEncoder,
}

impl PhiCodeAnalyzer {
    /// Create a new analyzer
    pub fn new(encoder: CodeHDEncoder) -> Self {
        Self { encoder }
    }

    /// Create with default dimension
    pub fn default_dim() -> Self {
        Self::new(CodeHDEncoder::new(512))
    }

    /// Measure the Phi-like integration of parsed code
    pub fn measure_code_phi(&self, parsed: &ParsedCode) -> PhiCodeScore {
        let entities = parsed.all_entities();
        let entity_count = entities.len();

        if entity_count == 0 {
            return PhiCodeScore {
                integration: 0.0,
                cohesion: 0.0,
                connectivity: 0.0,
                dead_code_ratio: 0.0,
                component_count: 0,
                entity_count: 0,
            };
        }

        // Build adjacency from explicit relations
        let mut referenced: HashMap<String, bool> = HashMap::new();
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        for entity in &entities {
            adjacency.entry(entity.name.clone()).or_default();
            referenced.entry(entity.name.clone()).or_insert(false);
        }

        for rel in &parsed.structure.relations {
            adjacency
                .entry(rel.source.clone())
                .or_default()
                .push(rel.target.clone());
            *referenced.entry(rel.target.clone()).or_insert(false) = true;
        }

        // Connectivity: fraction of entities with at least one reference
        let connected_count = adjacency.values().filter(|adj| !adj.is_empty()).count();
        let connectivity = connected_count as f32 / entity_count as f32;

        // Dead code: entities that have no references TO them (except the "main" entries)
        let dead_count = referenced.values().filter(|&&is_ref| !is_ref).count();
        // Don't count top-level exports as dead
        let dead_code_ratio = if entity_count > 1 {
            (dead_count.saturating_sub(1)) as f32 / entity_count as f32
        } else {
            0.0
        };

        // Cohesion: average pairwise similarity of entity HVs
        let cohesion = self.compute_hdc_cohesion(&entities);

        // Connected components using simple DFS
        let component_count = self.count_components(&adjacency);

        // Integration: normalized combination
        let max_components = entity_count.max(1) as f32;
        let component_integration = 1.0 - (component_count as f32 - 1.0).max(0.0) / max_components;

        let integration = (cohesion + connectivity + component_integration) / 3.0;

        PhiCodeScore {
            integration,
            cohesion,
            connectivity,
            dead_code_ratio,
            component_count,
            entity_count,
        }
    }

    /// Compare Phi before and after a change
    pub fn phi_diff(&self, before: &ParsedCode, after: &ParsedCode) -> PhiDelta {
        let before_score = self.measure_code_phi(before);
        let after_score = self.measure_code_phi(after);

        let delta = after_score.quality_score() - before_score.quality_score();

        PhiDelta {
            improved: delta > 0.0,
            delta,
            before: before_score,
            after: after_score,
        }
    }

    /// Compute HDC-based cohesion: average pairwise entity similarity
    fn compute_hdc_cohesion(&self, entities: &[&Entity]) -> f32 {
        if entities.len() < 2 {
            return 1.0;
        }

        let hvs: Vec<ContinuousHV> = entities
            .iter()
            .map(|e| self.encoder.encode_entity(e))
            .collect();

        let mut total_sim = 0.0f32;
        let mut count = 0u32;

        for i in 0..hvs.len() {
            for j in (i + 1)..hvs.len() {
                total_sim += hvs[i].similarity(&hvs[j]);
                count += 1;
            }
        }

        if count > 0 {
            (total_sim / count as f32).max(0.0)
        } else {
            1.0
        }
    }

    /// Count connected components in the adjacency graph
    fn count_components(&self, adjacency: &HashMap<String, Vec<String>>) -> usize {
        let mut visited: HashMap<&str, bool> = HashMap::new();
        let mut components = 0;

        for node in adjacency.keys() {
            if !visited.contains_key(node.as_str()) {
                components += 1;
                // DFS
                let mut stack = vec![node.as_str()];
                while let Some(current) = stack.pop() {
                    if visited.contains_key(current) {
                        continue;
                    }
                    visited.insert(current, true);

                    if let Some(neighbors) = adjacency.get(current) {
                        for neighbor in neighbors {
                            if !visited.contains_key(neighbor.as_str()) {
                                stack.push(neighbor.as_str());
                            }
                        }
                    }
                }
            }
        }

        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_parser::{CodeStructure, EntityKind, EntityRelation, Relation, Span};

    fn test_span() -> Span {
        Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        }
    }

    #[test]
    fn test_empty_code_phi() {
        let analyzer = PhiCodeAnalyzer::default_dim();
        let parsed = ParsedCode::new("", "rust");
        let score = analyzer.measure_code_phi(&parsed);
        assert_eq!(score.entity_count, 0);
        // Empty code has 0 integration/cohesion/connectivity but 0 dead code
        // Formula: 0*0.3 + 0*0.3 + 0*0.2 + (1-0)*0.2 = 0.2
        assert!((score.quality_score() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_single_function() {
        let analyzer = PhiCodeAnalyzer::default_dim();
        let mut parsed = ParsedCode::new("fn main() {}", "rust");
        parsed
            .entities
            .push(Entity::new(EntityKind::Function, "main", test_span()));

        let score = analyzer.measure_code_phi(&parsed);
        assert_eq!(score.entity_count, 1);
        assert!(score.cohesion > 0.0);
    }

    #[test]
    fn test_connected_code_higher_phi() {
        let analyzer = PhiCodeAnalyzer::default_dim();

        // Connected code: A calls B
        let mut connected = ParsedCode::new("", "rust");
        connected
            .entities
            .push(Entity::new(EntityKind::Function, "process", test_span()));
        connected
            .entities
            .push(Entity::new(EntityKind::Function, "validate", test_span()));
        connected.structure.relations.push(EntityRelation {
            source: "process".to_string(),
            relation: Relation::Calls,
            target: "validate".to_string(),
        });

        // Disconnected code: A and B are unrelated
        let mut disconnected = ParsedCode::new("", "rust");
        disconnected
            .entities
            .push(Entity::new(EntityKind::Function, "foo", test_span()));
        disconnected
            .entities
            .push(Entity::new(EntityKind::Struct, "Bar", test_span()));

        let connected_phi = analyzer.measure_code_phi(&connected);
        let disconnected_phi = analyzer.measure_code_phi(&disconnected);

        // Connected code should have higher connectivity
        assert!(
            connected_phi.connectivity >= disconnected_phi.connectivity,
            "Connected code should have higher connectivity: {} vs {}",
            connected_phi.connectivity,
            disconnected_phi.connectivity
        );
    }

    #[test]
    fn test_phi_diff_improvement() {
        let analyzer = PhiCodeAnalyzer::default_dim();

        // Before: disconnected
        let mut before = ParsedCode::new("", "rust");
        before
            .entities
            .push(Entity::new(EntityKind::Function, "a", test_span()));
        before
            .entities
            .push(Entity::new(EntityKind::Function, "b", test_span()));

        // After: connected (a calls b)
        let mut after = ParsedCode::new("", "rust");
        after
            .entities
            .push(Entity::new(EntityKind::Function, "a", test_span()));
        after
            .entities
            .push(Entity::new(EntityKind::Function, "b", test_span()));
        after.structure.relations.push(EntityRelation {
            source: "a".to_string(),
            relation: Relation::Calls,
            target: "b".to_string(),
        });

        let diff = analyzer.phi_diff(&before, &after);
        assert!(diff.after.connectivity >= diff.before.connectivity);
    }

    #[test]
    fn test_quality_levels() {
        let score_excellent = PhiCodeScore {
            integration: 0.9,
            cohesion: 0.9,
            connectivity: 0.9,
            dead_code_ratio: 0.0,
            component_count: 1,
            entity_count: 10,
        };
        assert!(score_excellent.quality_level().contains("Excellent"));

        let score_poor = PhiCodeScore {
            integration: 0.1,
            cohesion: 0.1,
            connectivity: 0.1,
            dead_code_ratio: 0.8,
            component_count: 5,
            entity_count: 10,
        };
        assert!(
            score_poor.quality_level().contains("Poor")
                || score_poor.quality_level().contains("Critical")
        );
    }
}
