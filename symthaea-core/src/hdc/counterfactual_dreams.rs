// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual Dreams Integration
//!
//! Bridges counterfactual reasoning with dream generation to enable:
//! - Dreams that explore "what-if" scenarios
//! - Causal intervention simulation during sleep
//! - Memory consolidation through counterfactual replay
//!
//! # Theoretical Foundation
//!
//! Dreams have been theorized to serve functions including:
//! - **Memory Consolidation**: Replaying and strengthening memories
//! - **Emotional Processing**: Working through emotional experiences
//! - **Problem Solving**: Exploring solutions without constraints
//! - **Counterfactual Simulation**: Testing alternative outcomes
//!
//! This module combines Pearl's do-calculus counterfactuals with HDC dream
//! generation to create a system that can:
//! 1. Generate dream scenarios that explore counterfactual outcomes
//! 2. Use causal models to guide dream narrative
//! 3. Consolidate counterfactual insights into memory
//!
//! # Architecture
//!
//! ```text
//!         ┌─────────────────────────────────────────────────┐
//!         │         COUNTERFACTUAL DREAM ENGINE              │
//!         ├─────────────────────────────────────────────────┤
//!         │                                                 │
//!         │    ┌─────────────┐      ┌─────────────┐        │
//!         │    │ Causal Mind │─────►│ Counterfact │        │
//!         │    │  (do-calc)  │      │  Scenarios  │        │
//!         │    └─────────────┘      └──────┬──────┘        │
//!         │                                │               │
//!         │                       ┌────────▼────────┐      │
//!         │    ┌─────────────┐    │ Dream Generator │      │
//!         │    │   Memory    │───►│  (HDC-based)   │      │
//!         │    │  Fragments  │    └────────┬────────┘      │
//!         │    └─────────────┘             │               │
//!         │                                ▼               │
//!         │                     ┌──────────────────┐       │
//!         │                     │  Dream Scenario  │       │
//!         │                     │ (what-if dreams) │       │
//!         │                     └──────────────────┘       │
//!         └─────────────────────────────────────────────────┘
//! ```
//!
//! # Example Usage
//!
//! ```ignore
//! let mut engine = CounterfactualDreamEngine::new();
//!
//! // Add a memory and its counterfactual
//! engine.add_memory_with_counterfactual(
//!     "interview",
//!     interview_hv,
//!     "What if I had prepared more?",
//!     prepared_hv,
//!     0.7,  // Emotional intensity
//! );
//!
//! // Generate a counterfactual dream
//! let dream = engine.generate_counterfactual_dream(5.0); // 5 minute dream
//! println!("Dream explored: {}", dream.counterfactual_theme);
//! ```

use super::binary_hv::BinaryHV;
use super::sleep_and_altered_states::{DreamFragment, DreamFragmentSource, DreamScenario};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// COUNTERFACTUAL MEMORY
// ============================================================================

/// A memory with its associated counterfactual
#[derive(Debug, Clone)]
pub struct CounterfactualMemory {
    /// Unique identifier
    pub id: u64,
    /// Label/description of the memory
    pub label: String,
    /// HDC representation of actual outcome
    pub actual_hv: BinaryHV,
    /// Description of the counterfactual scenario
    pub counterfactual_question: String,
    /// HDC representation of counterfactual outcome
    pub counterfactual_hv: BinaryHV,
    /// Emotional intensity (affects dream probability)
    pub emotional_intensity: f64,
    /// Valence of the counterfactual (-1 = regret, +1 = hope)
    pub valence: f64,
    /// How many times this has been dreamed about
    pub dream_count: usize,
    /// Last dream timestamp
    pub last_dreamed: Option<u64>,
}

impl CounterfactualMemory {
    pub fn new(
        id: u64,
        label: &str,
        actual: BinaryHV,
        question: &str,
        counterfactual: BinaryHV,
        intensity: f64,
    ) -> Self {
        Self {
            id,
            label: label.to_string(),
            actual_hv: actual,
            counterfactual_question: question.to_string(),
            counterfactual_hv: counterfactual,
            emotional_intensity: intensity.clamp(0.0, 1.0),
            valence: 0.0,
            dream_count: 0,
            last_dreamed: None,
        }
    }

    pub fn with_valence(mut self, valence: f64) -> Self {
        self.valence = valence.clamp(-1.0, 1.0);
        self
    }

    /// Dream weight (probability of appearing in dreams)
    pub fn dream_weight(&self) -> f64 {
        // High emotion = more likely to dream
        // Frequent dreaming = less likely (habituation)
        let habituation = 1.0 / (1.0 + self.dream_count as f64 * 0.2);
        self.emotional_intensity * habituation
    }
}

// ============================================================================
// COUNTERFACTUAL DREAM FRAGMENT
// ============================================================================

/// A dream fragment that represents a counterfactual exploration
#[derive(Debug, Clone)]
pub struct CounterfactualDreamFragment {
    /// The base dream fragment
    pub base_fragment: DreamFragment,
    /// The counterfactual being explored
    pub counterfactual_question: String,
    /// Is this the actual or counterfactual version?
    pub is_counterfactual: bool,
    /// Similarity to the actual outcome (0 = very different)
    pub divergence: f64,
    /// Source memory ID
    pub source_memory_id: u64,
}

// ============================================================================
// COUNTERFACTUAL DREAM SCENARIO
// ============================================================================

/// A complete dream scenario exploring counterfactuals
#[derive(Debug, Clone)]
pub struct CounterfactualDreamScenario {
    /// Base dream scenario
    pub base_scenario: DreamScenario,
    /// The main counterfactual theme
    pub counterfactual_theme: String,
    /// Source memories involved
    pub source_memories: Vec<u64>,
    /// Counterfactual fragments
    pub counterfactual_fragments: Vec<CounterfactualDreamFragment>,
    /// Resolution type
    pub resolution: DreamResolution,
    /// Insight generated (if any)
    pub insight: Option<String>,
}

/// How the counterfactual dream resolved
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DreamResolution {
    /// Dreamer accepted both outcomes
    Acceptance,
    /// Dreamer chose counterfactual as preferred
    CounterfactualPreferred,
    /// Dreamer chose actual as preferred
    ActualPreferred,
    /// Dream remained unresolved
    Unresolved,
    /// Dream led to new insight
    InsightGenerated,
}

// ============================================================================
// COUNTERFACTUAL DREAM ENGINE
// ============================================================================

/// Engine for generating counterfactual dreams
pub struct CounterfactualDreamEngine {
    /// Stored counterfactual memories
    memories: HashMap<u64, CounterfactualMemory>,
    /// Next memory ID
    next_id: u64,
    /// Random seed
    seed: u64,
    /// Dream history
    dream_history: Vec<CounterfactualDreamScenario>,
    /// Maximum history length
    max_history: usize,
    /// Bizarreness factor for dreams (0 = realistic, 1 = very bizarre)
    bizarreness: f64,
}

impl CounterfactualDreamEngine {
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
            next_id: 1,
            seed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42),
            dream_history: Vec::new(),
            max_history: 50,
            bizarreness: 0.5,
        }
    }

    /// Add a memory with its counterfactual
    pub fn add_memory_with_counterfactual(
        &mut self,
        label: &str,
        actual: BinaryHV,
        question: &str,
        counterfactual: BinaryHV,
        emotional_intensity: f64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let memory = CounterfactualMemory::new(
            id,
            label,
            actual,
            question,
            counterfactual,
            emotional_intensity,
        );

        self.memories.insert(id, memory);
        id
    }

    /// Add a memory with valence
    pub fn add_memory_with_valence(
        &mut self,
        label: &str,
        actual: BinaryHV,
        question: &str,
        counterfactual: BinaryHV,
        emotional_intensity: f64,
        valence: f64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let memory = CounterfactualMemory::new(
            id,
            label,
            actual,
            question,
            counterfactual,
            emotional_intensity,
        )
        .with_valence(valence);

        self.memories.insert(id, memory);
        id
    }

    /// Set bizarreness factor
    pub fn set_bizarreness(&mut self, bizarreness: f64) {
        self.bizarreness = bizarreness.clamp(0.0, 1.0);
    }

    /// Generate a counterfactual dream scenario
    pub fn generate_counterfactual_dream(
        &mut self,
        duration_minutes: f64,
    ) -> CounterfactualDreamScenario {
        // Select memories to include based on dream weight
        let selected = self.select_dream_memories(3);

        if selected.is_empty() {
            return self.empty_dream(duration_minutes);
        }

        // Generate fragments
        let num_fragments = (duration_minutes * 2.0) as usize;
        let mut fragments = Vec::new();
        let mut cf_fragments = Vec::new();

        for i in 0..num_fragments {
            let seed = self.seed.wrapping_add(i as u64);
            self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);

            // Pick a memory
            let memory_idx = i % selected.len();
            let memory_id = selected[memory_idx];
            let memory = self.memories.get(&memory_id).unwrap();

            // Decide if this fragment shows actual or counterfactual
            let is_counterfactual = (seed % 100) < 60; // 60% counterfactual

            let hv = if is_counterfactual {
                self.generate_bizarre_blend(&memory.counterfactual_hv, seed)
            } else {
                self.generate_bizarre_blend(&memory.actual_hv, seed)
            };

            // Calculate divergence (cast f32 similarity to f64)
            let divergence = 1.0 - memory.actual_hv.similarity(&memory.counterfactual_hv) as f64;

            let base_fragment = DreamFragment {
                content_hv: hv,
                description: if is_counterfactual {
                    format!("[CF] {}: {}", memory.label, memory.counterfactual_question)
                } else {
                    format!("[ACTUAL] {}", memory.label)
                },
                valence: memory.valence * if is_counterfactual { 1.2 } else { 0.8 },
                bizarreness: self.bizarreness + (seed % 30) as f64 / 100.0,
                source: if is_counterfactual {
                    DreamFragmentSource::NarrativeFill
                } else {
                    DreamFragmentSource::MemoryReplay
                },
            };

            let cf_fragment = CounterfactualDreamFragment {
                base_fragment: base_fragment.clone(),
                counterfactual_question: memory.counterfactual_question.clone(),
                is_counterfactual,
                divergence,
                source_memory_id: memory_id,
            };

            fragments.push(base_fragment);
            cf_fragments.push(cf_fragment);
        }

        // Determine resolution
        let resolution = self.determine_resolution(&selected);

        // Generate insight if applicable
        let insight = if resolution == DreamResolution::InsightGenerated {
            self.generate_insight(&selected)
        } else {
            None
        };

        // Build main counterfactual theme
        let main_memory = self.memories.get(&selected[0]).unwrap();
        let theme = format!("Exploring: {}", main_memory.counterfactual_question);

        // Update dream counts
        for id in &selected {
            if let Some(m) = self.memories.get_mut(id) {
                m.dream_count += 1;
                m.last_dreamed = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                );
            }
        }

        // Compute overall valence and coherence
        let emotional_valence = cf_fragments
            .iter()
            .map(|f| f.base_fragment.valence)
            .sum::<f64>()
            / cf_fragments.len().max(1) as f64;

        let narrative_coherence = 1.0 - self.bizarreness * 0.5;

        let base_scenario = DreamScenario {
            fragments,
            emotional_valence,
            narrative_coherence,
            lucidity: 0.2, // Not lucid by default
            themes: vec![theme.clone()],
            subjective_duration: duration_minutes,
        };

        let scenario = CounterfactualDreamScenario {
            base_scenario,
            counterfactual_theme: theme,
            source_memories: selected,
            counterfactual_fragments: cf_fragments,
            resolution,
            insight,
        };

        // Store in history
        self.dream_history.push(scenario.clone());
        if self.dream_history.len() > self.max_history {
            self.dream_history.remove(0);
        }

        scenario
    }

    /// Generate a lucid counterfactual dream (higher awareness)
    pub fn generate_lucid_counterfactual_dream(
        &mut self,
        duration_minutes: f64,
        focus_memory_id: Option<u64>,
    ) -> CounterfactualDreamScenario {
        // Lower bizarreness for lucid dreams
        let old_bizarreness = self.bizarreness;
        self.bizarreness *= 0.5;

        let mut dream = self.generate_counterfactual_dream(duration_minutes);

        // Modify for lucidity
        dream.base_scenario.lucidity = 0.8;
        dream.base_scenario.narrative_coherence *= 1.3;

        // If focused on specific memory, emphasize it
        if let Some(id) = focus_memory_id {
            if let Some(memory) = self.memories.get(&id) {
                dream.counterfactual_theme =
                    format!("Lucid exploration: {}", memory.counterfactual_question);
            }
        }

        self.bizarreness = old_bizarreness;
        dream
    }

    /// Generate a nightmare based on negative counterfactuals
    pub fn generate_counterfactual_nightmare(
        &mut self,
        duration_minutes: f64,
    ) -> CounterfactualDreamScenario {
        // Select only negative valence memories
        let negative_ids: Vec<u64> = self
            .memories
            .iter()
            .filter(|(_, m)| m.valence < 0.0)
            .map(|(id, _)| *id)
            .collect();

        if negative_ids.is_empty() {
            return self.empty_dream(duration_minutes);
        }

        // Increase bizarreness
        let old_bizarreness = self.bizarreness;
        self.bizarreness = 0.8;

        let mut dream = self.generate_counterfactual_dream(duration_minutes);

        // Make it more nightmarish
        dream.base_scenario.emotional_valence = -0.7;
        dream.base_scenario.narrative_coherence *= 0.7;
        dream.counterfactual_theme = format!(
            "Nightmare: What if the worst happened? - {}",
            dream.counterfactual_theme
        );

        self.bizarreness = old_bizarreness;
        dream
    }

    /// Select memories for dreaming based on weights
    fn select_dream_memories(&self, max_count: usize) -> Vec<u64> {
        let mut weighted: Vec<_> = self
            .memories
            .iter()
            .map(|(id, m)| (*id, m.dream_weight()))
            .collect();

        // Sort by weight
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        weighted
            .into_iter()
            .take(max_count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Generate bizarre blend of HV (dream distortion)
    fn generate_bizarre_blend(&self, hv: &BinaryHV, seed: u64) -> BinaryHV {
        if self.bizarreness < 0.1 {
            return *hv;
        }

        // Create a random HV for blending
        let noise = BinaryHV::random(seed);

        // Blend based on bizarreness
        Self::weighted_bundle(&[(*hv, 1.0 - self.bizarreness), (noise, self.bizarreness)])
    }

    /// Weighted bundle operation
    fn weighted_bundle(weighted_hvs: &[(BinaryHV, f64)]) -> BinaryHV {
        if weighted_hvs.is_empty() {
            return BinaryHV::random(42);
        }

        const DIMENSIONS: usize = 16384;
        let mut accumulator = vec![0.0f64; DIMENSIONS];

        for (hv, weight) in weighted_hvs {
            for (i, byte) in hv.0.iter().enumerate() {
                for bit in 0..8 {
                    let idx = i * 8 + bit;
                    if idx < DIMENSIONS {
                        let bit_val = (byte >> (7 - bit)) & 1;
                        accumulator[idx] += if bit_val == 1 { *weight } else { -*weight };
                    }
                }
            }
        }

        let mut result = [0u8; 2048];
        for (i, &sum) in accumulator.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            if sum > 0.0 {
                result[byte_idx] |= 1 << bit_idx;
            }
        }

        BinaryHV(result)
    }

    /// Determine how the dream resolves
    fn determine_resolution(&self, memory_ids: &[u64]) -> DreamResolution {
        if memory_ids.is_empty() {
            return DreamResolution::Unresolved;
        }

        // Analyze valences of involved memories
        let avg_valence: f64 = memory_ids
            .iter()
            .filter_map(|id| self.memories.get(id))
            .map(|m| m.valence)
            .sum::<f64>()
            / memory_ids.len() as f64;

        // Use seed for some randomness
        let roll = (self.seed % 100) as f64 / 100.0;

        if roll < 0.15 {
            DreamResolution::InsightGenerated
        } else if avg_valence < -0.3 {
            DreamResolution::ActualPreferred
        } else if avg_valence > 0.3 {
            DreamResolution::CounterfactualPreferred
        } else if roll < 0.5 {
            DreamResolution::Acceptance
        } else {
            DreamResolution::Unresolved
        }
    }

    /// Generate insight from dream
    fn generate_insight(&self, memory_ids: &[u64]) -> Option<String> {
        if memory_ids.is_empty() {
            return None;
        }

        let memory = self.memories.get(&memory_ids[0])?;

        Some(format!(
            "Insight from counterfactual exploration: The question '{}' reveals that what matters is not the outcome, but the learning from the experience.",
            memory.counterfactual_question
        ))
    }

    /// Create empty dream
    fn empty_dream(&self, duration: f64) -> CounterfactualDreamScenario {
        CounterfactualDreamScenario {
            base_scenario: DreamScenario {
                fragments: vec![],
                emotional_valence: 0.0,
                narrative_coherence: 0.5,
                lucidity: 0.1,
                themes: vec!["Empty dream".to_string()],
                subjective_duration: duration,
            },
            counterfactual_theme: "No counterfactuals to explore".to_string(),
            source_memories: vec![],
            counterfactual_fragments: vec![],
            resolution: DreamResolution::Unresolved,
            insight: None,
        }
    }

    /// Get memory by ID
    pub fn get_memory(&self, id: u64) -> Option<&CounterfactualMemory> {
        self.memories.get(&id)
    }

    /// Get all memories
    pub fn all_memories(&self) -> Vec<&CounterfactualMemory> {
        self.memories.values().collect()
    }

    /// Get dream history
    pub fn dream_history(&self) -> &[CounterfactualDreamScenario] {
        &self.dream_history
    }

    /// Get memory count
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Generate report of counterfactual dreaming activity
    pub fn report(&self) -> String {
        let total_dreams: usize = self.memories.values().map(|m| m.dream_count).sum();

        let most_dreamed = self
            .memories
            .values()
            .max_by_key(|m| m.dream_count)
            .map(|m| format!("'{}' ({} times)", m.label, m.dream_count))
            .unwrap_or_else(|| "None".to_string());

        format!(
            "=== Counterfactual Dream Report ===\n\
             Memories stored: {}\n\
             Total dreams: {}\n\
             Dream history: {} scenarios\n\
             Most dreamed: {}\n\
             Current bizarreness: {:.2}",
            self.memories.len(),
            total_dreams,
            self.dream_history.len(),
            most_dreamed,
            self.bizarreness
        )
    }
}

impl Default for CounterfactualDreamEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = CounterfactualDreamEngine::new();
        assert_eq!(engine.memory_count(), 0);
    }

    #[test]
    fn test_add_memory() {
        let mut engine = CounterfactualDreamEngine::new();

        let id = engine.add_memory_with_counterfactual(
            "test_memory",
            BinaryHV::random(1),
            "What if I had done X?",
            BinaryHV::random(2),
            0.7,
        );

        assert!(id > 0);
        assert_eq!(engine.memory_count(), 1);
    }

    #[test]
    fn test_generate_dream() {
        let mut engine = CounterfactualDreamEngine::new();

        engine.add_memory_with_counterfactual(
            "decision",
            BinaryHV::random(1),
            "What if I had chosen differently?",
            BinaryHV::random(2),
            0.8,
        );

        let dream = engine.generate_counterfactual_dream(5.0);

        assert!(!dream.counterfactual_theme.is_empty());
        assert!(!dream.source_memories.is_empty());
    }

    #[test]
    fn test_lucid_dream() {
        let mut engine = CounterfactualDreamEngine::new();

        let id = engine.add_memory_with_counterfactual(
            "test",
            BinaryHV::random(1),
            "What if?",
            BinaryHV::random(2),
            0.5,
        );

        let dream = engine.generate_lucid_counterfactual_dream(5.0, Some(id));

        assert!(dream.base_scenario.lucidity > 0.5);
    }

    #[test]
    fn test_dream_weight() {
        let memory = CounterfactualMemory::new(
            1,
            "test",
            BinaryHV::random(1),
            "What if?",
            BinaryHV::random(2),
            0.9,
        );

        assert!(memory.dream_weight() > 0.5);
    }

    #[test]
    fn test_report() {
        let mut engine = CounterfactualDreamEngine::new();

        engine.add_memory_with_counterfactual(
            "memory1",
            BinaryHV::random(1),
            "What if 1?",
            BinaryHV::random(2),
            0.5,
        );

        let report = engine.report();
        assert!(report.contains("Counterfactual Dream Report"));
    }
}
