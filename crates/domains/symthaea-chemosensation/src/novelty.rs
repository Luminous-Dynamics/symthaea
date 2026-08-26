// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Traceable novelty memory for chemical percepts.
//!
//! Novelty is assessed against previously admitted fingerprints from the same
//! chemical modality. Assessment and memory admission are separate operations
//! so a single anomalous or low-confidence exposure does not automatically
//! become the new normal.

use std::collections::{HashMap, VecDeque};

use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::{ChemicalModality, ChemicalPercept};

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalNoveltyConfig {
    /// Maximum retained references per modality.
    pub capacity_per_modality: usize,
    /// Minimum percept confidence required for memory admission.
    pub min_admission_confidence: f32,
    /// Similarity at or above which a percept is treated as already represented.
    pub duplicate_similarity: f32,
}

impl Default for ChemicalNoveltyConfig {
    fn default() -> Self {
        Self {
            capacity_per_modality: 64,
            min_admission_confidence: 0.6,
            duplicate_similarity: 0.98,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NoveltyConfigError {
    ZeroCapacity,
    InvalidMinimumConfidence(f32),
    InvalidDuplicateSimilarity(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalMemoryReference {
    pub similarity: f32,
    pub timestamp_us: u64,
    pub source: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoveltyAssessment {
    /// Bounded novelty in [0, 1]. One means no similar admitted reference.
    pub novelty: f32,
    pub nearest: Option<ChemicalMemoryReference>,
    pub memory_size: usize,
}

#[derive(Debug, Clone)]
struct MemoryEntry {
    vector: ContinuousHV,
    timestamp_us: u64,
    source: String,
    confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ChemicalNoveltyMemory {
    config: ChemicalNoveltyConfig,
    entries: HashMap<ChemicalModality, VecDeque<MemoryEntry>>,
}

impl ChemicalNoveltyMemory {
    pub fn new(config: ChemicalNoveltyConfig) -> Result<Self, NoveltyConfigError> {
        if config.capacity_per_modality == 0 {
            return Err(NoveltyConfigError::ZeroCapacity);
        }
        if !config.min_admission_confidence.is_finite()
            || !(0.0..=1.0).contains(&config.min_admission_confidence)
        {
            return Err(NoveltyConfigError::InvalidMinimumConfidence(
                config.min_admission_confidence,
            ));
        }
        if !config.duplicate_similarity.is_finite()
            || !(0.0..=1.0).contains(&config.duplicate_similarity)
        {
            return Err(NoveltyConfigError::InvalidDuplicateSimilarity(
                config.duplicate_similarity,
            ));
        }

        Ok(Self {
            config,
            entries: HashMap::new(),
        })
    }

    pub fn config(&self) -> &ChemicalNoveltyConfig {
        &self.config
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self, modality: ChemicalModality) -> usize {
        self.entries.get(&modality).map_or(0, VecDeque::len)
    }

    /// Assess novelty without changing memory.
    pub fn assess(&self, percept: &ChemicalPercept) -> NoveltyAssessment {
        let modality = percept.evidence.modality;
        let Some(entries) = self.entries.get(&modality) else {
            return NoveltyAssessment {
                novelty: 1.0,
                nearest: None,
                memory_size: 0,
            };
        };

        let mut nearest: Option<ChemicalMemoryReference> = None;
        for entry in entries {
            let similarity = entry
                .vector
                .similarity(&percept.fingerprint.vector)
                .clamp(0.0, 1.0);
            let replace = nearest
                .as_ref()
                .is_none_or(|current| similarity > current.similarity);
            if replace {
                nearest = Some(ChemicalMemoryReference {
                    similarity,
                    timestamp_us: entry.timestamp_us,
                    source: entry.source.clone(),
                    confidence: entry.confidence,
                });
            }
        }

        let novelty = nearest
            .as_ref()
            .map_or(1.0, |reference| 1.0 - reference.similarity);

        NoveltyAssessment {
            novelty,
            nearest,
            memory_size: entries.len(),
        }
    }

    /// Admit a percept to novelty memory when it is trustworthy and not already
    /// represented. Returns true only when a new reference was stored.
    pub fn admit(&mut self, percept: &ChemicalPercept) -> bool {
        if percept.confidence() < self.config.min_admission_confidence {
            return false;
        }

        let assessment = self.assess(percept);
        if assessment
            .nearest
            .as_ref()
            .is_some_and(|nearest| nearest.similarity >= self.config.duplicate_similarity)
        {
            return false;
        }

        let modality = percept.evidence.modality;
        let entries = self.entries.entry(modality).or_default();
        if entries.len() >= self.config.capacity_per_modality {
            entries.pop_front();
        }
        entries.push_back(MemoryEntry {
            vector: percept.fingerprint.vector.clone(),
            timestamp_us: percept.timestamp_us(),
            source: percept.evidence.source.clone(),
            confidence: percept.confidence(),
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalFingerprint, ChemicalObservation, EnvironmentReading};
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};

    fn percept(
        modality: ChemicalModality,
        timestamp_us: u64,
        seed: u64,
        confidence: f32,
    ) -> ChemicalPercept {
        ChemicalPercept {
            evidence: ChemicalObservation {
                timestamp_us,
                modality,
                source: format!("source-{seed}"),
                channels: vec![],
                environment: EnvironmentReading::default(),
            },
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence,
                used_channels: 1,
                ignored_channels: 0,
            },
        }
    }

    #[test]
    fn unseen_percept_is_maximally_novel_without_mutating_memory() {
        let memory = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        let candidate = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let assessment = memory.assess(&candidate);
        assert_eq!(assessment.novelty, 1.0);
        assert!(assessment.nearest.is_none());
        assert_eq!(memory.len(ChemicalModality::Olfactory), 0);
    }

    #[test]
    fn admitted_reference_is_traceable() {
        let mut memory = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        let reference = percept(ChemicalModality::Olfactory, 123, 1, 0.9);
        assert!(memory.admit(&reference));
        let assessment = memory.assess(&reference);
        let nearest = assessment.nearest.unwrap();
        assert!(assessment.novelty < 1e-6);
        assert_eq!(nearest.timestamp_us, 123);
        assert_eq!(nearest.source, "source-1");
        assert!((nearest.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn low_confidence_percept_is_not_learned() {
        let mut memory = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        let weak = percept(ChemicalModality::Olfactory, 1, 1, 0.2);
        assert!(!memory.admit(&weak));
        assert_eq!(memory.len(ChemicalModality::Olfactory), 0);
    }

    #[test]
    fn duplicate_reference_does_not_consume_capacity() {
        let mut memory = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        let reference = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let mut duplicate = reference.clone();
        duplicate.evidence.timestamp_us = 2;
        assert!(memory.admit(&reference));
        assert!(!memory.admit(&duplicate));
        assert_eq!(memory.len(ChemicalModality::Olfactory), 1);
    }

    #[test]
    fn smell_and_taste_novelty_memories_are_separate() {
        let mut memory = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let taste = percept(ChemicalModality::Gustatory, 2, 1, 0.9);
        assert!(memory.admit(&odor));
        assert_eq!(memory.assess(&taste).novelty, 1.0);
    }

    #[test]
    fn memory_is_bounded_per_modality() {
        let config = ChemicalNoveltyConfig {
            capacity_per_modality: 2,
            min_admission_confidence: 0.5,
            duplicate_similarity: 1.0,
        };
        let mut memory = ChemicalNoveltyMemory::new(config).unwrap();
        assert!(memory.admit(&percept(ChemicalModality::Olfactory, 1, 1, 0.9)));
        assert!(memory.admit(&percept(ChemicalModality::Olfactory, 2, 2, 0.9)));
        assert!(memory.admit(&percept(ChemicalModality::Olfactory, 3, 3, 0.9)));
        assert_eq!(memory.len(ChemicalModality::Olfactory), 2);
    }
}
