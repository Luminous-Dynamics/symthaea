// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Traceable novelty memory for chemical percepts.
//!
//! Novelty is assessed against previously admitted fingerprints from the same
//! chemical modality and the same HDC encoding space. Assessment and memory
//! admission are separate operations so a single anomalous or low-confidence
//! exposure does not automatically become the new normal.
//!
//! Representation migrations are bounded explicitly. Each modality retains a
//! fixed number of encoding spaces, and each space retains a fixed number of
//! references. A stream of new encoder versions therefore cannot grow novelty
//! memory without limit.

use std::collections::{HashMap, VecDeque};

use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::{ChemicalClockDomainId, ChemicalEncodingSpaceId, ChemicalModality, ChemicalPercept};

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalNoveltyConfig {
    /// Maximum retained references within one modality+encoding-space namespace.
    pub capacity_per_space: usize,
    /// Maximum retained encoding-space generations for one modality. When a new
    /// generation exceeds this limit, the oldest retained space is evicted in
    /// full before the new reference is admitted.
    pub max_spaces_per_modality: usize,
    /// Minimum percept confidence required for memory admission.
    pub min_admission_confidence: f32,
    /// Similarity at or above which a percept is treated as already represented.
    pub duplicate_similarity: f32,
}

impl Default for ChemicalNoveltyConfig {
    fn default() -> Self {
        Self {
            capacity_per_space: 64,
            max_spaces_per_modality: 4,
            min_admission_confidence: 0.6,
            duplicate_similarity: 0.98,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NoveltyConfigError {
    ZeroSpaceCapacity,
    ZeroEncodingSpaceHistory,
    InvalidMinimumConfidence(f32),
    InvalidDuplicateSimilarity(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalMemoryReference {
    pub similarity: f32,
    pub timestamp_us: u64,
    /// Clock domain attached to `timestamp_us`, when one was declared by the
    /// admitted evidence. Novelty does not compare timestamps; this is retained
    /// strictly so the reference remains traceable without inventing an epoch.
    pub clock_domain: Option<ChemicalClockDomainId>,
    pub source: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoveltyAssessment {
    /// Bounded novelty in [0, 1]. One means no similar admitted reference in
    /// the current modality+encoding-space namespace.
    pub novelty: f32,
    pub nearest: Option<ChemicalMemoryReference>,
    pub memory_size: usize,
}

#[derive(Debug, Clone)]
struct MemoryEntry {
    vector: ContinuousHV,
    timestamp_us: u64,
    clock_domain: Option<ChemicalClockDomainId>,
    source: String,
    confidence: f32,
}

type MemoryKey = (ChemicalModality, ChemicalEncodingSpaceId);

#[derive(Debug, Clone)]
pub struct ChemicalNoveltyMemory {
    config: ChemicalNoveltyConfig,
    entries: HashMap<MemoryKey, VecDeque<MemoryEntry>>,
    /// Oldest retained encoding space at the front for each modality.
    space_order: HashMap<ChemicalModality, VecDeque<ChemicalEncodingSpaceId>>,
}

impl ChemicalNoveltyMemory {
    pub fn new(config: ChemicalNoveltyConfig) -> Result<Self, NoveltyConfigError> {
        if config.capacity_per_space == 0 {
            return Err(NoveltyConfigError::ZeroSpaceCapacity);
        }
        if config.max_spaces_per_modality == 0 {
            return Err(NoveltyConfigError::ZeroEncodingSpaceHistory);
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
            space_order: HashMap::new(),
        })
    }

    pub fn config(&self) -> &ChemicalNoveltyConfig {
        &self.config
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.space_order.clear();
    }

    /// Total retained references for one modality across all retained
    /// representation generations. Geometric comparisons remain space-local.
    pub fn len(&self, modality: ChemicalModality) -> usize {
        self.entries
            .iter()
            .filter(|((stored_modality, _), _)| *stored_modality == modality)
            .map(|(_, entries)| entries.len())
            .sum()
    }

    pub fn len_in_space(
        &self,
        modality: ChemicalModality,
        encoding_space_id: ChemicalEncodingSpaceId,
    ) -> usize {
        self.entries
            .get(&(modality, encoding_space_id))
            .map_or(0, VecDeque::len)
    }

    pub fn retained_space_count(&self, modality: ChemicalModality) -> usize {
        self.space_order.get(&modality).map_or(0, VecDeque::len)
    }

    /// Assess novelty without changing memory.
    pub fn assess(&self, percept: &ChemicalPercept) -> NoveltyAssessment {
        let key = (
            percept.evidence.modality,
            percept.fingerprint.encoding_space_id,
        );
        let Some(entries) = self.entries.get(&key) else {
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
                    clock_domain: entry.clock_domain.clone(),
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
    /// represented in the same encoding space. Returns true only when a new
    /// reference was stored.
    ///
    /// New encoding-space generations are registered transactionally before the
    /// reference is inserted. If the per-modality generation history is full,
    /// the oldest space and all of its references are removed together.
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
        let encoding_space_id = percept.fingerprint.encoding_space_id;
        let key = (modality, encoding_space_id);

        if !self.entries.contains_key(&key) {
            let order = self.space_order.entry(modality).or_default();
            if order.len() >= self.config.max_spaces_per_modality {
                if let Some(evicted_space) = order.pop_front() {
                    self.entries.remove(&(modality, evicted_space));
                }
            }
            order.push_back(encoding_space_id);
        }

        let entries = self.entries.entry(key).or_default();
        if entries.len() >= self.config.capacity_per_space {
            entries.pop_front();
        }
        entries.push_back(MemoryEntry {
            vector: percept.fingerprint.vector.clone(),
            timestamp_us: percept.timestamp_us(),
            clock_domain: percept.evidence.clock_domain.clone(),
            source: percept.evidence.source.clone(),
            confidence: percept.confidence(),
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalFingerprint, ChemicalObservation};
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};

    fn test_clock() -> ChemicalClockDomainId {
        ChemicalClockDomainId::new("test-rig/monotonic").unwrap()
    }

    fn percept(
        modality: ChemicalModality,
        timestamp_us: u64,
        seed: u64,
        confidence: f32,
    ) -> ChemicalPercept {
        ChemicalPercept {
            evidence: ChemicalObservation::new(
                timestamp_us,
                modality,
                format!("source-{seed}"),
                vec![],
            )
            .with_clock_domain(test_clock()),
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([7; 32]),
            },
        }
    }

    fn in_space(mut percept: ChemicalPercept, byte: u8) -> ChemicalPercept {
        percept.fingerprint.encoding_space_id = ChemicalEncodingSpaceId::from_bytes([byte; 32]);
        percept
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
        assert_eq!(nearest.clock_domain, Some(test_clock()));
        assert_eq!(nearest.source, "source-1");
        assert!((nearest.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn unclocked_reference_remains_traceable_without_inventing_epoch() {
        let mut memory = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        let mut reference = percept(ChemicalModality::Olfactory, 123, 1, 0.9);
        reference.evidence.clock_domain = None;
        assert!(memory.admit(&reference));
        let nearest = memory.assess(&reference).nearest.unwrap();
        assert_eq!(nearest.timestamp_us, 123);
        assert_eq!(nearest.clock_domain, None);
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
    fn changed_encoding_space_is_a_separate_novelty_namespace() {
        let mut memory = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        let reference = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        assert!(memory.admit(&reference));

        let migrated = in_space(reference.clone(), 8);
        let assessment = memory.assess(&migrated);
        assert_eq!(assessment.novelty, 1.0);
        assert!(assessment.nearest.is_none());
        assert_eq!(assessment.memory_size, 0);
        assert!(memory.admit(&migrated));
        assert_eq!(memory.len(ChemicalModality::Olfactory), 2);
        assert_eq!(
            memory.len_in_space(
                ChemicalModality::Olfactory,
                ChemicalEncodingSpaceId::from_bytes([8; 32])
            ),
            1
        );
    }

    #[test]
    fn memory_is_bounded_within_each_space() {
        let config = ChemicalNoveltyConfig {
            capacity_per_space: 2,
            max_spaces_per_modality: 2,
            min_admission_confidence: 0.5,
            duplicate_similarity: 1.0,
        };
        let mut memory = ChemicalNoveltyMemory::new(config).unwrap();
        assert!(memory.admit(&percept(ChemicalModality::Olfactory, 1, 1, 0.9)));
        assert!(memory.admit(&percept(ChemicalModality::Olfactory, 2, 2, 0.9)));
        assert!(memory.admit(&percept(ChemicalModality::Olfactory, 3, 3, 0.9)));
        assert_eq!(memory.len(ChemicalModality::Olfactory), 2);
    }

    #[test]
    fn representation_migration_history_is_bounded_per_modality() {
        let config = ChemicalNoveltyConfig {
            capacity_per_space: 2,
            max_spaces_per_modality: 2,
            min_admission_confidence: 0.5,
            duplicate_similarity: 1.0,
        };
        let mut memory = ChemicalNoveltyMemory::new(config).unwrap();
        let base = percept(ChemicalModality::Olfactory, 1, 1, 0.9);

        assert!(memory.admit(&in_space(base.clone(), 7)));
        assert!(memory.admit(&in_space(base.clone(), 8)));
        assert_eq!(memory.retained_space_count(ChemicalModality::Olfactory), 2);

        assert!(memory.admit(&in_space(base, 9)));
        assert_eq!(memory.retained_space_count(ChemicalModality::Olfactory), 2);
        assert_eq!(
            memory.len_in_space(
                ChemicalModality::Olfactory,
                ChemicalEncodingSpaceId::from_bytes([7; 32])
            ),
            0
        );
        assert_eq!(
            memory.len_in_space(
                ChemicalModality::Olfactory,
                ChemicalEncodingSpaceId::from_bytes([8; 32])
            ),
            1
        );
        assert_eq!(
            memory.len_in_space(
                ChemicalModality::Olfactory,
                ChemicalEncodingSpaceId::from_bytes([9; 32])
            ),
            1
        );
    }

    #[test]
    fn zero_space_limits_are_rejected() {
        let mut config = ChemicalNoveltyConfig::default();
        config.capacity_per_space = 0;
        assert!(matches!(
            ChemicalNoveltyMemory::new(config),
            Err(NoveltyConfigError::ZeroSpaceCapacity)
        ));

        let mut config = ChemicalNoveltyConfig::default();
        config.max_spaces_per_modality = 0;
        assert!(matches!(
            ChemicalNoveltyMemory::new(config),
            Err(NoveltyConfigError::ZeroEncodingSpaceHistory)
        ));
    }
}
