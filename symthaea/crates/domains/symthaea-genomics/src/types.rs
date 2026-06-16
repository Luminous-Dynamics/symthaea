// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for the genomics pipeline.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

// =============================================================================
// BASE AND SEQUENCE
// =============================================================================

/// DNA nucleotide base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Base {
    /// Adenine
    A,
    /// Thymine
    T,
    /// Guanine
    G,
    /// Cytosine
    C,
    /// Unknown / ambiguous base
    N,
}

impl Base {
    /// Convert base to a deterministic seed for HDC encoding.
    pub fn as_seed(self) -> u64 {
        match self {
            Base::A => 1,
            Base::T => 2,
            Base::G => 3,
            Base::C => 4,
            Base::N => 5,
        }
    }

    /// Complement base (Watson-Crick pairing).
    pub fn complement(self) -> Self {
        match self {
            Base::A => Base::T,
            Base::T => Base::A,
            Base::G => Base::C,
            Base::C => Base::G,
            Base::N => Base::N,
        }
    }

    /// Convert from ASCII character.
    pub fn from_char(c: char) -> Self {
        match c {
            'A' | 'a' => Base::A,
            'T' | 't' => Base::T,
            'G' | 'g' => Base::G,
            'C' | 'c' => Base::C,
            _ => Base::N,
        }
    }

    /// Convert to ASCII character.
    pub fn to_char(self) -> char {
        match self {
            Base::A => 'A',
            Base::T => 'T',
            Base::G => 'G',
            Base::C => 'C',
            Base::N => 'N',
        }
    }
}

/// A DNA sequence as a vector of bases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DnaSequence {
    pub bases: Vec<Base>,
}

impl DnaSequence {
    /// Create a new DNA sequence from bases.
    pub fn new(bases: Vec<Base>) -> Self {
        Self { bases }
    }

    /// Create from a string of ATGCN characters.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self {
            bases: s.chars().map(Base::from_char).collect(),
        }
    }

    /// Length of the sequence.
    pub fn len(&self) -> usize {
        self.bases.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    /// Get a subsequence (k-mer).
    pub fn subsequence(&self, start: usize, end: usize) -> DnaSequence {
        let end = end.min(self.bases.len());
        let start = start.min(end);
        DnaSequence {
            bases: self.bases[start..end].to_vec(),
        }
    }

    /// Reverse complement of the sequence.
    pub fn reverse_complement(&self) -> DnaSequence {
        DnaSequence {
            bases: self.bases.iter().rev().map(|b| b.complement()).collect(),
        }
    }

    /// Convert to string representation.
    pub fn to_string_repr(&self) -> String {
        self.bases.iter().map(|b| b.to_char()).collect()
    }

    /// Count occurrences of a specific base.
    pub fn count_base(&self, base: Base) -> usize {
        self.bases.iter().filter(|&&b| b == base).count()
    }

    /// GC content as a fraction [0, 1].
    pub fn gc_content(&self) -> f64 {
        if self.bases.is_empty() {
            return 0.0;
        }
        let gc = self.count_base(Base::G) + self.count_base(Base::C);
        gc as f64 / self.bases.len() as f64
    }
}

// =============================================================================
// DAMAGE TYPES
// =============================================================================

/// Type of DNA damage observed or predicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DamageType {
    /// C->T (or G->A on complementary strand) deamination.
    /// Most common ancient DNA damage pattern.
    Deamination,
    /// Loss of purine base (A or G), creating an abasic site.
    Depurination,
    /// Single-strand break in the phosphodiester backbone.
    StrandBreak,
    /// Interstrand crosslink preventing strand separation.
    Crosslink,
    /// Oxidative damage (e.g., 8-oxoguanine).
    Oxidation,
}

/// A single damage event at a specific position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DamageEvent {
    /// Position in the sequence where damage occurred.
    pub position: usize,
    /// Type of damage.
    pub damage_type: DamageType,
    /// Severity of the damage [0.0, 1.0].
    pub severity: f64,
}

// =============================================================================
// ASSEMBLY TYPES
// =============================================================================

/// An assembled contig with quality information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyContig {
    /// The assembled consensus sequence.
    pub sequence: DnaSequence,
    /// Per-base quality scores [0.0, 1.0].
    pub quality_scores: Vec<f64>,
    /// Per-base read coverage.
    pub coverage: Vec<u32>,
    /// HDC hypervector representation of the contig.
    pub hv: ContinuousHV,
}

/// A sample of degraded/ancient DNA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradedSample {
    /// The observed (possibly damaged) sequence.
    pub sequence: DnaSequence,
    /// Estimated age of the sample in years.
    pub estimated_age_years: f64,
    /// Storage temperature in Celsius.
    pub storage_temp_celsius: f64,
    /// Estimated contamination fraction [0.0, 1.0].
    pub contamination_estimate: f64,
    /// Map of detected damage events.
    pub damage_map: Vec<DamageEvent>,
}

// =============================================================================
// QUALITY TYPES
// =============================================================================

/// Assembly quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeQuality {
    /// N50: contig length such that 50% of total assembly is in contigs >= this length.
    pub n50: usize,
    /// Total assembled length in bases.
    pub total_length: usize,
    /// Number of contigs.
    pub num_contigs: usize,
    /// Mean read coverage across all positions.
    pub mean_coverage: f64,
    /// Estimated completeness [0.0, 1.0].
    pub completeness: f64,
    /// Estimated error rate [0.0, 1.0].
    pub error_rate: f64,
}

// =============================================================================
// REPAIR TYPES
// =============================================================================

/// Strategy for repairing a damaged region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepairStrategy {
    /// Revert C->T deamination back to original base.
    DeaminationRevert,
    /// Use majority voting across overlapping reads.
    ConsensusVoting,
    /// Fill gap with synthetic sequence (e.g., from reference).
    SyntheticFillIn,
    /// Use a reference genome to guide repair.
    ReferenceGuided,
}

/// A repairable region with a chosen strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairRegion {
    /// Start position (inclusive).
    pub start: usize,
    /// End position (exclusive).
    pub end: usize,
    /// Type of damage in this region.
    pub damage_type: DamageType,
    /// Chosen repair strategy.
    pub repair_strategy: RepairStrategy,
    /// Confidence in repair [0.0, 1.0].
    pub confidence: f64,
}

/// A complete repair plan for a degraded sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairPlan {
    /// Regions that can be repaired.
    pub repairable_regions: Vec<RepairRegion>,
    /// Gaps that cannot be repaired (start, end).
    pub irreparable_gaps: Vec<(usize, usize)>,
    /// Total number of bases requiring synthetic fill.
    pub synthetic_fill_needed: usize,
    /// Overall confidence in the repair plan [0.0, 1.0].
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_complement() {
        assert_eq!(Base::A.complement(), Base::T);
        assert_eq!(Base::T.complement(), Base::A);
        assert_eq!(Base::G.complement(), Base::C);
        assert_eq!(Base::C.complement(), Base::G);
        assert_eq!(Base::N.complement(), Base::N);
    }

    #[test]
    fn test_base_from_char() {
        assert_eq!(Base::from_char('A'), Base::A);
        assert_eq!(Base::from_char('t'), Base::T);
        assert_eq!(Base::from_char('X'), Base::N);
    }

    #[test]
    fn test_base_roundtrip() {
        for base in [Base::A, Base::T, Base::G, Base::C, Base::N] {
            assert_eq!(Base::from_char(base.to_char()), base);
        }
    }

    #[test]
    fn test_dna_sequence_from_str() {
        let seq = DnaSequence::from_str("ATGCN");
        assert_eq!(seq.len(), 5);
        assert_eq!(seq.bases[0], Base::A);
        assert_eq!(seq.bases[4], Base::N);
    }

    #[test]
    fn test_dna_sequence_reverse_complement() {
        let seq = DnaSequence::from_str("ATGC");
        let rc = seq.reverse_complement();
        assert_eq!(rc.to_string_repr(), "GCAT");
    }

    #[test]
    fn test_gc_content() {
        let seq = DnaSequence::from_str("AATTGGCC");
        let gc = seq.gc_content();
        assert!((gc - 0.5).abs() < 1e-10, "Expected 0.5, got {}", gc);
    }

    #[test]
    fn test_gc_content_empty() {
        let seq = DnaSequence::new(vec![]);
        assert_eq!(seq.gc_content(), 0.0);
    }

    #[test]
    fn test_subsequence() {
        let seq = DnaSequence::from_str("ATGCATGC");
        let sub = seq.subsequence(2, 5);
        assert_eq!(sub.to_string_repr(), "GCA");
    }

    #[test]
    fn test_subsequence_bounds() {
        let seq = DnaSequence::from_str("ATGC");
        let sub = seq.subsequence(2, 100);
        assert_eq!(sub.to_string_repr(), "GC");
    }

    #[test]
    fn test_count_base() {
        let seq = DnaSequence::from_str("AAATTTGC");
        assert_eq!(seq.count_base(Base::A), 3);
        assert_eq!(seq.count_base(Base::T), 3);
        assert_eq!(seq.count_base(Base::G), 1);
        assert_eq!(seq.count_base(Base::C), 1);
    }

    #[test]
    fn test_damage_type_variants() {
        // Ensure all variants are distinct
        let types = [
            DamageType::Deamination,
            DamageType::Depurination,
            DamageType::StrandBreak,
            DamageType::Crosslink,
            DamageType::Oxidation,
        ];
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                assert_ne!(types[i], types[j]);
            }
        }
    }

    #[test]
    fn test_repair_strategy_variants() {
        let strategies = [
            RepairStrategy::DeaminationRevert,
            RepairStrategy::ConsensusVoting,
            RepairStrategy::SyntheticFillIn,
            RepairStrategy::ReferenceGuided,
        ];
        for i in 0..strategies.len() {
            for j in (i + 1)..strategies.len() {
                assert_ne!(strategies[i], strategies[j]);
            }
        }
    }
}
