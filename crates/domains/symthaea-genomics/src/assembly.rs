// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Overlap-layout-consensus assembly using HDC similarity for k-mer matching.
//!
//! Traditional sequence assemblers use hash tables for exact k-mer matching.
//! This module instead encodes each k-mer as a 16,384D hypervector and uses
//! cosine similarity to find overlaps, naturally tolerating errors and damage.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::{AssemblyContig, Base, DnaSequence};

/// HDC-based sequence assembler.
///
/// Uses hypervector similarity for overlap detection, which naturally handles
/// mismatches from sequencing errors or ancient DNA damage.
#[derive(Debug, Clone)]
pub struct HdcAssembler {
    /// k-mer size for overlap encoding (default 21).
    pub k: usize,
    /// Minimum HDC similarity for an overlap (default 0.7).
    pub min_overlap_similarity: f32,
    /// Hypervector dimension (default 16,384).
    pub dimension: usize,
}

impl Default for HdcAssembler {
    fn default() -> Self {
        Self {
            k: 21,
            min_overlap_similarity: 0.7,
            dimension: HDC_DIMENSION,
        }
    }
}

impl HdcAssembler {
    /// Create a new assembler with custom parameters.
    pub fn new(k: usize, min_overlap_similarity: f32) -> Self {
        Self {
            k,
            min_overlap_similarity,
            dimension: HDC_DIMENSION,
        }
    }

    /// Encode a k-mer as a hypervector.
    ///
    /// Uses positional binding: HV = base\[0\] bind permute(base\[1\], 1) bind permute(base\[2\], 2) ...
    /// This creates a position-sensitive encoding where the same base at different
    /// positions produces distinct (near-orthogonal) contributions.
    pub fn encode_kmer(kmer: &[Base], dimension: usize) -> ContinuousHV {
        if kmer.is_empty() {
            return ContinuousHV::zero(dimension);
        }

        // Create a base HV for the first base
        let mut result = ContinuousHV::random(dimension, kmer[0].as_seed());

        // Bind with position-shifted base HVs for subsequent bases
        for (pos, &base) in kmer.iter().enumerate().skip(1) {
            let base_hv = ContinuousHV::random(dimension, base.as_seed());
            let positioned = base_hv.permute(pos);
            result = result.bind(&positioned);
        }

        result.normalize()
    }

    /// Encode an entire read as a hypervector (bundle of its k-mer HVs).
    fn encode_read(&self, read: &DnaSequence) -> ContinuousHV {
        if read.len() < self.k {
            return ContinuousHV::zero(self.dimension);
        }

        let kmer_hvs: Vec<ContinuousHV> = (0..=(read.len() - self.k))
            .map(|i| Self::encode_kmer(&read.bases[i..i + self.k], self.dimension))
            .collect();

        let refs: Vec<&ContinuousHV> = kmer_hvs.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Find overlapping reads using HDC similarity.
    ///
    /// Returns triples of (read_i, read_j, similarity) for all pairs above threshold.
    pub fn find_overlaps(&self, reads: &[DnaSequence]) -> Vec<(usize, usize, f32)> {
        let read_hvs: Vec<ContinuousHV> = reads.iter().map(|r| self.encode_read(r)).collect();

        let mut overlaps = Vec::new();

        for i in 0..read_hvs.len() {
            for j in (i + 1)..read_hvs.len() {
                let sim = read_hvs[i].similarity(&read_hvs[j]);
                if sim >= self.min_overlap_similarity {
                    overlaps.push((i, j, sim));
                }
            }
        }

        // Sort by similarity descending (best overlaps first)
        overlaps.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        overlaps
    }

    /// Assemble reads into contigs using greedy overlap-layout-consensus.
    ///
    /// 1. Encode all reads as hypervectors
    /// 2. Find pairwise overlaps above threshold
    /// 3. Greedily merge reads with highest overlap first
    /// 4. Produce consensus contigs
    pub fn assemble(&self, reads: &[DnaSequence]) -> Vec<AssemblyContig> {
        if reads.is_empty() {
            return Vec::new();
        }

        // Special case: single read
        if reads.len() == 1 {
            return vec![self.read_to_contig(&reads[0])];
        }

        let overlaps = self.find_overlaps(reads);

        // Track which reads have been merged
        let mut merged: Vec<bool> = vec![false; reads.len()];
        let mut contigs: Vec<Vec<usize>> = Vec::new();

        // Greedy merge: take the highest-similarity pair first
        for (i, j, _sim) in &overlaps {
            if merged[*i] && merged[*j] {
                continue;
            }

            if !merged[*i] && !merged[*j] {
                // Neither merged yet: start a new contig
                contigs.push(vec![*i, *j]);
                merged[*i] = true;
                merged[*j] = true;
            } else if !merged[*j] {
                // i is already in a contig; add j to it
                if let Some(contig) = contigs.iter_mut().find(|c| c.contains(i)) {
                    contig.push(*j);
                    merged[*j] = true;
                }
            } else if !merged[*i] {
                // j is already in a contig; add i to it
                if let Some(contig) = contigs.iter_mut().find(|c| c.contains(j)) {
                    contig.push(*i);
                    merged[*i] = true;
                }
            }
        }

        // Any unmerged reads become singleton contigs
        for (i, was_merged) in merged.iter().enumerate() {
            if !was_merged {
                contigs.push(vec![i]);
            }
        }

        // Build consensus for each contig
        contigs
            .iter()
            .map(|read_indices| {
                let contig_reads: Vec<&DnaSequence> =
                    read_indices.iter().map(|&i| &reads[i]).collect();
                self.consensus(&contig_reads)
            })
            .collect()
    }

    /// Build a consensus sequence from a set of overlapping reads.
    fn consensus(&self, reads: &[&DnaSequence]) -> AssemblyContig {
        if reads.is_empty() {
            return AssemblyContig {
                sequence: DnaSequence::new(vec![]),
                quality_scores: vec![],
                coverage: vec![],
                hv: ContinuousHV::zero(self.dimension),
            };
        }

        if reads.len() == 1 {
            return self.read_to_contig(reads[0]);
        }

        // Find the longest read as the backbone
        let max_len = reads.iter().map(|r| r.len()).max().unwrap_or(0);

        let mut consensus_bases = Vec::with_capacity(max_len);
        let mut quality_scores = Vec::with_capacity(max_len);
        let mut coverage = Vec::with_capacity(max_len);

        // Per-position majority voting
        for pos in 0..max_len {
            let mut base_counts = [0u32; 5]; // A, T, G, C, N

            for read in reads {
                if pos < read.len() {
                    let idx = match read.bases[pos] {
                        Base::A => 0,
                        Base::T => 1,
                        Base::G => 2,
                        Base::C => 3,
                        Base::N => 4,
                    };
                    base_counts[idx] += 1;
                }
            }

            let total: u32 = base_counts.iter().sum();
            coverage.push(total);

            if total == 0 {
                consensus_bases.push(Base::N);
                quality_scores.push(0.0);
                continue;
            }

            // Find majority base
            let (best_idx, &best_count) = base_counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
                .expect("base_counts is non-empty (4 bases)");

            let base = match best_idx {
                0 => Base::A,
                1 => Base::T,
                2 => Base::G,
                3 => Base::C,
                _ => Base::N,
            };

            consensus_bases.push(base);
            quality_scores.push(best_count as f64 / total as f64);
        }

        let sequence = DnaSequence::new(consensus_bases);
        let hv = self.encode_read(&sequence);

        AssemblyContig {
            sequence,
            quality_scores,
            coverage,
            hv,
        }
    }

    /// Convert a single read to a contig.
    fn read_to_contig(&self, read: &DnaSequence) -> AssemblyContig {
        let hv = self.encode_read(read);
        AssemblyContig {
            sequence: read.clone(),
            quality_scores: vec![1.0; read.len()],
            coverage: vec![1; read.len()],
            hv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_kmer_deterministic() {
        let kmer = [Base::A, Base::T, Base::G, Base::C];
        let hv1 = HdcAssembler::encode_kmer(&kmer, HDC_DIMENSION);
        let hv2 = HdcAssembler::encode_kmer(&kmer, HDC_DIMENSION);

        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Same k-mer should produce identical HV, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_encode_kmer_different_kmers_dissimilar() {
        let kmer1 = [Base::A, Base::T, Base::G, Base::C];
        let kmer2 = [Base::C, Base::G, Base::T, Base::A];
        let hv1 = HdcAssembler::encode_kmer(&kmer1, HDC_DIMENSION);
        let hv2 = HdcAssembler::encode_kmer(&kmer2, HDC_DIMENSION);

        let sim = hv1.similarity(&hv2);
        assert!(
            sim.abs() < 0.5,
            "Different k-mers should be dissimilar, got {}",
            sim
        );
    }

    #[test]
    fn test_encode_kmer_position_sensitive() {
        // Same bases in different order should produce different HVs
        let kmer1 = [Base::A, Base::T, Base::G];
        let kmer2 = [Base::G, Base::T, Base::A];
        let hv1 = HdcAssembler::encode_kmer(&kmer1, HDC_DIMENSION);
        let hv2 = HdcAssembler::encode_kmer(&kmer2, HDC_DIMENSION);

        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.8,
            "Position-swapped k-mers should be distinguishable, got {}",
            sim
        );
    }

    #[test]
    fn test_encode_kmer_empty() {
        let hv = HdcAssembler::encode_kmer(&[], HDC_DIMENSION);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        // Should be zero vector
        assert!(hv.norm() < 1e-6);
    }

    #[test]
    fn test_find_overlaps_identical_reads() {
        let assembler = HdcAssembler::new(5, 0.5);
        let read = DnaSequence::from_str("ATGCATGCATGCATGCATGC");
        let reads = vec![read.clone(), read];

        let overlaps = assembler.find_overlaps(&reads);
        assert!(
            !overlaps.is_empty(),
            "Identical reads should have high overlap"
        );
        assert!(
            overlaps[0].2 > 0.9,
            "Identical reads should have similarity near 1.0"
        );
    }

    #[test]
    fn test_find_overlaps_no_overlap() {
        let assembler = HdcAssembler::new(5, 0.9);
        // Very different sequences
        let reads = vec![
            DnaSequence::from_str("AAAAAAAAAAAAAAAAAAAAA"),
            DnaSequence::from_str("CCCCCCCCCCCCCCCCCCCCC"),
        ];

        let overlaps = assembler.find_overlaps(&reads);
        // With a high threshold, dissimilar reads should not overlap
        assert!(
            overlaps.is_empty() || overlaps[0].2 < 0.9,
            "Very different sequences should not overlap"
        );
    }

    #[test]
    fn test_assemble_single_read() {
        let assembler = HdcAssembler::default();
        let reads = vec![DnaSequence::from_str("ATGCATGCATGCATGCATGCATGC")];
        let contigs = assembler.assemble(&reads);
        assert_eq!(contigs.len(), 1);
        assert_eq!(contigs[0].sequence.len(), 24);
    }

    #[test]
    fn test_assemble_empty() {
        let assembler = HdcAssembler::default();
        let contigs = assembler.assemble(&[]);
        assert!(contigs.is_empty());
    }

    #[test]
    fn test_assemble_overlapping_reads() {
        let assembler = HdcAssembler::new(5, 0.3);
        // Create reads that share significant sequence
        let reads = vec![
            DnaSequence::from_str("ATGCATGCATGCATGCATGC"),
            DnaSequence::from_str("ATGCATGCATGCATGCATGC"),
            DnaSequence::from_str("ATGCATGCATGCATGCATGC"),
        ];
        let contigs = assembler.assemble(&reads);

        // All identical reads should merge into one contig
        assert!(
            contigs.len() <= 2,
            "Identical reads should merge into few contigs, got {}",
            contigs.len()
        );
    }

    #[test]
    fn test_consensus_majority_voting() {
        let assembler = HdcAssembler::new(3, 0.3);
        // Three reads where majority agrees
        let reads = vec![
            DnaSequence::from_str("ATGC"),
            DnaSequence::from_str("ATGC"),
            DnaSequence::from_str("ATAC"), // G->A error at position 2
        ];

        let refs: Vec<&DnaSequence> = reads.iter().collect();
        let contig = assembler.consensus(&refs);

        // Majority should win: position 2 should be G
        assert_eq!(contig.sequence.bases[2], Base::G);
        assert_eq!(contig.coverage[0], 3);
        assert!(contig.quality_scores[2] > 0.6);
    }

    #[test]
    fn test_contig_has_hv() {
        let assembler = HdcAssembler::default();
        let reads = vec![DnaSequence::from_str("ATGCATGCATGCATGCATGCATGCATGC")];
        let contigs = assembler.assemble(&reads);

        assert_eq!(contigs[0].hv.dim(), HDC_DIMENSION);
        assert!(contigs[0].hv.norm() > 0.0, "Contig HV should be non-zero");
    }

    #[test]
    fn test_similar_reads_produce_similar_hvs() {
        let assembler = HdcAssembler::new(5, 0.5);
        let read1 = DnaSequence::from_str("ATGCATGCATGCATGC");
        let read2 = DnaSequence::from_str("ATGCATGCATGCATGC");
        let read3 = DnaSequence::from_str("CCCCCCCCCCCCCCCC");

        let hv1 = assembler.encode_read(&read1);
        let hv2 = assembler.encode_read(&read2);
        let hv3 = assembler.encode_read(&read3);

        let sim_same = hv1.similarity(&hv2);
        let sim_diff = hv1.similarity(&hv3);

        assert!(
            sim_same > sim_diff,
            "Similar reads should have higher similarity: same={}, diff={}",
            sim_same,
            sim_diff
        );
    }
}
