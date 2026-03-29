// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for population genetics.
//!
//! Provides the fundamental data structures: individuals, populations, pedigrees,
//! mating pairs, alleles, loci, genotypes, and breeding strategies.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Biological sex for breeding pair selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiologicalSex {
    /// Female sex, typically the egg-producing parent.
    Female,
    /// Male sex, typically the sperm-producing parent.
    Male,
}

/// A single allele variant at a locus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allele {
    /// Unique identifier for this allele variant.
    pub id: u64,
    /// Whether this allele carries a deleterious mutation.
    pub is_deleterious: bool,
    /// Selection coefficient: 0.0 = neutral, negative = harmful, positive = advantageous.
    pub selection_coefficient: f64,
    /// Dominance coefficient: 0.0 = fully recessive, 1.0 = fully dominant.
    pub dominance: f64,
}

impl Allele {
    /// Create a neutral (non-deleterious) allele.
    pub fn neutral(id: u64) -> Self {
        Self {
            id,
            is_deleterious: false,
            selection_coefficient: 0.0,
            dominance: 0.5,
        }
    }

    /// Create a deleterious allele with given selection and dominance coefficients.
    pub fn deleterious(id: u64, selection_coefficient: f64, dominance: f64) -> Self {
        Self {
            id,
            is_deleterious: true,
            selection_coefficient,
            dominance,
        }
    }
}

/// A genetic locus (position on a chromosome).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Locus {
    /// Unique locus identifier.
    pub id: u64,
    /// Human-readable locus name.
    pub name: String,
    /// Chromosome number.
    pub chromosome: u8,
    /// Position on chromosome (base pairs).
    pub position: u64,
    /// 16,384D HDC hypervector identifying this locus.
    pub hv: ContinuousHV,
}

/// A diploid genotype at a single locus (two alleles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genotype {
    /// Which locus this genotype is at.
    pub locus_id: u64,
    /// First allele (e.g., maternal).
    pub allele_a: Allele,
    /// Second allele (e.g., paternal).
    pub allele_b: Allele,
}

impl Genotype {
    /// Whether this genotype is heterozygous (different alleles).
    pub fn is_heterozygous(&self) -> bool {
        self.allele_a.id != self.allele_b.id
    }

    /// Whether this genotype is homozygous (same allele).
    pub fn is_homozygous(&self) -> bool {
        self.allele_a.id == self.allele_b.id
    }

    /// Whether either allele is deleterious.
    pub fn has_deleterious(&self) -> bool {
        self.allele_a.is_deleterious || self.allele_b.is_deleterious
    }

    /// Whether both alleles are deleterious (homozygous deleterious).
    pub fn is_homozygous_deleterious(&self) -> bool {
        self.allele_a.is_deleterious && self.allele_b.is_deleterious
    }
}

/// An individual organism with diploid genome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Unique individual identifier.
    pub id: u64,
    /// Biological sex.
    pub sex: BiologicalSex,
    /// Genotypes at each locus (one per locus).
    pub genotypes: Vec<Genotype>,
    /// 16,384D bundled genome hypervector encoding.
    pub genome_hv: ContinuousHV,
    /// Generation number (0 = founder).
    pub generation: u32,
    /// Parent IDs (None for founders).
    pub parent_ids: (Option<u64>, Option<u64>),
    /// Individual fitness (1.0 = wild-type).
    pub fitness: f64,
}

/// A population of individuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    /// All individuals in the population.
    pub individuals: Vec<Individual>,
    /// Current generation number.
    pub generation: u32,
    /// Original founding population size.
    pub founding_size: usize,
    /// Bundled population diversity hypervector.
    pub diversity_hv: ContinuousHV,
}

impl Population {
    /// Count females in the population.
    pub fn count_females(&self) -> usize {
        self.individuals
            .iter()
            .filter(|i| i.sex == BiologicalSex::Female)
            .count()
    }

    /// Count males in the population.
    pub fn count_males(&self) -> usize {
        self.individuals
            .iter()
            .filter(|i| i.sex == BiologicalSex::Male)
            .count()
    }

    /// Get population size.
    pub fn size(&self) -> usize {
        self.individuals.len()
    }

    /// Get all females.
    pub fn females(&self) -> Vec<&Individual> {
        self.individuals
            .iter()
            .filter(|i| i.sex == BiologicalSex::Female)
            .collect()
    }

    /// Get all males.
    pub fn males(&self) -> Vec<&Individual> {
        self.individuals
            .iter()
            .filter(|i| i.sex == BiologicalSex::Male)
            .collect()
    }
}

/// A pedigree tracking ancestry across generations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pedigree {
    /// All pedigree entries.
    pub entries: Vec<PedigreeEntry>,
}

impl Pedigree {
    /// Create an empty pedigree.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an entry to the pedigree.
    pub fn add_entry(&mut self, entry: PedigreeEntry) {
        self.entries.push(entry);
    }

    /// Find entry for a given individual.
    pub fn find(&self, individual_id: u64) -> Option<&PedigreeEntry> {
        self.entries
            .iter()
            .find(|e| e.individual_id == individual_id)
    }

    /// Get all entries for a given generation.
    pub fn generation_entries(&self, generation: u32) -> Vec<&PedigreeEntry> {
        self.entries
            .iter()
            .filter(|e| e.generation == generation)
            .collect()
    }
}

impl Default for Pedigree {
    fn default() -> Self {
        Self::new()
    }
}

/// A single pedigree entry linking individual to parents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedigreeEntry {
    /// The individual.
    pub individual_id: u64,
    /// Parent A (e.g., mother). None for founders.
    pub parent_a_id: Option<u64>,
    /// Parent B (e.g., father). None for founders.
    pub parent_b_id: Option<u64>,
    /// Generation of this individual.
    pub generation: u32,
}

/// A proposed mating pair with computed metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatingPair {
    /// ID of first parent.
    pub parent_a: u64,
    /// ID of second parent.
    pub parent_b: u64,
    /// Pedigree-based kinship coefficient.
    pub kinship: f64,
    /// HLA complementarity score (higher = more complementary).
    pub hla_complementarity: f64,
}

/// A genetic rescue plan to restore diversity via migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticRescuePlan {
    /// Name of source population for migrants.
    pub source_population: String,
    /// Number of migrants to introduce.
    pub num_migrants: usize,
    /// Target generation for introduction.
    pub target_generation: u32,
    /// Expected heterozygosity gain from rescue.
    pub expected_heterozygosity_gain: f64,
}

/// Breeding strategy for pair selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreedingStrategy {
    /// Random pairing.
    Random,
    /// Minimize mean kinship (Ballou & Lacy 1995).
    MinimumKinship,
    /// Maximum avoidance of inbreeding (MAI).
    MaximumAvoidance,
    /// Novel: maximize HDC distance between mates.
    HdcDistanceMaximize,
    /// Equal family sizes across the population.
    BalancedContribution,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::HDC_DIMENSION;

    #[test]
    fn test_biological_sex_equality() {
        assert_eq!(BiologicalSex::Female, BiologicalSex::Female);
        assert_ne!(BiologicalSex::Female, BiologicalSex::Male);
    }

    #[test]
    fn test_allele_neutral() {
        let a = Allele::neutral(1);
        assert_eq!(a.id, 1);
        assert!(!a.is_deleterious);
        assert_eq!(a.selection_coefficient, 0.0);
    }

    #[test]
    fn test_allele_deleterious() {
        let a = Allele::deleterious(2, -0.1, 0.3);
        assert!(a.is_deleterious);
        assert_eq!(a.selection_coefficient, -0.1);
        assert_eq!(a.dominance, 0.3);
    }

    #[test]
    fn test_genotype_heterozygous() {
        let g = Genotype {
            locus_id: 0,
            allele_a: Allele::neutral(1),
            allele_b: Allele::neutral(2),
        };
        assert!(g.is_heterozygous());
        assert!(!g.is_homozygous());
    }

    #[test]
    fn test_genotype_homozygous() {
        let g = Genotype {
            locus_id: 0,
            allele_a: Allele::neutral(1),
            allele_b: Allele::neutral(1),
        };
        assert!(g.is_homozygous());
        assert!(!g.is_heterozygous());
    }

    #[test]
    fn test_genotype_has_deleterious() {
        let g = Genotype {
            locus_id: 0,
            allele_a: Allele::neutral(1),
            allele_b: Allele::deleterious(2, -0.1, 0.5),
        };
        assert!(g.has_deleterious());
        assert!(!g.is_homozygous_deleterious());
    }

    #[test]
    fn test_genotype_homozygous_deleterious() {
        let g = Genotype {
            locus_id: 0,
            allele_a: Allele::deleterious(1, -0.1, 0.5),
            allele_b: Allele::deleterious(1, -0.1, 0.5),
        };
        assert!(g.is_homozygous_deleterious());
    }

    #[test]
    fn test_population_counts() {
        let pop = Population {
            individuals: vec![
                Individual {
                    id: 0,
                    sex: BiologicalSex::Female,
                    genotypes: vec![],
                    genome_hv: ContinuousHV::zero(HDC_DIMENSION),
                    generation: 0,
                    parent_ids: (None, None),
                    fitness: 1.0,
                },
                Individual {
                    id: 1,
                    sex: BiologicalSex::Male,
                    genotypes: vec![],
                    genome_hv: ContinuousHV::zero(HDC_DIMENSION),
                    generation: 0,
                    parent_ids: (None, None),
                    fitness: 1.0,
                },
                Individual {
                    id: 2,
                    sex: BiologicalSex::Female,
                    genotypes: vec![],
                    genome_hv: ContinuousHV::zero(HDC_DIMENSION),
                    generation: 0,
                    parent_ids: (None, None),
                    fitness: 1.0,
                },
            ],
            generation: 0,
            founding_size: 3,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(pop.count_females(), 2);
        assert_eq!(pop.count_males(), 1);
        assert_eq!(pop.size(), 3);
    }

    #[test]
    fn test_pedigree_operations() {
        let mut ped = Pedigree::new();
        ped.add_entry(PedigreeEntry {
            individual_id: 1,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 2,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 3,
            parent_a_id: Some(1),
            parent_b_id: Some(2),
            generation: 1,
        });
        assert!(ped.find(1).is_some());
        assert!(ped.find(99).is_none());
        assert_eq!(ped.generation_entries(0).len(), 2);
        assert_eq!(ped.generation_entries(1).len(), 1);
    }

    #[test]
    fn test_breeding_strategy_serialization() {
        let strategies = [
            BreedingStrategy::Random,
            BreedingStrategy::MinimumKinship,
            BreedingStrategy::MaximumAvoidance,
            BreedingStrategy::HdcDistanceMaximize,
            BreedingStrategy::BalancedContribution,
        ];
        for s in &strategies {
            let json = serde_json::to_string(s).unwrap();
            let back: BreedingStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
    }

    #[test]
    fn test_mating_pair_creation() {
        let pair = MatingPair {
            parent_a: 1,
            parent_b: 2,
            kinship: 0.0625,
            hla_complementarity: 0.85,
        };
        assert_eq!(pair.parent_a, 1);
        assert_eq!(pair.parent_b, 2);
    }

    #[test]
    fn test_genetic_rescue_plan() {
        let plan = GeneticRescuePlan {
            source_population: "zoo_population_a".to_string(),
            num_migrants: 5,
            target_generation: 10,
            expected_heterozygosity_gain: 0.05,
        };
        let json = serde_json::to_string(&plan).unwrap();
        let back: GeneticRescuePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_migrants, 5);
        assert_eq!(back.source_population, "zoo_population_a");
    }
}
