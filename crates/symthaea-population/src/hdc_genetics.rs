// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 16,384D genome encoding using HDC bind/bundle operations.
//!
//! Pattern: bind(locus_hv, allele_hv) per locus, bundle all -> individual genome HV.
//! Genetic distance = 1.0 - similarity(a, b).
//!
//! This encoding allows genetic similarity to be computed in O(1) via cosine
//! similarity of bundled genome hypervectors, avoiding O(n_loci) comparisons.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::{Genotype, Individual, Locus, Population};

/// Salt constant mixed into locus seed to ensure locus HVs are orthogonal to allele HVs.
const LOCUS_SALT: u64 = 0x10C0_5A17_DEAD_BEEF;

/// Encode an allele as a deterministic 16,384D hypervector.
///
/// The same allele ID always produces the same HV (reproducible via seeded RNG).
pub fn encode_allele(allele_id: u64) -> ContinuousHV {
    ContinuousHV::random(HDC_DIMENSION, allele_id)
}

/// Encode a locus identifier as a 16,384D hypervector.
///
/// Uses a salt to ensure locus HVs are quasi-orthogonal to allele HVs.
pub fn encode_locus(locus_id: u64) -> ContinuousHV {
    ContinuousHV::random(HDC_DIMENSION, locus_id.wrapping_add(LOCUS_SALT))
}

/// Encode a single diploid genotype at a locus.
///
/// Binds each allele to the locus HV, then bundles the two bound HVs.
/// This preserves which alleles are at which locus while allowing
/// similarity comparison of genotypes.
pub fn encode_genotype(locus: &Locus, genotype: &Genotype) -> ContinuousHV {
    let locus_hv = &locus.hv;
    let allele_a_hv = encode_allele(genotype.allele_a.id);
    let allele_b_hv = encode_allele(genotype.allele_b.id);
    let bound_a = locus_hv.bind(&allele_a_hv);
    let bound_b = locus_hv.bind(&allele_b_hv);
    bound_a.add(&bound_b).normalize()
}

/// Encode an individual's full genome as a bundled 16,384D HV.
///
/// Bundles all genotype encodings across all loci into a single genome HV.
/// Requires `loci` and `genotypes` to be in matching order.
pub fn encode_individual(loci: &[Locus], genotypes: &[Genotype]) -> ContinuousHV {
    assert!(
        !genotypes.is_empty(),
        "Cannot encode individual with zero genotypes"
    );
    let encoded: Vec<ContinuousHV> = genotypes
        .iter()
        .enumerate()
        .map(|(i, g)| encode_genotype(&loci[i % loci.len()], g))
        .collect();
    let refs: Vec<&ContinuousHV> = encoded.iter().collect();
    ContinuousHV::bundle(&refs)
}

/// Compute genetic distance between two individuals.
///
/// Returns a value in [0, ~2] where 0 = identical genomes and higher = more divergent.
/// Based on 1.0 - cosine_similarity of their genome HVs.
pub fn genetic_distance(a: &Individual, b: &Individual) -> f64 {
    1.0 - a.genome_hv.similarity(&b.genome_hv) as f64
}

/// Compute mean pairwise genetic distance across a population.
///
/// Returns 0.0 for populations with fewer than 2 individuals.
pub fn population_diversity(population: &Population) -> f64 {
    let n = population.individuals.len();
    if n < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    let mut pair_count = 0u64;

    for i in 0..n {
        for j in (i + 1)..n {
            total_distance +=
                genetic_distance(&population.individuals[i], &population.individuals[j]);
            pair_count += 1;
        }
    }

    total_distance / pair_count as f64
}

/// Compute HLA complementarity between two genome HVs.
///
/// Lower similarity = higher complementarity for immune system genes.
/// Returns a value in [0, ~2] where higher = more complementary.
pub fn hla_complementarity(a: &ContinuousHV, b: &ContinuousHV) -> f64 {
    1.0 - a.similarity(b).abs() as f64
}

/// Update the population diversity HV by bundling all individual genome HVs.
pub fn compute_diversity_hv(population: &Population) -> ContinuousHV {
    if population.individuals.is_empty() {
        return ContinuousHV::zero(HDC_DIMENSION);
    }
    let refs: Vec<&ContinuousHV> = population
        .individuals
        .iter()
        .map(|ind| &ind.genome_hv)
        .collect();
    ContinuousHV::bundle(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Allele, BiologicalSex};

    fn make_locus(id: u64) -> Locus {
        Locus {
            id,
            name: format!("locus_{}", id),
            chromosome: 1,
            position: id * 1000,
            hv: encode_locus(id),
        }
    }

    fn make_individual(id: u64, loci: &[Locus], allele_pairs: &[(u64, u64)]) -> Individual {
        let genotypes: Vec<Genotype> = allele_pairs
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| Genotype {
                locus_id: loci[i].id,
                allele_a: Allele::neutral(a),
                allele_b: Allele::neutral(b),
            })
            .collect();
        let genome_hv = encode_individual(loci, &genotypes);
        Individual {
            id,
            sex: if id % 2 == 0 {
                BiologicalSex::Female
            } else {
                BiologicalSex::Male
            },
            genotypes,
            genome_hv,
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        }
    }

    #[test]
    fn test_encode_allele_deterministic() {
        let a1 = encode_allele(42);
        let a2 = encode_allele(42);
        assert_eq!(a1.dim(), HDC_DIMENSION);
        assert!(
            (a1.similarity(&a2) - 1.0).abs() < 1e-6,
            "same seed => identical HV"
        );
    }

    #[test]
    fn test_encode_allele_different_ids_quasi_orthogonal() {
        let a1 = encode_allele(1);
        let a2 = encode_allele(2);
        let sim = a1.similarity(&a2).abs();
        assert!(
            sim < 0.1,
            "different allele IDs should be quasi-orthogonal, got {sim}"
        );
    }

    #[test]
    fn test_encode_locus_uses_salt() {
        // Locus and allele with same numeric ID should differ due to salt
        let locus_hv = encode_locus(100);
        let allele_hv = encode_allele(100);
        let sim = locus_hv.similarity(&allele_hv).abs();
        assert!(
            sim < 0.1,
            "locus and allele with same ID should be quasi-orthogonal, got {sim}"
        );
    }

    #[test]
    fn test_encode_genotype_normalized() {
        let locus = make_locus(0);
        let g = Genotype {
            locus_id: 0,
            allele_a: Allele::neutral(1),
            allele_b: Allele::neutral(2),
        };
        let hv = encode_genotype(&locus, &g);
        let norm = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "encoded genotype should be normalized, got norm={norm}"
        );
    }

    #[test]
    fn test_self_distance_is_zero() {
        let loci = vec![make_locus(0), make_locus(1)];
        let ind = make_individual(1, &loci, &[(10, 20), (30, 40)]);
        let d = genetic_distance(&ind, &ind);
        assert!(d.abs() < 1e-5, "self-distance should be ~0, got {d}");
    }

    #[test]
    fn test_unrelated_distance_positive() {
        let loci = vec![make_locus(0), make_locus(1), make_locus(2)];
        let ind_a = make_individual(1, &loci, &[(10, 20), (30, 40), (50, 60)]);
        let ind_b = make_individual(2, &loci, &[(70, 80), (90, 100), (110, 120)]);
        let d = genetic_distance(&ind_a, &ind_b);
        assert!(
            d > 0.5,
            "unrelated individuals should have large distance, got {d}"
        );
    }

    #[test]
    fn test_identical_genomes_zero_distance() {
        let loci = vec![make_locus(0), make_locus(1)];
        let ind_a = make_individual(1, &loci, &[(10, 20), (30, 40)]);
        let ind_b = make_individual(2, &loci, &[(10, 20), (30, 40)]);
        let d = genetic_distance(&ind_a, &ind_b);
        assert!(
            d.abs() < 1e-5,
            "identical genomes should have distance ~0, got {d}"
        );
    }

    #[test]
    fn test_population_diversity_empty() {
        let pop = Population {
            individuals: vec![],
            generation: 0,
            founding_size: 0,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(population_diversity(&pop), 0.0);
    }

    #[test]
    fn test_population_diversity_single() {
        let loci = vec![make_locus(0)];
        let pop = Population {
            individuals: vec![make_individual(1, &loci, &[(10, 20)])],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(population_diversity(&pop), 0.0);
    }

    #[test]
    fn test_population_diversity_increases_with_heterozygosity() {
        let loci = vec![make_locus(0), make_locus(1), make_locus(2)];

        // Homogeneous population: all same genotypes
        let homogeneous = Population {
            individuals: (0..5)
                .map(|i| make_individual(i, &loci, &[(10, 10), (20, 20), (30, 30)]))
                .collect(),
            generation: 0,
            founding_size: 5,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };

        // Heterogeneous population: all different genotypes
        let heterogeneous = Population {
            individuals: (0..5)
                .map(|i| {
                    let base = (i as u64) * 100;
                    make_individual(
                        i,
                        &loci,
                        &[(base, base + 1), (base + 2, base + 3), (base + 4, base + 5)],
                    )
                })
                .collect(),
            generation: 0,
            founding_size: 5,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };

        let d_homo = population_diversity(&homogeneous);
        let d_hetero = population_diversity(&heterogeneous);
        assert!(
            d_hetero > d_homo,
            "heterogeneous pop should be more diverse: {d_hetero} vs {d_homo}"
        );
    }

    #[test]
    fn test_hla_complementarity_self_low() {
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let comp = hla_complementarity(&hv, &hv);
        assert!(
            comp < 0.01,
            "self-complementarity should be near 0, got {comp}"
        );
    }

    #[test]
    fn test_hla_complementarity_unrelated_high() {
        let hv_a = ContinuousHV::random(HDC_DIMENSION, 42);
        let hv_b = ContinuousHV::random(HDC_DIMENSION, 99);
        let comp = hla_complementarity(&hv_a, &hv_b);
        assert!(
            comp > 0.8,
            "unrelated HLA should have high complementarity, got {comp}"
        );
    }

    #[test]
    fn test_compute_diversity_hv() {
        let loci = vec![make_locus(0)];
        let pop = Population {
            individuals: vec![
                make_individual(1, &loci, &[(10, 20)]),
                make_individual(2, &loci, &[(30, 40)]),
            ],
            generation: 0,
            founding_size: 2,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let div_hv = compute_diversity_hv(&pop);
        assert_eq!(div_hv.dim(), HDC_DIMENSION);
        // Should be non-zero
        let norm = div_hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "diversity HV should be non-zero");
    }

    #[test]
    fn test_encode_individual_multiple_loci() {
        let loci = vec![make_locus(0), make_locus(1), make_locus(2), make_locus(3)];
        let ind = make_individual(1, &loci, &[(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(ind.genome_hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_genetic_distance_symmetry() {
        let loci = vec![make_locus(0), make_locus(1)];
        let a = make_individual(1, &loci, &[(10, 20), (30, 40)]);
        let b = make_individual(2, &loci, &[(50, 60), (70, 80)]);
        let d_ab = genetic_distance(&a, &b);
        let d_ba = genetic_distance(&b, &a);
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "distance should be symmetric: {d_ab} vs {d_ba}"
        );
    }
}
