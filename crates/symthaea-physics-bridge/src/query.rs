// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Similarity search engine for the physics catalog.
//!
//! Provides weighted multi-aspect search across the catalog:
//! - **Structural (0.4)**: Skeleton similarity (cross-domain analogy)
//! - **Symmetry (0.3)**: Lie group / discrete symmetry match
//! - **Dimensional (0.2)**: SI dimension agreement
//! - **Full (0.1)**: Complete equation match (with names)

use crate::catalog::PhysicsCatalog;
use crate::dimensional::DimensionalEncoder;
use crate::equation_ast::EquationEncoder;
use crate::symmetry::SymmetryEncoder;
use crate::types::*;
use symthaea_core::hdc::ContinuousHV;

/// Default weight for structural (skeleton) similarity.
const W_STRUCTURAL: f32 = 0.4;
/// Default weight for symmetry similarity.
const W_SYMMETRY: f32 = 0.3;
/// Default weight for dimensional similarity.
const W_DIMENSIONAL: f32 = 0.2;
/// Default weight for full equation similarity.
const W_FULL: f32 = 0.1;

/// Physics search engine with weighted multi-aspect similarity.
pub struct PhysicsSearchEngine {
    catalog: PhysicsCatalog,
    eq_encoder: EquationEncoder,
    sym_encoder: SymmetryEncoder,
    dim_encoder: DimensionalEncoder,
    weights: SearchWeights,
}

/// Configurable search weights.
#[derive(Debug, Clone, Copy)]
pub struct SearchWeights {
    pub structural: f32,
    pub symmetry: f32,
    pub dimensional: f32,
    pub full: f32,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            structural: W_STRUCTURAL,
            symmetry: W_SYMMETRY,
            dimensional: W_DIMENSIONAL,
            full: W_FULL,
        }
    }
}

impl PhysicsSearchEngine {
    /// Create a search engine with default weights.
    pub fn new() -> Self {
        Self {
            catalog: PhysicsCatalog::new(),
            eq_encoder: EquationEncoder::new(),
            sym_encoder: SymmetryEncoder::new(),
            dim_encoder: DimensionalEncoder::new(),
            weights: SearchWeights::default(),
        }
    }

    /// Create with custom weights.
    pub fn with_weights(weights: SearchWeights) -> Self {
        Self {
            catalog: PhysicsCatalog::new(),
            eq_encoder: EquationEncoder::new(),
            sym_encoder: SymmetryEncoder::new(),
            dim_encoder: DimensionalEncoder::new(),
            weights,
        }
    }

    /// Search for equations similar to the given equation.
    ///
    /// Returns results sorted by descending score.
    pub fn search_equation(
        &self,
        equation: &PhysicsEquation,
        max_results: usize,
    ) -> Vec<SearchResult> {
        let full_hv = self.eq_encoder.encode(&equation.ast);
        let skeleton_hv = self.eq_encoder.encode_skeleton(&equation.ast);
        let symmetry_hv = self.sym_encoder.encode(&equation.symmetries);
        let dimensional_hv = self.dim_encoder.encode(&equation.dimensions);

        self.search_multi_aspect(
            &full_hv,
            &skeleton_hv,
            &symmetry_hv,
            &dimensional_hv,
            max_results,
        )
    }

    /// Search for equations similar to the given equation, with custom
    /// per-query weights. Identical to [`search_equation`] but lets callers
    /// override `SearchWeights` for a single query without mutating the
    /// engine's stored defaults. Useful for recognition paths that want to
    /// down-weight the full-encoding axis (which is dominated by random-HV
    /// hash noise when the top candidates all tie on structural + symmetry
    /// + dimensional) without breaking other search-engine consumers.
    pub fn search_equation_with_weights(
        &self,
        equation: &PhysicsEquation,
        max_results: usize,
        weights: SearchWeights,
    ) -> Vec<SearchResult> {
        let full_hv = self.eq_encoder.encode(&equation.ast);
        let skeleton_hv = self.eq_encoder.encode_skeleton(&equation.ast);
        let symmetry_hv = self.sym_encoder.encode(&equation.symmetries);
        let dimensional_hv = self.dim_encoder.encode(&equation.dimensions);

        let mut results: Vec<SearchResult> = self
            .catalog
            .entries()
            .iter()
            .map(|entry| {
                let structural = entry.skeleton_hv.similarity(&skeleton_hv);
                let symmetry = entry.symmetry_hv.similarity(&symmetry_hv);
                let dimensional = entry.dimensional_hv.similarity(&dimensional_hv);
                let full = entry.full_hv.similarity(&full_hv);
                let score = weights.structural * structural
                    + weights.symmetry * symmetry
                    + weights.dimensional * dimensional
                    + weights.full * full;
                SearchResult {
                    name: entry.equation.name.clone(),
                    domain: entry.equation.domain,
                    score,
                    breakdown: SimilarityBreakdown {
                        structural,
                        symmetry,
                        dimensional,
                        full,
                    },
                }
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Search by skeleton only (best for cross-domain analogy discovery).
    pub fn search_by_skeleton(&self, ast: &EquationNode, max_results: usize) -> Vec<SearchResult> {
        let skeleton_hv = self.eq_encoder.encode_skeleton(ast);
        let mut results: Vec<SearchResult> = self
            .catalog
            .entries()
            .iter()
            .map(|entry| {
                let score = entry.skeleton_hv.similarity(&skeleton_hv);
                SearchResult {
                    name: entry.equation.name.clone(),
                    domain: entry.equation.domain,
                    score,
                    breakdown: SimilarityBreakdown {
                        structural: score,
                        symmetry: 0.0,
                        dimensional: 0.0,
                        full: 0.0,
                    },
                }
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Search by symmetry group profile.
    pub fn search_by_symmetry(
        &self,
        symmetry: &SymmetryDescriptor,
        max_results: usize,
    ) -> Vec<SearchResult> {
        let sym_hv = self.sym_encoder.encode(symmetry);
        let mut results: Vec<SearchResult> = self
            .catalog
            .entries()
            .iter()
            .map(|entry| {
                let score = entry.symmetry_hv.similarity(&sym_hv);
                SearchResult {
                    name: entry.equation.name.clone(),
                    domain: entry.equation.domain,
                    score,
                    breakdown: SimilarityBreakdown {
                        structural: 0.0,
                        symmetry: score,
                        dimensional: 0.0,
                        full: 0.0,
                    },
                }
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Search by SI dimensional signature.
    pub fn search_by_dimensions(
        &self,
        dims: &DimensionalSignature,
        max_results: usize,
    ) -> Vec<SearchResult> {
        let dim_hv = self.dim_encoder.encode(dims);
        let mut results: Vec<SearchResult> = self
            .catalog
            .entries()
            .iter()
            .map(|entry| {
                let score = entry.dimensional_hv.similarity(&dim_hv);
                SearchResult {
                    name: entry.equation.name.clone(),
                    domain: entry.equation.domain,
                    score,
                    breakdown: SimilarityBreakdown {
                        structural: 0.0,
                        symmetry: 0.0,
                        dimensional: score,
                        full: 0.0,
                    },
                }
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Search by raw hypervector (for cognitive loop integration).
    ///
    /// Compares the query HV against each catalog entry's full encoding.
    pub fn search_by_hv(&self, query_hv: &ContinuousHV, max_results: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self
            .catalog
            .entries()
            .iter()
            .map(|entry| {
                let score = entry.full_hv.similarity(query_hv);
                SearchResult {
                    name: entry.equation.name.clone(),
                    domain: entry.equation.domain,
                    score,
                    breakdown: SimilarityBreakdown {
                        structural: entry.skeleton_hv.similarity(query_hv),
                        symmetry: entry.symmetry_hv.similarity(query_hv),
                        dimensional: entry.dimensional_hv.similarity(query_hv),
                        full: score,
                    },
                }
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Search for metric solutions similar to a given tensor descriptor.
    pub fn search_metrics(
        &self,
        query: &TensorDescriptor,
        max_results: usize,
    ) -> Vec<SearchResult> {
        let tensor_encoder = crate::tensor_structure::TensorEncoder::new();
        let query_hv = tensor_encoder.encode(query);

        let mut results: Vec<SearchResult> = self
            .catalog
            .entries()
            .iter()
            .filter_map(|entry| {
                let tensor = entry.equation.tensor.as_ref()?;
                let entry_hv = tensor_encoder.encode(tensor);
                let score = query_hv.similarity(&entry_hv);
                Some(SearchResult {
                    name: entry.equation.name.clone(),
                    domain: entry.equation.domain,
                    score,
                    breakdown: SimilarityBreakdown {
                        structural: score,
                        symmetry: 0.0,
                        dimensional: 0.0,
                        full: 0.0,
                    },
                })
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Access the underlying catalog.
    pub fn catalog(&self) -> &PhysicsCatalog {
        &self.catalog
    }

    /// Number of equations in the catalog.
    pub fn catalog_size(&self) -> usize {
        self.catalog.len()
    }

    /// Weighted multi-aspect search across all catalog entries.
    fn search_multi_aspect(
        &self,
        full_hv: &ContinuousHV,
        skeleton_hv: &ContinuousHV,
        symmetry_hv: &ContinuousHV,
        dimensional_hv: &ContinuousHV,
        max_results: usize,
    ) -> Vec<SearchResult> {
        let w = &self.weights;

        let mut results: Vec<SearchResult> = self
            .catalog
            .entries()
            .iter()
            .map(|entry| {
                let structural = entry.skeleton_hv.similarity(skeleton_hv);
                let symmetry = entry.symmetry_hv.similarity(symmetry_hv);
                let dimensional = entry.dimensional_hv.similarity(dimensional_hv);
                let full = entry.full_hv.similarity(full_hv);

                let score = w.structural * structural
                    + w.symmetry * symmetry
                    + w.dimensional * dimensional
                    + w.full * full;

                SearchResult {
                    name: entry.equation.name.clone(),
                    domain: entry.equation.domain,
                    score,
                    breakdown: SimilarityBreakdown {
                        structural,
                        symmetry,
                        dimensional,
                        full,
                    },
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }
}

impl Default for PhysicsSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equation_ast::{make_diffop, make_equals, make_field};
    use symthaea_core::hdc::HDC_DIMENSION;

    fn engine() -> PhysicsSearchEngine {
        PhysicsSearchEngine::new()
    }

    #[test]
    fn search_returns_results() {
        let eng = engine();
        let results = eng.search_by_dimensions(&DimensionalSignature::ENERGY, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn search_results_sorted_descending() {
        let eng = engine();
        let results = eng.search_by_dimensions(&DimensionalSignature::ENERGY, 10);
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "Results not sorted: {} ({}) before {} ({})",
                window[0].name,
                window[0].score,
                window[1].name,
                window[1].score
            );
        }
    }

    #[test]
    fn self_search_returns_top() {
        let eng = engine();
        let gauss = eng.catalog().find_by_name("Maxwell Gauss Law").unwrap();
        let eq = &gauss.equation;
        let results = eng.search_equation(eq, 5);
        assert_eq!(
            results[0].name, "Maxwell Gauss Law",
            "Self-search should return self as top result"
        );
        assert!(results[0].score > 0.8, "Self-search score should be high");
    }

    #[test]
    fn skeleton_search_finds_analogies() {
        let eng = engine();
        // Wave equation structure: □ψ = 0
        let lor4 = MetricSignature::Lorentzian(4);
        let wave_ast = make_equals(
            make_diffop(
                DiffOperator::DAlembertian,
                make_field("φ", TensorDescriptor::scalar(lor4)),
            ),
            EquationNode::Scalar(0.0),
        );
        let results = eng.search_by_skeleton(&wave_ast, 5);
        // Wave equation should be in top results
        assert!(
            results.iter().any(|r| r.name == "Wave Equation"),
            "Wave equation should appear in skeleton search"
        );
    }

    #[test]
    fn symmetry_search_finds_gauge_theories() {
        let eng = engine();
        let u1_gauge = SymmetryDescriptor::from_lie_groups(vec![LieGroup::U(1)], true);
        let results = eng.search_by_symmetry(&u1_gauge, 50);
        // Many equations now have U(1) gauge symmetry. Verify EM equations appear somewhere.
        let em_count = results
            .iter()
            .filter(|r| r.domain == PhysicsDomain::Electromagnetism)
            .count();
        assert!(
            em_count >= 1,
            "Should find at least one EM equation with U(1) gauge (found {} in {} results)",
            em_count,
            results.len()
        );
    }

    #[test]
    fn dimensional_search_groups_energies() {
        let eng = engine();
        let results = eng.search_by_dimensions(&DimensionalSignature::ENERGY, 10);
        // Schrödinger, Dirac, Klein-Gordon all have energy dimensions
        let qm_names: Vec<&str> = results
            .iter()
            .filter(|r| {
                r.domain == PhysicsDomain::QuantumMechanics
                    || r.domain == PhysicsDomain::QuantumFieldTheory
            })
            .map(|r| r.name.as_str())
            .collect();
        assert!(
            !qm_names.is_empty(),
            "Energy-dimension search should find quantum equations"
        );
    }

    #[test]
    fn search_by_hv() {
        let eng = engine();
        let random_hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let results = eng.search_by_hv(&random_hv, 5);
        assert_eq!(results.len(), 5);
        // Should have breakdown scores
        for r in &results {
            assert!(r.breakdown.full.is_finite());
        }
    }

    #[test]
    fn search_metrics_finds_black_holes() {
        let eng = engine();
        let lor4 = MetricSignature::Lorentzian(4);
        let mut query = TensorDescriptor::symmetric_2(lor4);
        query.solution = Some(MetricSolution {
            singularity: true,
            horizon: true,
            vacuum: true,
            stationary: true,
            axisymmetric: true,
            spherically_symmetric: true,
            cosmological: false,
            has_cosmological_constant: false,
            has_charge: false,
        });
        let results = eng.search_metrics(&query, 5);
        assert!(
            results[0].name == "Schwarzschild Metric",
            "Schwarzschild should be top metric match, got {}",
            results[0].name
        );
    }

    #[test]
    fn nuclear_equations_found_by_full_search() {
        let eng = engine();
        // Search for the Gamow peak integral itself — should find related nuclear eqs
        let gamow = eng.catalog().find_by_name("Gamow Peak Integral").unwrap();
        let results = eng.search_equation(&gamow.equation, 10);
        // Gamow should be top, with other nuclear physics near it. We use top-10
        // rather than top-5 because the recognition catalog now includes
        // structurally-similar entries from other domains (e.g. Angular Momentum
        // 2D Cartesian, added for Ramanujan Protocol showcase) that can land
        // between nuclear entries without changing the clustering intent: nuclear
        // physics still forms a tight neighborhood around any nuclear query.
        assert_eq!(results[0].name, "Gamow Peak Integral");
        let nuclear_count = results
            .iter()
            .filter(|r| r.domain == PhysicsDomain::NuclearPhysics)
            .count();
        assert!(
            nuclear_count >= 2,
            "Expected >= 2 nuclear equations in top 10, got {nuclear_count}"
        );
    }

    #[test]
    fn custom_weights() {
        let weights = SearchWeights {
            structural: 0.0,
            symmetry: 1.0,
            dimensional: 0.0,
            full: 0.0,
        };
        let eng = PhysicsSearchEngine::with_weights(weights);
        let eq = eng.catalog().find_by_name("Maxwell Gauss Law").unwrap();
        let results = eng.search_equation(&eq.equation, 5);
        // With symmetry-only weights, should cluster by symmetry group
        assert!(!results.is_empty());
    }

    #[test]
    fn catalog_size() {
        let eng = engine();
        assert!(eng.catalog_size() >= 27);
    }
}
