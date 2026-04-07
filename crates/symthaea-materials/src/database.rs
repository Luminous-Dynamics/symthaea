// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Material database with HDC similarity search and constraint filtering.

use crate::encoder::MaterialHdcEncoder;
use crate::properties::MaterialProperty;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// A single result from an HDC similarity search.
#[derive(Debug, Clone)]
pub struct MaterialSearchResult {
    /// The matched material.
    pub material: MaterialProperty,
    /// Cosine similarity to the query (higher = more similar).
    pub similarity: f32,
}

/// In-memory material database with HDC similarity search.
///
/// Materials are encoded into 16,384D hypervectors on insertion.
/// Queries find the most similar materials by cosine similarity,
/// optionally filtered by property constraints.
pub struct MaterialDatabase {
    materials: Vec<MaterialProperty>,
    encoder: MaterialHdcEncoder,
    hvs: Vec<ContinuousHV>,
}

impl MaterialDatabase {
    /// Create an empty database.
    pub fn new() -> Self {
        Self {
            materials: Vec::new(),
            encoder: MaterialHdcEncoder::new(),
            hvs: Vec::new(),
        }
    }

    /// Create a database pre-loaded with the 5 preset reference materials.
    pub fn with_presets() -> Self {
        let mut db = Self::new();
        for m in MaterialProperty::presets() {
            db.add(m);
        }
        db
    }

    /// Add a material to the database (encodes its HV on insertion).
    pub fn add(&mut self, material: MaterialProperty) {
        let hv = self.encoder.encode(&material);
        self.hvs.push(hv);
        self.materials.push(material);
    }

    /// Number of materials in the database.
    pub fn len(&self) -> usize {
        self.materials.len()
    }
    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Find the `top_k` most similar materials to a query material.
    pub fn search_similar(
        &self,
        query: &MaterialProperty,
        top_k: usize,
    ) -> Vec<MaterialSearchResult> {
        self.search_by_hv(&self.encoder.encode(query), top_k)
    }

    /// Find the `top_k` most similar materials to a raw hypervector query.
    pub fn search_by_hv(&self, query_hv: &ContinuousHV, top_k: usize) -> Vec<MaterialSearchResult> {
        let mut results: Vec<MaterialSearchResult> = self
            .materials
            .iter()
            .zip(self.hvs.iter())
            .map(|(m, hv)| MaterialSearchResult {
                material: m.clone(),
                similarity: query_hv.similarity(hv),
            })
            .collect();
        results.sort_by(|a, b| nan_safe_cmp_desc(a.similarity, b.similarity));
        results.truncate(top_k);
        results
    }

    /// Find the `top_k` most similar materials matching a constraint predicate.
    pub fn constrained_search<F>(
        &self,
        query: &MaterialProperty,
        constraint: F,
        top_k: usize,
    ) -> Vec<MaterialSearchResult>
    where
        F: Fn(&MaterialProperty) -> bool,
    {
        let query_hv = self.encoder.encode(query);
        let mut results: Vec<MaterialSearchResult> = self
            .materials
            .iter()
            .zip(self.hvs.iter())
            .filter(|(m, _)| constraint(m))
            .map(|(m, hv)| MaterialSearchResult {
                material: m.clone(),
                similarity: query_hv.similarity(hv),
            })
            .collect();
        results.sort_by(|a, b| nan_safe_cmp_desc(a.similarity, b.similarity));
        results.truncate(top_k);
        results
    }
}

impl Default for MaterialDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// NaN-safe descending comparison: NaN sorts last (below all finite values).
fn nan_safe_cmp_desc(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater, // NaN sorts after finite
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::properties::MaterialCategory;

    #[test]
    fn test_with_presets() {
        assert_eq!(MaterialDatabase::with_presets().len(), 6);
    }
    #[test]
    fn test_search_similar_returns_top_k() {
        assert_eq!(
            MaterialDatabase::with_presets()
                .search_similar(&MaterialProperty::steel_a36(), 3)
                .len(),
            3
        );
    }

    #[test]
    fn test_search_self_match_first() {
        let r = MaterialDatabase::with_presets().search_similar(&MaterialProperty::steel_a36(), 5);
        assert_eq!(r[0].material.name, "Steel A36");
        assert!(r[0].similarity > 0.99);
    }

    #[test]
    fn test_constrained_search_metals() {
        let r = MaterialDatabase::with_presets().constrained_search(
            &MaterialProperty::titanium_ti6al4v(),
            |m| m.category == MaterialCategory::Metal,
            10,
        );
        assert_eq!(r.len(), 3);
        for x in &r {
            assert_eq!(x.material.category, MaterialCategory::Metal);
        }
    }

    #[test]
    fn test_constrained_search_yield() {
        let r = MaterialDatabase::with_presets().constrained_search(
            &MaterialProperty::titanium_ti6al4v(),
            |m| m.category == MaterialCategory::Metal && m.yield_strength_mpa > 200.0,
            10,
        );
        for x in &r {
            assert!(x.material.yield_strength_mpa > 200.0);
        }
    }

    #[test]
    fn test_search_by_hv() {
        let db = MaterialDatabase::with_presets();
        let enc = MaterialHdcEncoder::new();
        let r = db.search_by_hv(&enc.encode(&MaterialProperty::steel_a36()), 2);
        assert_eq!(r.len(), 2);
        assert!(r[0].similarity >= r[1].similarity);
    }

    #[test]
    fn test_empty_database() {
        let db = MaterialDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }
    #[test]
    fn test_add_material() {
        let mut db = MaterialDatabase::new();
        db.add(MaterialProperty::steel_a36());
        assert_eq!(db.len(), 1);
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    fn test_search_results_sorted_descending() {
        let db = MaterialDatabase::with_presets();
        let results = db.search_similar(&MaterialProperty::steel_a36(), 5);
        for i in 1..results.len() {
            assert!(
                results[i - 1].similarity >= results[i].similarity,
                "Results not sorted: {} < {} at position {}",
                results[i - 1].similarity,
                results[i].similarity,
                i
            );
        }
    }

    #[test]
    fn test_search_top_k_zero() {
        let db = MaterialDatabase::with_presets();
        let results = db.search_similar(&MaterialProperty::steel_a36(), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_exceeds_database() {
        let db = MaterialDatabase::with_presets();
        let results = db.search_similar(&MaterialProperty::steel_a36(), 100);
        assert_eq!(results.len(), 6); // 6 presets including Bi-Mg metamaterial
    }
}
