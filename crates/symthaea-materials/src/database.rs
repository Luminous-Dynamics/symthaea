//! Material database with HDC similarity search and constraint filtering.

use crate::encoder::MaterialHdcEncoder;
use crate::properties::MaterialProperty;
use symthaea_core::hdc::unified_hv::ContinuousHV;

#[derive(Debug, Clone)]
pub struct MaterialSearchResult {
    pub material: MaterialProperty,
    pub similarity: f32,
}

pub struct MaterialDatabase {
    materials: Vec<MaterialProperty>,
    encoder: MaterialHdcEncoder,
    hvs: Vec<ContinuousHV>,
}

impl MaterialDatabase {
    pub fn new() -> Self { Self { materials: Vec::new(), encoder: MaterialHdcEncoder::new(), hvs: Vec::new() } }

    pub fn with_presets() -> Self {
        let mut db = Self::new();
        for m in MaterialProperty::presets() { db.add(m); }
        db
    }

    pub fn add(&mut self, material: MaterialProperty) {
        let hv = self.encoder.encode(&material);
        self.hvs.push(hv);
        self.materials.push(material);
    }

    pub fn len(&self) -> usize { self.materials.len() }
    pub fn is_empty(&self) -> bool { self.materials.is_empty() }

    pub fn search_similar(&self, query: &MaterialProperty, top_k: usize) -> Vec<MaterialSearchResult> {
        self.search_by_hv(&self.encoder.encode(query), top_k)
    }

    pub fn search_by_hv(&self, query_hv: &ContinuousHV, top_k: usize) -> Vec<MaterialSearchResult> {
        let mut results: Vec<MaterialSearchResult> = self.materials.iter().zip(self.hvs.iter())
            .map(|(m, hv)| MaterialSearchResult { material: m.clone(), similarity: query_hv.similarity(hv) })
            .collect();
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    pub fn constrained_search<F>(&self, query: &MaterialProperty, constraint: F, top_k: usize) -> Vec<MaterialSearchResult>
    where F: Fn(&MaterialProperty) -> bool {
        let query_hv = self.encoder.encode(query);
        let mut results: Vec<MaterialSearchResult> = self.materials.iter().zip(self.hvs.iter())
            .filter(|(m, _)| constraint(m))
            .map(|(m, hv)| MaterialSearchResult { material: m.clone(), similarity: query_hv.similarity(hv) })
            .collect();
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

impl Default for MaterialDatabase { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::properties::MaterialCategory;

    #[test] fn test_with_presets() { assert_eq!(MaterialDatabase::with_presets().len(), 5); }
    #[test] fn test_search_similar_returns_top_k() { assert_eq!(MaterialDatabase::with_presets().search_similar(&MaterialProperty::steel_a36(), 3).len(), 3); }

    #[test]
    fn test_search_self_match_first() {
        let r = MaterialDatabase::with_presets().search_similar(&MaterialProperty::steel_a36(), 5);
        assert_eq!(r[0].material.name, "Steel A36");
        assert!(r[0].similarity > 0.99);
    }

    #[test]
    fn test_constrained_search_metals() {
        let r = MaterialDatabase::with_presets().constrained_search(&MaterialProperty::titanium_ti6al4v(), |m| m.category == MaterialCategory::Metal, 10);
        assert_eq!(r.len(), 3);
        for x in &r { assert_eq!(x.material.category, MaterialCategory::Metal); }
    }

    #[test]
    fn test_constrained_search_yield() {
        let r = MaterialDatabase::with_presets().constrained_search(&MaterialProperty::titanium_ti6al4v(), |m| m.category == MaterialCategory::Metal && m.yield_strength_mpa > 200.0, 10);
        for x in &r { assert!(x.material.yield_strength_mpa > 200.0); }
    }

    #[test] fn test_search_by_hv() {
        let db = MaterialDatabase::with_presets();
        let enc = MaterialHdcEncoder::new();
        let r = db.search_by_hv(&enc.encode(&MaterialProperty::steel_a36()), 2);
        assert_eq!(r.len(), 2);
        assert!(r[0].similarity >= r[1].similarity);
    }

    #[test] fn test_empty_database() { let db = MaterialDatabase::new(); assert!(db.is_empty()); assert_eq!(db.len(), 0); }
    #[test] fn test_add_material() { let mut db = MaterialDatabase::new(); db.add(MaterialProperty::steel_a36()); assert_eq!(db.len(), 1); }
}
