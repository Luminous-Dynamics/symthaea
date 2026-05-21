// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC-indexed memory for narrow SMT proof results.
//!
//! This is deliberately solver-agnostic. The existing Z3 bridge can populate
//! these records; live generation can retrieve similar solved proof shapes
//! without invoking an SMT solver on every candidate.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::rust_ast_hdc::{ast_feature_cosine_similarity, encode_rust_ast_hdc};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofVerdict {
    Proven,
    Refuted,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRecord {
    pub label: String,
    pub verdict: ProofVerdict,
    pub ast_features: BTreeMap<String, usize>,
    pub smtlib2: String,
    pub summary: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofMemory {
    records: Vec<ProofRecord>,
}

impl ProofMemory {
    pub fn observe(&mut self, record: ProofRecord) {
        self.records.push(record);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn nearest<'a>(
        &'a self,
        ast_features: &BTreeMap<String, usize>,
        min_similarity: f32,
    ) -> Option<(&'a ProofRecord, f32)> {
        self.records
            .iter()
            .filter_map(|record| {
                ast_feature_cosine_similarity(ast_features, &record.ast_features)
                    .map(|score| (record, score))
            })
            .filter(|(_, score)| *score >= min_similarity)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

pub fn proof_record_for_rust_source(
    label: impl Into<String>,
    source: &str,
    smtlib2: impl Into<String>,
    verdict: ProofVerdict,
    summary: impl Into<String>,
) -> Result<ProofRecord, syn::Error> {
    let encoded = encode_rust_ast_hdc(source, symthaea_core::hdc::unified_hv::HDC_DIMENSION)?;
    Ok(ProofRecord {
        label: label.into(),
        verdict,
        ast_features: encoded.features,
        smtlib2: smtlib2.into(),
        summary: summary.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieves_nearest_proof_shape() {
        let mut memory = ProofMemory::default();
        let record = proof_record_for_rust_source(
            "increment",
            "pub fn inc(x: i32) -> i32 { x + 1 }",
            "(assert (= y (+ x 1)))",
            ProofVerdict::Proven,
            "increments by one",
        )
        .unwrap();
        memory.observe(record);

        let query = encode_rust_ast_hdc("pub fn bump(value: i32) -> i32 { value + 1 }", 512)
            .unwrap()
            .features;
        let (nearest, score) = memory.nearest(&query, 0.1).unwrap();

        assert_eq!(nearest.verdict, ProofVerdict::Proven);
        assert!(score > 0.1);
    }

    #[test]
    fn proof_memory_round_trips_json() {
        let mut memory = ProofMemory::default();
        memory.observe(
            proof_record_for_rust_source(
                "is_positive",
                "pub fn is_positive(x: i32) -> bool { x > 0 }",
                "(assert (> x 0))",
                ProofVerdict::Unknown,
                "predicate shape",
            )
            .unwrap(),
        );

        let restored = ProofMemory::from_json(&memory.to_json().unwrap()).unwrap();
        assert_eq!(restored.len(), 1);
    }
}
