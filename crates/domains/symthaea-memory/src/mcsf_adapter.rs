// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, read-only HDC retrieval adapter for matched MCSF experiments.
//!
//! This first adapter deliberately uses Symthaea's real `CharNgramEncoder`
//! and `SemanticMemory` implementations. It is a character-trigram HDC
//! baseline, not a pretrained semantic encoder and not an HDC-CfC hybrid.
//! The benchmark harness must retain this encoder identity when comparing
//! it with BM25, dense-vector, or learned semantic systems.
//!
//! The adapter rebuilds an in-memory index per call so each benchmark query
//! is isolated from prior queries and cannot mutate production memory state.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::semantic_encoder::{CharNgramEncoder, SemanticEncoder};
use symthaea_core::hdc::unified_hv::HDC_DIMENSION;

use crate::semantic::SemanticMemory;

/// Similarity scores are exported as signed millionths of cosine similarity.
pub const HDC_RETRIEVAL_SCORE_SCALE: f64 = 1_000_000.0;
/// Versioned identity: changing encoder semantics requires a new identifier.
pub const HDC_CHAR_NGRAM_ADAPTER_ID: &str = "symthaea.hdc.char-ngram.semantic-memory.v0.1";

/// One frozen memory candidate supplied by the benchmark harness.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HdcMemoryRecord {
    pub memory_id: String,
    pub content: String,
}

/// One real Symthaea retrieval result with a content-integrity digest.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HdcRetrievalHit {
    pub memory_id: String,
    pub content_digest_blake3: String,
    /// Cosine similarity × 1,000,000, rounded to the nearest integer.
    pub similarity_micros: i64,
}

/// Execution metadata required to interpret and reproduce adapter output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HdcAdapterMetadata {
    pub adapter_id: String,
    pub encoder: String,
    pub encoder_seed: u64,
    pub hypervector_dimensions: usize,
    pub retrieval_similarity_threshold: String,
    pub score_scale: i64,
    pub score_quantization: String,
    pub implementation: String,
    pub cfc_participates_in_retrieval: bool,
}

/// Explicitly configured character-trigram HDC baseline over Symthaea memory.
#[derive(Debug)]
pub struct SymthaeaHdcMcsfAdapter {
    encoder: CharNgramEncoder,
    seed: u64,
}

impl SymthaeaHdcMcsfAdapter {
    /// Construct an encoder with a fixed seed. Identical inputs and seed
    /// produce identical hypervectors with the pinned Symthaea implementation.
    pub fn new(seed: u64) -> Self {
        Self {
            encoder: CharNgramEncoder::new(seed),
            seed,
        }
    }

    /// Describe the exact retrieval implementation and score convention.
    pub fn metadata(&self) -> HdcAdapterMetadata {
        HdcAdapterMetadata {
            adapter_id: HDC_CHAR_NGRAM_ADAPTER_ID.to_owned(),
            encoder: self.encoder.name().to_owned(),
            encoder_seed: self.seed,
            hypervector_dimensions: HDC_DIMENSION,
            retrieval_similarity_threshold: "f32::MIN (rank all finite cosine scores)".to_owned(),
            score_scale: HDC_RETRIEVAL_SCORE_SCALE as i64,
            score_quantization: "round(cosine_similarity * 1000000)".to_owned(),
            implementation: "Symthaea SemanticEncoder::CharNgramEncoder + SemanticMemory::find_similar".to_owned(),
            cfc_participates_in_retrieval: false,
        }
    }

    /// Retrieve from a frozen candidate set without modifying persistent memory.
    ///
    /// Ordering is cosine similarity descending, then stable memory ID
    /// ascending. IDs are validated before retrieval; malformed or duplicate
    /// records fail closed rather than producing ambiguous evidence.
    pub fn search(
        &self,
        records: &[HdcMemoryRecord],
        query: &str,
        top_k: usize,
    ) -> Result<Vec<HdcRetrievalHit>, String> {
        if top_k == 0 {
            return Err("retrieval Top K must be greater than zero".to_owned());
        }
        if query.trim().is_empty() {
            return Err("retrieval query must not be empty".to_owned());
        }
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let mut ids = BTreeSet::new();
        for record in records {
            if record.memory_id.trim().is_empty()
                || record.memory_id != record.memory_id.trim()
                || record.content.trim().is_empty()
            {
                return Err("memory records require stable non-empty IDs and content".to_owned());
            }
            if !ids.insert(record.memory_id.as_str()) {
                return Err(format!("duplicate memory ID: {}", record.memory_id));
            }
        }

        let query_hv = self.encoder.encode(query);
        validate_hypervector(&query_hv.values, "query")?;

        // Keep the engine's threshold below the full cosine range so the
        // adapter can apply deterministic tie-breaking before Top-K truncation.
        let mut memory = SemanticMemory::with_threshold(records.len().max(1), f32::MIN);
        for record in records {
            let encoded = self.encoder.encode(&record.content);
            validate_hypervector(&encoded.values, &record.memory_id)?;
            if encoded.values.len() != query_hv.values.len() {
                return Err(format!(
                    "hypervector dimension mismatch for memory {}: expected {}, got {}",
                    record.memory_id,
                    query_hv.values.len(),
                    encoded.values.len()
                ));
            }
            memory.store(encoded.values, 0.0, None);
        }

        // Ask the actual Symthaea memory engine for all candidates. Sorting
        // and Top-K are applied below so ties at the K boundary are stable.
        let candidates = memory.find_similar(&query_hv.values, records.len());
        let mut hits = Vec::with_capacity(candidates.len());
        for (index, similarity) in candidates {
            let record = records
                .get(index)
                .ok_or_else(|| format!("Symthaea returned out-of-range memory index: {index}"))?;
            if !similarity.is_finite() {
                return Err(format!(
                    "Symthaea returned non-finite similarity for memory {}",
                    record.memory_id
                ));
            }
            let scaled = (similarity.clamp(-1.0, 1.0) as f64 * HDC_RETRIEVAL_SCORE_SCALE)
                .round();
            hits.push((
                similarity,
                HdcRetrievalHit {
                    memory_id: record.memory_id.clone(),
                    content_digest_blake3: blake3::hash(record.content.as_bytes())
                        .to_hex()
                        .to_string(),
                    similarity_micros: scaled as i64,
                },
            ));
        }

        hits.sort_by(|(left_score, left_hit), (right_score, right_hit)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_hit.memory_id.cmp(&right_hit.memory_id))
        });
        hits.truncate(top_k);
        Ok(hits.into_iter().map(|(_, hit)| hit).collect())
    }
}

fn validate_hypervector(values: &[f32], label: &str) -> Result<(), String> {
    if values.len() != HDC_DIMENSION {
        return Err(format!(
            "{label} hypervector has wrong dimension: expected {HDC_DIMENSION}, got {}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} hypervector contains a non-finite value"));
    }
    let norm_squared = values.iter().map(|value| value * value).sum::<f32>();
    if !norm_squared.is_finite() || norm_squared <= 1e-12 {
        return Err(format!(
            "{label} encoded to a zero or invalid hypervector; refusing to score it"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(memory_id: &str, content: &str) -> HdcMemoryRecord {
        HdcMemoryRecord {
            memory_id: memory_id.to_owned(),
            content: content.to_owned(),
        }
    }

    #[test]
    fn exact_recall_uses_symthaea_memory_and_binds_content_digest() {
        let adapter = SymthaeaHdcMcsfAdapter::new(42);
        let records = vec![
            record("memory-b", "The evidence receipt binds the exact source digest."),
            record("memory-a", "A separate note discusses rainfall and gardening."),
        ];

        let hits = adapter
            .search(&records, "The evidence receipt binds the exact source digest.", 2)
            .unwrap();

        assert_eq!(hits[0].memory_id, "memory-b");
        assert_eq!(hits[0].similarity_micros, 1_000_000);
        assert_eq!(
            hits[0].content_digest_blake3,
            blake3::hash(records[0].content.as_bytes()).to_hex().to_string()
        );
    }

    #[test]
    fn ties_are_broken_by_stable_memory_id_before_top_k() {
        let adapter = SymthaeaHdcMcsfAdapter::new(42);
        let records = vec![
            record("zeta", "The same exact memory content is stored twice."),
            record("alpha", "The same exact memory content is stored twice."),
        ];

        let hits = adapter
            .search(&records, "The same exact memory content is stored twice.", 1)
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, "alpha");
    }

    #[test]
    fn fixed_seed_produces_identical_ordered_hits() {
        let records = vec![
            record("one", "An update supersedes a prior revision."),
            record("two", "Revocation must reach dependent records."),
            record("three", "The weather is sunny today."),
        ];
        let first = SymthaeaHdcMcsfAdapter::new(2026)
            .search(&records, "revocation dependent records", 3)
            .unwrap();
        let second = SymthaeaHdcMcsfAdapter::new(2026)
            .search(&records, "revocation dependent records", 3)
            .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn unicode_content_is_encoded_and_retrieved() {
        let adapter = SymthaeaHdcMcsfAdapter::new(7);
        let records = vec![
            record("zulu", "ukuvikelwa kwedatha nokwabelana kolwazi"),
            record("sesotho", "tšireletso ea tsebo le boinotši"),
        ];

        let zulu = adapter
            .search(&records, "ukuvikelwa kwedatha nokwabelana kolwazi", 2)
            .unwrap();
        let sesotho = adapter
            .search(&records, "tšireletso ea tsebo le boinotši", 2)
            .unwrap();

        assert_eq!(zulu[0].memory_id, "zulu");
        assert_eq!(sesotho[0].memory_id, "sesotho");
    }

    #[test]
    fn rejects_duplicate_ids_empty_values_and_short_zero_vectors() {
        let adapter = SymthaeaHdcMcsfAdapter::new(42);
        let duplicate = vec![
            record("same", "A long enough memory record."),
            record("same", "Another long enough memory record."),
        ];
        assert!(adapter.search(&duplicate, "A sufficiently long query", 2).is_err());

        let empty = vec![record("memory", "  ")];
        assert!(adapter.search(&empty, "A sufficiently long query", 2).is_err());

        let short = vec![record("short", "ab")];
        assert!(adapter.search(&short, "A sufficiently long query", 2).is_err());
    }

    #[test]
    fn rejects_invalid_query_and_top_k() {
        let adapter = SymthaeaHdcMcsfAdapter::new(42);
        let records = vec![record("one", "A long enough memory record.")];

        assert!(adapter.search(&records, "   ", 1).is_err());
        assert!(adapter.search(&records, "A sufficiently long query", 0).is_err());
        assert!(adapter.search(&records, "ab", 1).is_err());
    }

    #[test]
    fn empty_corpus_returns_empty_after_validating_controls() {
        let adapter = SymthaeaHdcMcsfAdapter::new(42);
        assert!(
            adapter
                .search(&[], "A sufficiently long query", 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn metadata_keeps_encoder_and_cfc_claims_explicit() {
        let metadata = SymthaeaHdcMcsfAdapter::new(123).metadata();

        assert_eq!(metadata.adapter_id, HDC_CHAR_NGRAM_ADAPTER_ID);
        assert_eq!(metadata.encoder, "CharNgram");
        assert_eq!(metadata.encoder_seed, 123);
        assert_eq!(metadata.hypervector_dimensions, HDC_DIMENSION);
        assert!(!metadata.cfc_participates_in_retrieval);
    }
}
