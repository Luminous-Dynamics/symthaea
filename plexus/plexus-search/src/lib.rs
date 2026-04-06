// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic knowledge search engine for Symthaea Prism.
//!
//! Uses Symthaea's 16,384-bit BinaryHV for semantic encoding and
//! batch similarity search over a pre-seeded claim index.
//!
//! Encoding: word-level random indexing (each word → deterministic BinaryHV,
//! bound with position vector, bundled into document vector).

use plexus_common::{EmpiricalLevel, SearchResult};
mod binary_hv;
use binary_hv::BinaryHV;

/// Epistemic knowledge search engine.
pub struct SearchEngine {
    claims: Vec<ClaimEntry>,
    vectors: Vec<BinaryHV>,
    /// Word → document frequency (how many claims contain this word).
    word_df: std::collections::HashMap<String, usize>,
}

struct ClaimEntry {
    content: String,
    sources: Vec<String>,
    empirical_level: EmpiricalLevel,
    tags: Vec<String>,
}

impl SearchEngine {
    /// Create an empty search engine.
    pub fn new() -> Self {
        Self {
            claims: Vec::new(),
            vectors: Vec::new(),
            word_df: std::collections::HashMap::new(),
        }
    }

    /// Create a search engine pre-loaded with claims from plexus-ingest.
    pub fn with_seed_claims() -> Self {
        let mut engine = Self::new();
        let all_claims = plexus_ingest::load_all_claims();

        // Build document frequency map
        for claim in &all_claims {
            let lowered = claim.content.to_lowercase();
            let words: std::collections::HashSet<&str> = lowered
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() >= 2 && !is_stop_word(w))
                .collect();
            for word in words {
                *engine.word_df.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Encode with IDF weighting
        let n_docs = all_claims.len() as f32;
        for claim in &all_claims {
            let hv = encode_text_idf(&claim.content, &engine.word_df, n_docs);
            engine.claims.push(ClaimEntry {
                content: claim.content.clone(),
                sources: claim.sources.clone(),
                empirical_level: claim.empirical_level,
                tags: claim.tags.clone(),
            });
            engine.vectors.push(hv);
        }

        log::info!("Search engine loaded {} claims", engine.claims.len());
        engine
    }

    /// Add a claim to the index.
    pub fn add_claim(
        &mut self,
        content: &str,
        empirical_level: EmpiricalLevel,
        sources: &[&str],
        tags: &[&str],
    ) {
        let hv = encode_text(content); // Fallback to non-IDF for dynamic adds
        self.claims.push(ClaimEntry {
            content: content.to_string(),
            sources: sources.iter().map(|s| s.to_string()).collect(),
            empirical_level,
            tags: tags.iter().map(|s| s.to_string()).collect(),
        });
        self.vectors.push(hv);
    }

    /// Search for claims matching a query. Returns top-k results ranked by composite score.
    ///
    /// Uses a two-pass approach for speed:
    /// 1. Fast hash-based pre-filter: sample 4 bytes from the query vector,
    ///    skip claims that differ in more than 3 of 4 sample positions
    /// 2. Exact similarity only on candidates that pass the pre-filter
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        if self.vectors.is_empty() || query.trim().is_empty() {
            return Vec::new();
        }

        let query_hv = if self.word_df.is_empty() {
            encode_text(query)
        } else {
            encode_text_idf(query, &self.word_df, self.claims.len() as f32)
        };

        // Pre-filter: sample 8 byte positions spread across the 2048-byte vector
        // Only compute full similarity for vectors that share at least some bits
        let sample_positions: [usize; 8] = [0, 256, 512, 768, 1024, 1280, 1536, 1792];
        let query_samples: [u8; 8] = sample_positions.map(|p| query_hv.0[p]);

        let candidates: Vec<usize> = if self.vectors.len() > 200 {
            // Pre-filter: keep vectors with >= 2/8 matching sample bytes
            self.vectors.iter().enumerate().filter_map(|(i, v)| {
                let matches = sample_positions.iter().enumerate()
                    .filter(|&(j, &p)| (v.0[p] & query_samples[j]).count_ones() > 2)
                    .count();
                if matches >= 2 { Some(i) } else { None }
            }).collect()
        } else {
            (0..self.vectors.len()).collect()
        };

        // Exact similarity on candidates only
        let baseline = 0.50_f32; // True baseline for 16,384-bit BinaryHV
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&i| {
                let sim = query_hv.similarity(&self.vectors[i]);
                let claim = &self.claims[i];
                let epistemic = claim.empirical_level.as_f32();
                let norm_sim = ((sim - baseline) / (1.0 - baseline)).clamp(0.0, 1.0);
                let composite = 0.7 * norm_sim + 0.2 * epistemic + 0.1 * 0.9;
                (i, composite)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(i, _)| {
                let claim = &self.claims[i];
                let sim = query_hv.similarity(&self.vectors[i]);
                let norm_sim = ((sim - baseline) / (1.0 - baseline)).clamp(0.0, 1.0);
                SearchResult {
                    content: claim.content.clone(),
                    sources: claim.sources.clone(),
                    empirical_level: claim.empirical_level,
                    query_similarity: norm_sim,
                    author_reputation: 0.9,
                    age_days: 1,
                    tags: claim.tags.clone(),
                }
            })
            .collect()
    }

    /// Number of indexed claims.
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self::with_seed_claims()
    }
}

/// Encode text with IDF weighting (used for indexing claims when corpus stats available).
fn encode_text_idf(text: &str, word_df: &std::collections::HashMap<String, usize>, n_docs: f32) -> BinaryHV {
    let lowered = text.to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !is_stop_word(w))
        .collect();

    if words.is_empty() {
        return BinaryHV::random(0);
    }

    // Weight each word by its IDF (rare words matter more)
    let mut hvs: Vec<BinaryHV> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    for word in &words {
        let df = *word_df.get(*word).unwrap_or(&1) as f32;
        let idf = (n_docs / df).ln().max(0.1); // log(N/df), floor at 0.1
        hvs.push(BinaryHV::random(hash_word(word)));
        weights.push(idf);
    }

    // Bigrams (high IDF — very specific)
    for pair in words.windows(2) {
        let bigram_seed = hash_word(pair[0]).wrapping_mul(31).wrapping_add(hash_word(pair[1]));
        hvs.push(BinaryHV::random(bigram_seed));
        weights.push(3.0); // Bigrams get high weight
    }

    if hvs.is_empty() {
        BinaryHV::random(0)
    } else {
        BinaryHV::weighted_bundle(&hvs, &weights)
    }
}

/// Encode text to a 16,384-bit BinaryHV using improved word-level random indexing.
///
/// Improvements over naive encoding:
/// 1. Stop word filtering (removes noise from common words)
/// 2. Bigram encoding (adjacent word pairs for phrase matching)
/// 3. No position binding for short queries (reduces noise)
/// 4. Content words weighted higher than function words
pub fn encode_text(text: &str) -> BinaryHV {
    let lowered = text.to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !is_stop_word(w))
        .collect();

    if words.is_empty() {
        return BinaryHV::random(0);
    }

    let mut hvs: Vec<BinaryHV> = Vec::with_capacity(words.len() * 2);

    // Unigrams (individual content words)
    for word in &words {
        hvs.push(BinaryHV::random(hash_word(word)));
    }

    // Bigrams (adjacent word pairs — captures "ocean acidification", "quantum physics")
    for pair in words.windows(2) {
        let bigram_seed = hash_word(pair[0]).wrapping_mul(31).wrapping_add(hash_word(pair[1]));
        hvs.push(BinaryHV::random(bigram_seed));
    }

    BinaryHV::bundle(&hvs)
}

/// Common English stop words that add noise to semantic encoding.
fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "the" | "is" | "are" | "was" | "were" | "be" | "been" | "being"
            | "have" | "has" | "had" | "do" | "does" | "did"
            | "will" | "would" | "could" | "should" | "may" | "might"
            | "shall" | "can" | "need" | "must"
            | "am" | "it" | "its" | "in" | "on" | "at" | "to" | "for"
            | "of" | "with" | "by" | "from" | "as" | "into" | "about"
            | "an" | "and" | "or" | "but" | "not" | "no" | "nor"
            | "so" | "if" | "than" | "too" | "very" | "just"
            | "that" | "this" | "these" | "those" | "what" | "which"
            | "who" | "whom" | "how" | "when" | "where" | "why"
            | "all" | "each" | "every" | "both" | "few" | "more"
            | "most" | "other" | "some" | "such" | "only" | "then"
    )
}

/// Deterministic hash of a word to a u64 seed for BinaryHV::random().
fn hash_word(word: &str) -> u64 {
    // FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in word.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Generate HTML for search results page.
pub fn render_search_results_html(query: &str, results: &[SearchResult]) -> String {
    let mut results_html = String::new();

    if results.is_empty() {
        results_html.push_str("<p>No results found. Try different keywords.</p>");
    } else {
        for (i, r) in results.iter().enumerate() {
            let e_badge = match r.empirical_level {
                EmpiricalLevel::E0 => "E0 Unverified",
                EmpiricalLevel::E1 => "E1 Preliminary",
                EmpiricalLevel::E2 => "E2 Tested",
                EmpiricalLevel::E3 => "E3 Replicated",
                EmpiricalLevel::E4 => "E4 Established",
            };
            let score_pct = (r.rank_score() * 100.0) as u32;
            let source = r.sources.first().map(|s| s.as_str()).unwrap_or("—");
            let tags = r.tags.join(", ");

            results_html.push_str(&format!(
                "<p><strong>{}. [{}] ({}%)</strong> {}</p>\n<p>Source: {} | Tags: {}</p>\n",
                i + 1,
                e_badge,
                score_pct,
                r.content,
                source,
                tags,
            ));
        }
    }

    format!(
        r#"<!DOCTYPE html>
<html>
<head><title>Search: {query}</title></head>
<body>
    <h1>Prism Search</h1>
    <h2>Results for "{query}"</h2>
    <p>{count} results from {total} indexed claims</p>
    <hr>
    {results_html}
</body>
</html>"#,
        query = query,
        count = results.len(),
        total = "seed",
        results_html = results_html,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_deterministic() {
        let hv1 = encode_text("hello world");
        let hv2 = encode_text("hello world");
        assert!(
            hv1.similarity(&hv2) > 0.99,
            "Same text should produce identical vectors"
        );
    }

    #[test]
    fn encode_similar_texts_correlate() {
        let hv1 = encode_text("ocean acidification causes");
        let hv2 = encode_text("what causes ocean acidification");
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.3,
            "Similar texts should have similarity > 0.3, got {}",
            sim
        );
    }

    #[test]
    fn encode_different_texts_diverge() {
        let hv1 = encode_text("quantum physics experiments");
        let hv2 = encode_text("chocolate cake recipe");
        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.6,
            "Unrelated texts should have low similarity, got {}",
            sim
        );
    }

    #[test]
    fn search_returns_relevant_results() {
        let engine = SearchEngine::with_seed_claims();
        let results = engine.search("ocean acidification", 5);
        assert!(!results.is_empty(), "Should find results for ocean acidification");
        // Top result should mention ocean or CO2
        let top = &results[0];
        let content_lower = top.content.to_lowercase();
        assert!(
            content_lower.contains("ocean") || content_lower.contains("co2") || content_lower.contains("acid"),
            "Top result should be relevant: {}",
            top.content
        );
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let engine = SearchEngine::with_seed_claims();
        assert!(engine.search("", 5).is_empty());
        assert!(engine.search("   ", 5).is_empty());
    }

    #[test]
    fn search_results_are_ranked() {
        let engine = SearchEngine::with_seed_claims();
        let results = engine.search("programming language memory safety", 10);
        if results.len() >= 2 {
            assert!(
                results[0].rank_score() >= results[1].rank_score(),
                "Results should be ranked by score"
            );
        }
    }

    #[test]
    fn seed_claims_loaded() {
        let engine = SearchEngine::with_seed_claims();
        assert!(engine.claim_count() >= 50, "Should have at least 50 seed claims, got {}", engine.claim_count());
    }

    #[test]
    fn render_results_html_valid() {
        let engine = SearchEngine::with_seed_claims();
        let results = engine.search("rust programming", 3);
        let html = render_search_results_html("rust programming", &results);
        assert!(html.contains("Prism Search"));
        assert!(html.contains("rust programming"));
    }
}
