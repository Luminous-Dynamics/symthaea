// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced query capabilities

use crate::{AudioEmbedding, DistanceMetric, Result, SearchError, SearchResult, VectorIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query builder for constructing complex searches
pub struct QueryBuilder {
    vector: Option<Vec<f32>>,
    k: usize,
    filters: Vec<Filter>,
    boost_factors: HashMap<String, f32>,
    min_score: Option<f32>,
    include_metadata: bool,
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            vector: None,
            k: 10,
            filters: Vec::new(),
            boost_factors: HashMap::new(),
            min_score: None,
            include_metadata: true,
        }
    }

    /// Set query vector
    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Set number of results to return
    pub fn limit(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Add metadata filter
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Filter by genre
    pub fn genre(self, genre: impl Into<String>) -> Self {
        self.filter(Filter::Equals {
            field: "genre".to_string(),
            value: genre.into(),
        })
    }

    /// Filter by artist
    pub fn artist(self, artist: impl Into<String>) -> Self {
        self.filter(Filter::Equals {
            field: "artist".to_string(),
            value: artist.into(),
        })
    }

    /// Filter by year range
    pub fn year_range(self, min: u32, max: u32) -> Self {
        self.filter(Filter::Range {
            field: "year".to_string(),
            min: min as f32,
            max: max as f32,
        })
    }

    /// Boost results matching field
    pub fn boost(mut self, field: impl Into<String>, factor: f32) -> Self {
        self.boost_factors.insert(field.into(), factor);
        self
    }

    /// Set minimum score threshold
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Include/exclude metadata in results
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Execute query against index
    pub fn execute(&self, index: &VectorIndex) -> Result<Vec<SearchResult>> {
        let vector = self
            .vector
            .as_ref()
            .ok_or_else(|| SearchError::QueryError("No query vector provided".to_string()))?;

        // Get initial results
        let mut results = index.search(vector, self.k * 2)?;

        // Apply filters
        results = self.apply_filters(results);

        // Apply boosts
        results = self.apply_boosts(results);

        // Apply minimum score threshold
        if let Some(min) = self.min_score {
            results.retain(|r| r.score >= min);
        }

        // Truncate to k
        results.truncate(self.k);

        // Clear metadata if not requested
        if !self.include_metadata {
            for result in &mut results {
                result.metadata = Default::default();
            }
        }

        Ok(results)
    }

    fn apply_filters(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter(|r| self.filters.iter().all(|f| f.matches(&r.metadata)))
            .collect()
    }

    fn apply_boosts(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
        for result in &mut results {
            for (field, &factor) in &self.boost_factors {
                if self.metadata_matches_field(&result.metadata, field) {
                    result.score *= factor;
                }
            }
        }

        // Re-sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    fn metadata_matches_field(
        &self,
        metadata: &crate::EmbeddingMetadata,
        field: &str,
    ) -> bool {
        match field {
            "genre" => metadata.genre.is_some(),
            "artist" => metadata.artist.is_some(),
            "album" => metadata.album.is_some(),
            "year" => metadata.year.is_some(),
            _ => metadata.custom.contains_key(field),
        }
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter for metadata-based filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// Exact match
    Equals { field: String, value: String },
    /// Numeric range
    Range { field: String, min: f32, max: f32 },
    /// Contains substring
    Contains { field: String, value: String },
    /// In list of values
    In { field: String, values: Vec<String> },
    /// Not equal
    NotEquals { field: String, value: String },
    /// And combination
    And(Vec<Filter>),
    /// Or combination
    Or(Vec<Filter>),
}

impl Filter {
    pub fn matches(&self, metadata: &crate::EmbeddingMetadata) -> bool {
        match self {
            Filter::Equals { field, value } => {
                self.get_field_value(metadata, field)
                    .map(|v| v == *value)
                    .unwrap_or(false)
            }
            Filter::Range { field, min, max } => {
                self.get_numeric_field(metadata, field)
                    .map(|v| v >= *min && v <= *max)
                    .unwrap_or(false)
            }
            Filter::Contains { field, value } => {
                self.get_field_value(metadata, field)
                    .map(|v| v.to_lowercase().contains(&value.to_lowercase()))
                    .unwrap_or(false)
            }
            Filter::In { field, values } => {
                self.get_field_value(metadata, field)
                    .map(|v| values.contains(&v))
                    .unwrap_or(false)
            }
            Filter::NotEquals { field, value } => {
                self.get_field_value(metadata, field)
                    .map(|v| v != *value)
                    .unwrap_or(true)
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }

    fn get_field_value(&self, metadata: &crate::EmbeddingMetadata, field: &str) -> Option<String> {
        match field {
            "track_id" => metadata.track_id.clone(),
            "title" => metadata.title.clone(),
            "artist" => metadata.artist.clone(),
            "album" => metadata.album.clone(),
            "genre" => metadata.genre.clone(),
            "year" => metadata.year.map(|y| y.to_string()),
            _ => metadata.custom.get(field).cloned(),
        }
    }

    fn get_numeric_field(&self, metadata: &crate::EmbeddingMetadata, field: &str) -> Option<f32> {
        match field {
            "year" => metadata.year.map(|y| y as f32),
            "duration_ms" => metadata.duration_ms.map(|d| d as f32),
            _ => metadata.custom.get(field).and_then(|v| v.parse().ok()),
        }
    }
}

/// Multi-vector query for hybrid search
pub struct HybridQuery {
    vectors: Vec<(Vec<f32>, f32)>, // (vector, weight)
    k: usize,
}

impl HybridQuery {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            k: 10,
        }
    }

    /// Add a query vector with weight
    pub fn add_vector(mut self, vector: Vec<f32>, weight: f32) -> Self {
        self.vectors.push((vector, weight));
        self
    }

    /// Set result limit
    pub fn limit(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Execute hybrid query
    pub fn execute(&self, index: &VectorIndex) -> Result<Vec<SearchResult>> {
        if self.vectors.is_empty() {
            return Err(SearchError::QueryError("No query vectors".to_string()));
        }

        // Get results for each vector
        let mut combined_scores: HashMap<uuid::Uuid, (f32, crate::EmbeddingMetadata)> = HashMap::new();

        for (vector, weight) in &self.vectors {
            let results = index.search(vector, self.k * 2)?;

            for result in results {
                let entry = combined_scores
                    .entry(result.id)
                    .or_insert((0.0, result.metadata.clone()));
                entry.0 += result.score * weight;
            }
        }

        // Normalize scores
        let total_weight: f32 = self.vectors.iter().map(|(_, w)| w).sum();

        let mut results: Vec<_> = combined_scores
            .into_iter()
            .map(|(id, (score, metadata))| {
                SearchResult::new(id, score / total_weight, 1.0 - score / total_weight, metadata)
            })
            .collect();

        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(self.k);

        Ok(results)
    }
}

impl Default for HybridQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Diversified search to reduce redundancy in results
pub struct DiversifiedSearch {
    k: usize,
    diversity: f32, // 0-1, higher = more diverse
}

impl DiversifiedSearch {
    pub fn new(k: usize, diversity: f32) -> Self {
        Self {
            k,
            diversity: diversity.clamp(0.0, 1.0),
        }
    }

    /// Execute search with MMR (Maximal Marginal Relevance)
    pub fn execute(&self, index: &VectorIndex, query: &[f32]) -> Result<Vec<SearchResult>> {
        // Get more candidates than needed
        let candidates = index.search(query, self.k * 5)?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // MMR selection
        let mut selected = Vec::with_capacity(self.k);
        let mut remaining: Vec<_> = candidates.into_iter().collect();

        // Select first result (most relevant)
        selected.push(remaining.remove(0));

        // Select remaining with MMR
        while selected.len() < self.k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::MIN;

            for (i, candidate) in remaining.iter().enumerate() {
                // Relevance to query
                let relevance = candidate.score;

                // Maximum similarity to already selected
                let max_sim = selected
                    .iter()
                    .map(|s| 1.0 - candidate.distance.abs()) // Simplified
                    .fold(0.0f32, f32::max);

                // MMR score
                let mmr = (1.0 - self.diversity) * relevance - self.diversity * max_sim;

                if mmr > best_score {
                    best_score = mmr;
                    best_idx = i;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        Ok(selected)
    }
}

/// Batch query executor for multiple queries
pub struct BatchQueryExecutor {
    queries: Vec<Vec<f32>>,
    k: usize,
}

impl BatchQueryExecutor {
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
            k: 10,
        }
    }

    pub fn add_query(mut self, vector: Vec<f32>) -> Self {
        self.queries.push(vector);
        self
    }

    pub fn limit(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Execute all queries in parallel (simplified sequential for now)
    pub fn execute(&self, index: &VectorIndex) -> Result<Vec<Vec<SearchResult>>> {
        self.queries
            .iter()
            .map(|q| index.search(q, self.k))
            .collect()
    }
}

impl Default for BatchQueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IndexConfig;

    #[test]
    fn test_query_builder() {
        let config = IndexConfig::for_dimension(4);
        let index = VectorIndex::new(config);

        // Add some embeddings
        for i in 0..10 {
            let mut emb = AudioEmbedding::new(vec![i as f32, 0.0, 0.0, 0.0]);
            emb.metadata.genre = Some("rock".to_string());
            index.add(emb).unwrap();
        }

        let results = QueryBuilder::new()
            .vector(vec![5.0, 0.0, 0.0, 0.0])
            .limit(3)
            .genre("rock")
            .execute(&index)
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_filter() {
        let mut metadata = crate::EmbeddingMetadata::new();
        metadata.genre = Some("rock".to_string());
        metadata.year = Some(2020);

        let filter = Filter::Equals {
            field: "genre".to_string(),
            value: "rock".to_string(),
        };
        assert!(filter.matches(&metadata));

        let filter = Filter::Range {
            field: "year".to_string(),
            min: 2015.0,
            max: 2025.0,
        };
        assert!(filter.matches(&metadata));
    }
}
