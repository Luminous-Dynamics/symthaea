// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio Similarity Search
//!
//! High-performance vector search using HNSW for finding similar audio content.
//! Supports audio embeddings, fingerprints, and feature vectors.

use ndarray::Array1;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

pub mod index;
pub mod embedding;
pub mod query;

// Re-export commonly used types
pub use index::{HnswIndex, HnswConfig, IndexStats};
pub use embedding::{EmbeddingGenerator, EmbeddingModel};

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, SearchError>;

/// Vector embedding for audio content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEmbedding {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub metadata: EmbeddingMetadata,
}

impl AudioEmbedding {
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            vector,
            metadata: EmbeddingMetadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: EmbeddingMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &AudioEmbedding) -> f32 {
        cosine_similarity(&self.vector, &other.vector)
    }

    /// Compute Euclidean distance
    pub fn euclidean_distance(&self, other: &AudioEmbedding) -> f32 {
        euclidean_distance(&self.vector, &other.vector)
    }

    /// Normalize the embedding vector
    pub fn normalize(&mut self) {
        let norm = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }
}

/// Metadata associated with an embedding
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    pub track_id: Option<String>,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub duration_ms: Option<u64>,
    pub genre: Option<String>,
    pub year: Option<u32>,
    pub tags: Vec<String>,
    /// Custom key-value pairs
    pub custom: HashMap<String, String>,
}

impl EmbeddingMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_track(mut self, track_id: impl Into<String>) -> Self {
        self.track_id = Some(track_id.into());
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_artist(mut self, artist: impl Into<String>) -> Self {
        self.artist = Some(artist.into());
        self
    }
}

/// Search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub score: f32,
    pub distance: f32,
    pub metadata: EmbeddingMetadata,
}

impl SearchResult {
    pub fn new(id: Uuid, score: f32, distance: f32, metadata: EmbeddingMetadata) -> Self {
        Self {
            id,
            score,
            distance,
            metadata,
        }
    }
}

/// Distance metric for similarity search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (converted to distance)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Dot product similarity
    DotProduct,
}

impl DistanceMetric {
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => 1.0 - cosine_similarity(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::Manhattan => manhattan_distance(a, b),
            DistanceMetric::DotProduct => -dot_product(a, b),
        }
    }

    pub fn to_similarity(&self, distance: f32) -> f32 {
        match self {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
            DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
            DistanceMetric::DotProduct => -distance,
        }
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-6 {
        dot / denom
    } else {
        0.0
    }
}

/// Compute Euclidean distance
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Compute Manhattan distance
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum()
}

/// Compute dot product
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Search index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Maximum connections per node (M parameter)
    pub max_connections: usize,
    /// Size of dynamic candidate list (ef_construction)
    pub ef_construction: usize,
    /// Search expansion factor (ef_search)
    pub ef_search: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 256,
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

impl IndexConfig {
    pub fn for_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Simple in-memory vector index using brute force
/// (Production would use HNSW from the hnsw crate)
pub struct VectorIndex {
    config: IndexConfig,
    embeddings: RwLock<HashMap<Uuid, AudioEmbedding>>,
}

impl VectorIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            embeddings: RwLock::new(HashMap::new()),
        }
    }

    /// Add embedding to index
    pub fn add(&self, embedding: AudioEmbedding) -> Result<Uuid> {
        if embedding.dimension() != self.config.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.config.dimension,
                actual: embedding.dimension(),
            });
        }

        let id = embedding.id;
        self.embeddings.write().insert(id, embedding);
        Ok(id)
    }

    /// Add multiple embeddings
    pub fn add_batch(&self, embeddings: Vec<AudioEmbedding>) -> Result<Vec<Uuid>> {
        let mut ids = Vec::with_capacity(embeddings.len());
        for emb in embeddings {
            ids.push(self.add(emb)?);
        }
        Ok(ids)
    }

    /// Remove embedding from index
    pub fn remove(&self, id: &Uuid) -> Result<()> {
        self.embeddings
            .write()
            .remove(id)
            .ok_or_else(|| SearchError::NotFound(id.to_string()))?;
        Ok(())
    }

    /// Search for similar embeddings
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let embeddings = self.embeddings.read();

        // Compute distances to all embeddings
        let mut results: Vec<_> = embeddings
            .values()
            .map(|emb| {
                let distance = self.config.metric.compute(query, &emb.vector);
                let score = self.config.metric.to_similarity(distance);
                SearchResult::new(emb.id, score, distance, emb.metadata.clone())
            })
            .collect();

        // Sort by distance (ascending)
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Take top k
        results.truncate(k);

        Ok(results)
    }

    /// Get embedding by ID
    pub fn get(&self, id: &Uuid) -> Option<AudioEmbedding> {
        self.embeddings.read().get(id).cloned()
    }

    /// Get index size
    pub fn len(&self) -> usize {
        self.embeddings.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.read().is_empty()
    }

    /// Get index configuration
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }
}

/// Search engine with multiple indexes
pub struct SearchEngine {
    indexes: HashMap<String, Arc<VectorIndex>>,
    default_index: String,
}

impl SearchEngine {
    pub fn new(default_config: IndexConfig) -> Self {
        let mut indexes = HashMap::new();
        let default_index = Arc::new(VectorIndex::new(default_config));
        indexes.insert("default".to_string(), default_index);

        Self {
            indexes,
            default_index: "default".to_string(),
        }
    }

    /// Create a new named index
    pub fn create_index(&mut self, name: impl Into<String>, config: IndexConfig) {
        let name = name.into();
        self.indexes.insert(name, Arc::new(VectorIndex::new(config)));
    }

    /// Get index by name
    pub fn get_index(&self, name: &str) -> Option<Arc<VectorIndex>> {
        self.indexes.get(name).cloned()
    }

    /// Get default index
    pub fn default_index(&self) -> Arc<VectorIndex> {
        self.indexes.get(&self.default_index).unwrap().clone()
    }

    /// Add embedding to default index
    pub fn add(&self, embedding: AudioEmbedding) -> Result<Uuid> {
        self.default_index().add(embedding)
    }

    /// Search default index
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.default_index().search(query, k)
    }

    /// Multi-index search (combines results from multiple indexes)
    pub fn search_multi(
        &self,
        index_names: &[&str],
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut all_results = Vec::new();

        for name in index_names {
            if let Some(index) = self.indexes.get(*name) {
                if let Ok(results) = index.search(query, k) {
                    all_results.extend(results);
                }
            }
        }

        // Sort combined results by distance
        all_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        all_results.truncate(k);

        Ok(all_results)
    }
}

/// Similar track finder using embeddings
pub struct SimilarTrackFinder {
    engine: SearchEngine,
}

impl SimilarTrackFinder {
    pub fn new(dimension: usize) -> Self {
        let config = IndexConfig::for_dimension(dimension)
            .with_metric(DistanceMetric::Cosine);

        Self {
            engine: SearchEngine::new(config),
        }
    }

    /// Index a track with its embedding
    pub fn index_track(
        &self,
        track_id: impl Into<String>,
        title: impl Into<String>,
        artist: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Result<Uuid> {
        let metadata = EmbeddingMetadata::new()
            .with_track(track_id)
            .with_title(title)
            .with_artist(artist);

        let emb = AudioEmbedding::new(embedding).with_metadata(metadata);
        self.engine.add(emb)
    }

    /// Find similar tracks
    pub fn find_similar(&self, embedding: &[f32], k: usize) -> Result<Vec<SimilarTrack>> {
        let results = self.engine.search(embedding, k)?;

        Ok(results
            .into_iter()
            .map(|r| SimilarTrack {
                track_id: r.metadata.track_id.unwrap_or_default(),
                title: r.metadata.title.unwrap_or_default(),
                artist: r.metadata.artist.unwrap_or_default(),
                similarity: r.score,
            })
            .collect())
    }

    /// Find similar tracks by track ID (requires the track to be indexed)
    pub fn find_similar_to(&self, id: &Uuid, k: usize) -> Result<Vec<SimilarTrack>> {
        let embedding = self
            .engine
            .default_index()
            .get(id)
            .ok_or_else(|| SearchError::NotFound(id.to_string()))?;

        // Search for k+1 since the query track will be in results
        let mut results = self.find_similar(&embedding.vector, k + 1)?;

        // Remove the query track from results
        results.retain(|r| r.track_id != embedding.metadata.track_id.as_deref().unwrap_or(""));

        results.truncate(k);
        Ok(results)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarTrack {
    pub track_id: String,
    pub title: String,
    pub artist: String,
    pub similarity: f32,
}

/// Playlist generator using similarity search
pub struct PlaylistGenerator {
    finder: SimilarTrackFinder,
}

impl PlaylistGenerator {
    pub fn new(dimension: usize) -> Self {
        Self {
            finder: SimilarTrackFinder::new(dimension),
        }
    }

    /// Generate a playlist starting from seed tracks
    pub fn generate_playlist(
        &self,
        seed_embeddings: &[Vec<f32>],
        target_length: usize,
        diversity: f32,
    ) -> Result<Vec<SimilarTrack>> {
        let mut playlist = Vec::new();
        let mut used_ids = std::collections::HashSet::new();

        // Average seed embeddings
        let dimension = seed_embeddings.first().map(|e| e.len()).unwrap_or(256);
        let mut query = vec![0.0f32; dimension];

        for emb in seed_embeddings {
            for (i, &v) in emb.iter().enumerate() {
                if i < query.len() {
                    query[i] += v;
                }
            }
        }

        let n = seed_embeddings.len() as f32;
        for v in &mut query {
            *v /= n;
        }

        // Find similar tracks iteratively
        while playlist.len() < target_length {
            let candidates = self.finder.find_similar(&query, 20)?;

            for candidate in candidates {
                if !used_ids.contains(&candidate.track_id) {
                    used_ids.insert(candidate.track_id.clone());
                    playlist.push(candidate);

                    if playlist.len() >= target_length {
                        break;
                    }
                }
            }

            // Diversify by adding noise to query
            for v in &mut query {
                *v += (rand_float() - 0.5) * diversity;
            }
        }

        Ok(playlist)
    }
}

/// Simple pseudo-random float (for diversification)
fn rand_float() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 1000) as f32 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_index() {
        let config = IndexConfig::for_dimension(4);
        let index = VectorIndex::new(config);

        let emb1 = AudioEmbedding::new(vec![1.0, 0.0, 0.0, 0.0]);
        let emb2 = AudioEmbedding::new(vec![0.9, 0.1, 0.0, 0.0]);
        let emb3 = AudioEmbedding::new(vec![0.0, 1.0, 0.0, 0.0]);

        index.add(emb1).unwrap();
        index.add(emb2).unwrap();
        index.add(emb3).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        // Most similar should be first
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_audio_embedding() {
        let mut emb = AudioEmbedding::new(vec![3.0, 4.0]);
        emb.normalize();

        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }
}
