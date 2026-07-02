// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Similarity Search Engine
//!
//! Provides music discovery through audio embedding similarity:
//! - KNN search for "find similar tracks"
//! - Mood-based playlist generation
//! - Genre clustering
//! - Cross-feature search (mood + tempo + genre)

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, debug};

/// Similarity search errors
#[derive(Debug, Error)]
pub enum SimilarityError {
    #[error("Track not found: {0}")]
    TrackNotFound(String),

    #[error("No embeddings available")]
    NoEmbeddings,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Index not built")]
    IndexNotBuilt,
}

/// Audio features for a track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackFeatures {
    pub track_id: String,
    pub embedding: Vec<f32>,        // Audio embedding vector (e.g., 1280-dim from EffNet)
    pub genre_scores: Vec<f32>,     // Genre classification scores
    pub mood_scores: Vec<f32>,      // Mood classification scores
    pub bpm: Option<f32>,
    pub key: Option<String>,
    pub energy: f32,                // 0-1 energy level
    pub valence: f32,               // 0-1 positivity/happiness
    pub danceability: f32,          // 0-1 danceability score
    pub acousticness: f32,          // 0-1 acoustic vs electronic
    pub instrumentalness: f32,      // 0-1 instrumental vs vocal
}

impl TrackFeatures {
    pub fn new(track_id: String, embedding: Vec<f32>) -> Self {
        Self {
            track_id,
            embedding,
            genre_scores: Vec::new(),
            mood_scores: Vec::new(),
            bpm: None,
            key: None,
            energy: 0.5,
            valence: 0.5,
            danceability: 0.5,
            acousticness: 0.5,
            instrumentalness: 0.5,
        }
    }
}

/// Search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub track_id: String,
    pub score: f32,
    pub distance: f32,
}

/// Search query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Reference track IDs for similarity
    pub seed_tracks: Vec<String>,

    /// Optional: target mood scores (normalized)
    pub target_mood: Option<Vec<f32>>,

    /// Optional: target genre scores (normalized)
    pub target_genre: Option<Vec<f32>>,

    /// Optional: BPM range
    pub bpm_range: Option<(f32, f32)>,

    /// Optional: energy range (0-1)
    pub energy_range: Option<(f32, f32)>,

    /// Optional: valence range (0-1)
    pub valence_range: Option<(f32, f32)>,

    /// Maximum number of results
    pub limit: usize,

    /// Exclude these track IDs from results
    pub exclude: Vec<String>,

    /// Weight for embedding similarity (0-1)
    pub embedding_weight: f32,

    /// Weight for mood similarity (0-1)
    pub mood_weight: f32,

    /// Weight for genre similarity (0-1)
    pub genre_weight: f32,

    /// Weight for audio features (0-1)
    pub feature_weight: f32,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            seed_tracks: Vec::new(),
            target_mood: None,
            target_genre: None,
            bpm_range: None,
            energy_range: None,
            valence_range: None,
            limit: 20,
            exclude: Vec::new(),
            embedding_weight: 0.6,
            mood_weight: 0.2,
            genre_weight: 0.1,
            feature_weight: 0.1,
        }
    }
}

/// Similarity search index
pub struct SimilarityIndex {
    /// Track features storage
    tracks: HashMap<String, TrackFeatures>,

    /// Embedding dimension
    embedding_dim: usize,

    /// Precomputed normalized embeddings for fast search
    normalized_embeddings: Vec<(String, Vec<f32>)>,

    /// Index built flag
    is_built: bool,
}

impl SimilarityIndex {
    /// Create a new similarity index
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            tracks: HashMap::new(),
            embedding_dim,
            normalized_embeddings: Vec::new(),
            is_built: false,
        }
    }

    /// Add a track to the index
    pub fn add_track(&mut self, features: TrackFeatures) -> Result<(), SimilarityError> {
        if features.embedding.len() != self.embedding_dim && !features.embedding.is_empty() {
            return Err(SimilarityError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: features.embedding.len(),
            });
        }

        self.tracks.insert(features.track_id.clone(), features);
        self.is_built = false; // Invalidate index
        Ok(())
    }

    /// Remove a track from the index
    pub fn remove_track(&mut self, track_id: &str) -> bool {
        if self.tracks.remove(track_id).is_some() {
            self.is_built = false;
            true
        } else {
            false
        }
    }

    /// Get track features
    pub fn get_track(&self, track_id: &str) -> Option<&TrackFeatures> {
        self.tracks.get(track_id)
    }

    /// Build the search index (call after adding tracks)
    pub fn build(&mut self) {
        self.normalized_embeddings = self.tracks
            .iter()
            .filter(|(_, f)| !f.embedding.is_empty())
            .map(|(id, f)| {
                let normalized = normalize_vector(&f.embedding);
                (id.clone(), normalized)
            })
            .collect();

        self.is_built = true;
        info!("Built similarity index with {} tracks", self.normalized_embeddings.len());
    }

    /// Search for similar tracks
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SimilarityResult>, SimilarityError> {
        if !self.is_built {
            return Err(SimilarityError::IndexNotBuilt);
        }

        // Get seed track embeddings
        let seed_embeddings: Vec<Vec<f32>> = query.seed_tracks
            .iter()
            .filter_map(|id| self.tracks.get(id))
            .filter(|f| !f.embedding.is_empty())
            .map(|f| normalize_vector(&f.embedding))
            .collect();

        if seed_embeddings.is_empty() && query.target_mood.is_none() && query.target_genre.is_none() {
            return Err(SimilarityError::NoEmbeddings);
        }

        // Average seed embeddings
        let avg_seed_embedding = if !seed_embeddings.is_empty() {
            average_vectors(&seed_embeddings)
        } else {
            vec![0.0; self.embedding_dim]
        };

        // Get seed track features for feature-based similarity
        let seed_features: Vec<&TrackFeatures> = query.seed_tracks
            .iter()
            .filter_map(|id| self.tracks.get(id))
            .collect();

        // Calculate similarity scores for all tracks
        let mut results: Vec<SimilarityResult> = self.tracks
            .iter()
            .filter(|(id, _)| !query.exclude.contains(id) && !query.seed_tracks.contains(id))
            .filter_map(|(id, features)| {
                // Apply filters
                if let Some((min_bpm, max_bpm)) = query.bpm_range {
                    if let Some(bpm) = features.bpm {
                        if bpm < min_bpm || bpm > max_bpm {
                            return None;
                        }
                    }
                }

                if let Some((min_energy, max_energy)) = query.energy_range {
                    if features.energy < min_energy || features.energy > max_energy {
                        return None;
                    }
                }

                if let Some((min_valence, max_valence)) = query.valence_range {
                    if features.valence < min_valence || features.valence > max_valence {
                        return None;
                    }
                }

                // Calculate similarity score
                let mut score = 0.0f32;
                let mut weight_sum = 0.0f32;

                // Embedding similarity
                if query.embedding_weight > 0.0 && !features.embedding.is_empty() && !avg_seed_embedding.is_empty() {
                    let normalized = normalize_vector(&features.embedding);
                    let cosine_sim = cosine_similarity(&avg_seed_embedding, &normalized);
                    score += cosine_sim * query.embedding_weight;
                    weight_sum += query.embedding_weight;
                }

                // Mood similarity
                if query.mood_weight > 0.0 {
                    let mood_sim = if let Some(ref target) = query.target_mood {
                        if !features.mood_scores.is_empty() {
                            cosine_similarity(target, &features.mood_scores)
                        } else {
                            0.0
                        }
                    } else if !seed_features.is_empty() {
                        // Average mood similarity to seeds
                        seed_features.iter()
                            .filter(|sf| !sf.mood_scores.is_empty() && !features.mood_scores.is_empty())
                            .map(|sf| cosine_similarity(&sf.mood_scores, &features.mood_scores))
                            .sum::<f32>() / seed_features.len().max(1) as f32
                    } else {
                        0.0
                    };
                    score += mood_sim * query.mood_weight;
                    weight_sum += query.mood_weight;
                }

                // Genre similarity
                if query.genre_weight > 0.0 {
                    let genre_sim = if let Some(ref target) = query.target_genre {
                        if !features.genre_scores.is_empty() {
                            cosine_similarity(target, &features.genre_scores)
                        } else {
                            0.0
                        }
                    } else if !seed_features.is_empty() {
                        seed_features.iter()
                            .filter(|sf| !sf.genre_scores.is_empty() && !features.genre_scores.is_empty())
                            .map(|sf| cosine_similarity(&sf.genre_scores, &features.genre_scores))
                            .sum::<f32>() / seed_features.len().max(1) as f32
                    } else {
                        0.0
                    };
                    score += genre_sim * query.genre_weight;
                    weight_sum += query.genre_weight;
                }

                // Audio feature similarity
                if query.feature_weight > 0.0 && !seed_features.is_empty() {
                    let feature_sim = seed_features.iter()
                        .map(|sf| audio_feature_similarity(sf, features))
                        .sum::<f32>() / seed_features.len() as f32;
                    score += feature_sim * query.feature_weight;
                    weight_sum += query.feature_weight;
                }

                // Normalize score
                let final_score = if weight_sum > 0.0 { score / weight_sum } else { 0.0 };

                Some(SimilarityResult {
                    track_id: id.clone(),
                    score: final_score,
                    distance: 1.0 - final_score,
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Limit results
        results.truncate(query.limit);

        Ok(results)
    }

    /// Find tracks by mood
    pub fn find_by_mood(&self, mood_vector: &[f32], limit: usize) -> Vec<SimilarityResult> {
        let mut results: Vec<SimilarityResult> = self.tracks
            .iter()
            .filter(|(_, f)| !f.mood_scores.is_empty())
            .map(|(id, f)| {
                let sim = cosine_similarity(mood_vector, &f.mood_scores);
                SimilarityResult {
                    track_id: id.clone(),
                    score: sim,
                    distance: 1.0 - sim,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Find tracks by genre
    pub fn find_by_genre(&self, genre_vector: &[f32], limit: usize) -> Vec<SimilarityResult> {
        let mut results: Vec<SimilarityResult> = self.tracks
            .iter()
            .filter(|(_, f)| !f.genre_scores.is_empty())
            .map(|(id, f)| {
                let sim = cosine_similarity(genre_vector, &f.genre_scores);
                SimilarityResult {
                    track_id: id.clone(),
                    score: sim,
                    distance: 1.0 - sim,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Get track count
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Get all track IDs
    pub fn track_ids(&self) -> Vec<String> {
        self.tracks.keys().cloned().collect()
    }
}

/// Thread-safe similarity index
pub struct SharedSimilarityIndex {
    inner: Arc<RwLock<SimilarityIndex>>,
}

impl SharedSimilarityIndex {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SimilarityIndex::new(embedding_dim))),
        }
    }

    pub fn add_track(&self, features: TrackFeatures) -> Result<(), SimilarityError> {
        self.inner.write().add_track(features)
    }

    pub fn remove_track(&self, track_id: &str) -> bool {
        self.inner.write().remove_track(track_id)
    }

    pub fn build(&self) {
        self.inner.write().build();
    }

    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SimilarityResult>, SimilarityError> {
        self.inner.read().search(query)
    }

    pub fn find_by_mood(&self, mood_vector: &[f32], limit: usize) -> Vec<SimilarityResult> {
        self.inner.read().find_by_mood(mood_vector, limit)
    }

    pub fn find_by_genre(&self, genre_vector: &[f32], limit: usize) -> Vec<SimilarityResult> {
        self.inner.read().find_by_genre(genre_vector, limit)
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

impl Clone for SharedSimilarityIndex {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// === Utility Functions ===

/// Normalize a vector to unit length
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Average multiple vectors
fn average_vectors(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let mut avg = vec![0.0f32; dim];

    for v in vectors {
        for (i, val) in v.iter().enumerate() {
            if i < dim {
                avg[i] += val;
            }
        }
    }

    let n = vectors.len() as f32;
    avg.iter_mut().for_each(|x| *x /= n);
    avg
}

/// Audio feature similarity (energy, valence, etc.)
fn audio_feature_similarity(a: &TrackFeatures, b: &TrackFeatures) -> f32 {
    let mut similarity = 0.0f32;
    let mut count = 0;

    // Energy similarity
    similarity += 1.0 - (a.energy - b.energy).abs();
    count += 1;

    // Valence similarity
    similarity += 1.0 - (a.valence - b.valence).abs();
    count += 1;

    // Danceability similarity
    similarity += 1.0 - (a.danceability - b.danceability).abs();
    count += 1;

    // Acousticness similarity
    similarity += 1.0 - (a.acousticness - b.acousticness).abs();
    count += 1;

    // Instrumentalness similarity
    similarity += 1.0 - (a.instrumentalness - b.instrumentalness).abs();
    count += 1;

    // BPM similarity (if both have BPM)
    if let (Some(bpm_a), Some(bpm_b)) = (a.bpm, b.bpm) {
        // Consider BPMs similar if within 10%
        let bpm_diff = (bpm_a - bpm_b).abs() / bpm_a.max(bpm_b);
        similarity += 1.0 - bpm_diff.min(1.0);
        count += 1;
    }

    similarity / count as f32
}

// === Mood Labels ===

pub const MOOD_LABELS: &[&str] = &[
    "happy", "sad", "energetic", "calm", "aggressive",
    "romantic", "melancholic", "uplifting", "dark", "dreamy",
];

pub const GENRE_LABELS: &[&str] = &[
    "electronic", "rock", "pop", "hip-hop", "jazz",
    "classical", "folk", "r&b", "metal", "ambient",
    "indie", "punk", "blues", "country", "reggae",
];

/// Create mood vector from label
pub fn mood_from_label(label: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; MOOD_LABELS.len()];
    if let Some(idx) = MOOD_LABELS.iter().position(|&l| l == label) {
        vec[idx] = 1.0;
    }
    vec
}

/// Create genre vector from label
pub fn genre_from_label(label: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; GENRE_LABELS.len()];
    if let Some(idx) = GENRE_LABELS.iter().position(|&l| l == label) {
        vec[idx] = 1.0;
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_track(id: &str, embedding: Vec<f32>) -> TrackFeatures {
        let mut features = TrackFeatures::new(id.to_string(), embedding);
        features.mood_scores = vec![0.8, 0.2, 0.5, 0.3, 0.1, 0.4, 0.2, 0.7, 0.1, 0.3];
        features.genre_scores = vec![0.1, 0.3, 0.7, 0.2, 0.1, 0.0, 0.2, 0.3, 0.0, 0.1, 0.4, 0.1, 0.1, 0.0, 0.1];
        features.energy = 0.7;
        features.valence = 0.6;
        features.bpm = Some(120.0);
        features
    }

    #[test]
    fn test_similarity_index() {
        let mut index = SimilarityIndex::new(128);

        // Add some tracks with random embeddings
        for i in 0..10 {
            let embedding: Vec<f32> = (0..128).map(|j| ((i + j) as f32 * 0.01).sin()).collect();
            let features = create_test_track(&format!("track_{}", i), embedding);
            index.add_track(features).unwrap();
        }

        index.build();

        // Search for similar tracks
        let query = SearchQuery {
            seed_tracks: vec!["track_0".to_string()],
            limit: 5,
            ..Default::default()
        };

        let results = index.search(&query).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // First result should have high similarity
        assert!(results[0].score > 0.5);
    }

    #[test]
    fn test_mood_search() {
        let mut index = SimilarityIndex::new(128);

        for i in 0..5 {
            let embedding: Vec<f32> = (0..128).map(|_| 0.0).collect();
            let mut features = TrackFeatures::new(format!("track_{}", i), embedding);
            features.mood_scores = vec![
                if i == 0 { 0.9 } else { 0.1 },
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            ];
            index.add_track(features).unwrap();
        }

        index.build();

        // Search for "happy" mood
        let mood = mood_from_label("happy");
        let results = index.find_by_mood(&mood, 3);

        assert!(!results.is_empty());
        assert_eq!(results[0].track_id, "track_0");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.0001);
    }
}
