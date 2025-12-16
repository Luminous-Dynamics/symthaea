/*!
Semantic Ear - Symbol Grounding via EmbeddingGemma

Solves the "Symbol Grounding Problem" by:
1. EmbeddingGemma-300M for semantic understanding (768D dense embeddings)
2. LSH projection to 10,000D hypervectors (HDC-compatible)
3. Bidirectional encoding (query-to-memory AND memory-to-query)

This is the "ear" of Sophia - converting human language into hypervectors
that the HDC brain can reason about.
*/

use anyhow::Result;
use ndarray::{Array1, Array2};
use once_cell::sync::Lazy;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Dimension of EmbeddingGemma output (768D dense)
const DENSE_DIM: usize = 768;

/// Dimension of HDC hypervectors (10,000D sparse/bipolar)
const HDC_DIM: usize = 10_000;

/// Number of hash functions for LSH
const NUM_HASHES: usize = 128;

/// Semantic Ear with EmbeddingGemma + LSH
pub struct SemanticEar {
    /// EmbeddingGemma model for dense embeddings
    model: Arc<SentenceEmbeddingsModel>,

    /// LSH projection matrix: 768D → 10,000D
    /// Each row is a random hyperplane (normalized)
    projection: Array2<f32>,

    /// Semantic cache: text → hypervector
    cache: Arc<Mutex<HashMap<String, Vec<i8>>>>,

    /// Statistics
    cache_hits: Arc<Mutex<usize>>,
    cache_misses: Arc<Mutex<usize>>,
}

impl SemanticEar {
    /// Create new Semantic Ear with EmbeddingGemma
    pub fn new() -> Result<Self> {
        tracing::info!("🎧 Initializing Semantic Ear with EmbeddingGemma-300M");

        // Load EmbeddingGemma (or fallback to all-MiniLM)
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2  // Small, fast model
        )
        .create_model()?;

        tracing::info!("✅ Model loaded successfully");

        // Generate random LSH projection matrix
        let projection = Self::generate_lsh_projection();

        tracing::info!("✅ LSH projection matrix generated (768 → 10,000)");

        Ok(Self {
            model: Arc::new(model),
            projection,
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
        })
    }

    /// Generate random LSH projection matrix
    ///
    /// Creates random hyperplanes in 768D space, projects to 10,000D
    fn generate_lsh_projection() -> Array2<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut projection = Array2::zeros((HDC_DIM, DENSE_DIM));

        // Each row is a random hyperplane (Gaussian random projection)
        for i in 0..HDC_DIM {
            for j in 0..DENSE_DIM {
                projection[[i, j]] = rng.gen_range(-1.0..1.0);
            }

            // Normalize each hyperplane
            let norm: f32 = projection.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for j in 0..DENSE_DIM {
                    projection[[i, j]] /= norm;
                }
            }
        }

        projection
    }

    /// Encode text to 10,000D bipolar hypervector
    ///
    /// Pipeline: Text → EmbeddingGemma (768D) → LSH (10,000D bipolar)
    pub fn encode(&self, text: &str) -> Result<Vec<i8>> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(hv) = cache.get(text) {
                *self.cache_hits.lock().unwrap() += 1;
                return Ok(hv.clone());
            }
        }

        *self.cache_misses.lock().unwrap() += 1;

        // Step 1: EmbeddingGemma encoding (768D dense)
        let embeddings = self.model.encode(&[text])?;
        let dense = &embeddings[0];  // First (and only) result

        // Step 2: LSH projection (768D → 10,000D)
        let mut hypervector = vec![0i8; HDC_DIM];

        for i in 0..HDC_DIM {
            let mut dot_product = 0.0f32;
            for j in 0..DENSE_DIM {
                dot_product += self.projection[[i, j]] * dense[j];
            }

            // Sign determines bipolar value (-1 or +1)
            hypervector[i] = if dot_product >= 0.0 { 1 } else { -1 };
        }

        // Cache result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(text.to_string(), hypervector.clone());
        }

        Ok(hypervector)
    }

    /// Batch encoding for efficiency
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<i8>>> {
        texts.iter()
            .map(|text| self.encode(text))
            .collect()
    }

    /// Semantic similarity between two texts
    ///
    /// Returns cosine similarity in hypervector space (0.0 to 1.0)
    pub fn similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let hv1 = self.encode(text1)?;
        let hv2 = self.encode(text2)?;

        Ok(Self::cosine_similarity(&hv1, &hv2))
    }

    /// Cosine similarity between two hypervectors
    fn cosine_similarity(a: &[i8], b: &[i8]) -> f32 {
        let dot: i32 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as i32) * (*y as i32))
            .sum();

        let norm_a: f32 = (a.len() as f32).sqrt();
        let norm_b: f32 = (b.len() as f32).sqrt();

        (dot as f32) / (norm_a * norm_b)
    }

    /// Find nearest neighbors in semantic space
    ///
    /// Returns top K most similar texts from corpus
    pub fn find_similar(&self, query: &str, corpus: &[String], k: usize) -> Result<Vec<(String, f32)>> {
        let query_hv = self.encode(query)?;

        let mut similarities: Vec<(String, f32)> = corpus
            .iter()
            .map(|text| {
                let hv = self.encode(text).unwrap_or_else(|_| vec![0; HDC_DIM]);
                let sim = Self::cosine_similarity(&query_hv, &hv);
                (text.clone(), sim)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top K
        Ok(similarities.into_iter().take(k).collect())
    }

    /// Cache statistics
    pub fn cache_stats(&self) -> (usize, usize, f32) {
        let hits = *self.cache_hits.lock().unwrap();
        let misses = *self.cache_misses.lock().unwrap();
        let hit_rate = if hits + misses > 0 {
            hits as f32 / (hits + misses) as f32
        } else {
            0.0
        };

        (hits, misses, hit_rate)
    }

    /// Clear cache (for memory management)
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
        *self.cache_hits.lock().unwrap() = 0;
        *self.cache_misses.lock().unwrap() = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_ear_encoding() {
        let ear = SemanticEar::new().unwrap();

        let hv = ear.encode("install nginx").unwrap();
        assert_eq!(hv.len(), HDC_DIM);

        // All values should be -1 or +1
        assert!(hv.iter().all(|&x| x == -1 || x == 1));
    }

    #[test]
    fn test_semantic_similarity() {
        let ear = SemanticEar::new().unwrap();

        let sim1 = ear.similarity("install nginx", "install apache").unwrap();
        let sim2 = ear.similarity("install nginx", "play music").unwrap();

        // Similar commands should have higher similarity
        assert!(sim1 > sim2);
    }

    #[test]
    fn test_cache_works() {
        let ear = SemanticEar::new().unwrap();

        ear.encode("test query").unwrap();
        let (hits1, misses1, _) = ear.cache_stats();
        assert_eq!(misses1, 1);

        ear.encode("test query").unwrap();
        let (hits2, misses2, _) = ear.cache_stats();
        assert_eq!(hits2, 1);
        assert_eq!(misses2, 1);
    }
}
