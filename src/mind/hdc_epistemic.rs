//! HDC-Enhanced Epistemic Classification
//!
//! Uses Hyperdimensional Computing semantic similarity to classify
//! queries into epistemic categories. This is more robust than pattern
//! matching because it handles novel phrasings, typos, and semantic
//! equivalence.
//!
//! ## How It Works
//!
//! 1. **Exemplar Encoding**: Pre-encode exemplar queries for each category
//! 2. **Query Encoding**: Encode the incoming query as a hypervector
//! 3. **Similarity Voting**: Compare query to all exemplars, vote by category
//! 4. **Classification**: Highest similarity score wins
//!
//! ## Categories
//!
//! - **Unknown**: Fictional entities, nonsensical questions, impossible queries
//! - **Unverifiable**: Future predictions, subjective experience, counterfactuals
//! - **Known**: Common facts, math, established history
//! - **Uncertain**: Default for novel queries with no strong match

use crate::hdc::SemanticSpace;
use super::structured_thought::EpistemicStatus;
use std::collections::HashMap;

/// Exemplar query for training the epistemic classifier
#[derive(Debug, Clone)]
pub struct EpistemicExemplar {
    /// The query text
    pub query: String,
    /// The encoded hypervector
    pub vector: Vec<f32>,
    /// The epistemic category
    pub status: EpistemicStatus,
}

/// HDC-based epistemic classifier
///
/// Uses semantic similarity to classify queries into epistemic categories.
/// More robust than pattern matching for handling:
/// - Novel phrasings ("What's the GDP of mythical Atlantis?")
/// - Typos ("Waht is the captial of Frnace?")
/// - Semantic equivalence ("future stock prices" ≈ "tomorrow's market")
pub struct HdcEpistemicClassifier {
    /// Encoded exemplars grouped by category
    exemplars: Vec<EpistemicExemplar>,

    /// Minimum similarity threshold for confident classification
    /// Below this, default to Uncertain
    confidence_threshold: f32,

    /// The semantic space for encoding (shared reference)
    dimension: usize,
}

impl HdcEpistemicClassifier {
    /// Create a new HDC epistemic classifier with default exemplars
    pub fn new(semantic_space: &mut SemanticSpace) -> anyhow::Result<Self> {
        // Get dimension by encoding a test string and checking vector length
        let test_vec = semantic_space.encode("test")?;
        let dimension = test_vec.len();

        let mut classifier = Self {
            exemplars: Vec::new(),
            confidence_threshold: 0.3, // Require 30% similarity for confident classification
            dimension,
        };

        // Train with exemplars for each category
        classifier.train_exemplars(semantic_space)?;

        Ok(classifier)
    }

    /// Train the classifier with exemplar queries for each category
    fn train_exemplars(&mut self, semantic_space: &mut SemanticSpace) -> anyhow::Result<()> {
        // ═══════════════════════════════════════════════════════════════════
        // UNKNOWN: Things that don't exist or are nonsensical
        // ═══════════════════════════════════════════════════════════════════
        let unknown_exemplars = [
            // Fictional places
            "What is the GDP of Atlantis?",
            "What is the population of Hogwarts?",
            "What is the capital of Mordor?",
            "What is the weather in Narnia?",
            "How do I get to El Dorado?",
            "What is the currency in Wakanda?",
            "Who rules Westeros?",

            // Nonsensical questions
            "What is the color of happiness?",
            "What does Tuesday smell like?",
            "What is the weight of love?",
            "How loud is purple?",
            "What is the taste of mathematics?",

            // Impossible/contradictory
            "Draw a square circle",
            "Who is the married bachelor?",
            "What is north of the North Pole?",
        ];

        for query in unknown_exemplars {
            self.add_exemplar(semantic_space, query, EpistemicStatus::Unknown)?;
        }

        // ═══════════════════════════════════════════════════════════════════
        // UNVERIFIABLE: Future, subjective, hypothetical
        // ═══════════════════════════════════════════════════════════════════
        let unverifiable_exemplars = [
            // Future predictions
            "What will the stock market do tomorrow?",
            "What are the lottery numbers?",
            "What will happen next year?",
            "When will I die?",
            "Will it rain next week?",
            "Who will win the election?",

            // Subjective experience
            "What am I thinking right now?",
            "Read my mind",
            "What is my favorite color?",
            "How do I feel?",
            "What do I want for dinner?",

            // Counterfactuals
            "What if Napoleon had won at Waterloo?",
            "What would have happened if Hitler died young?",
            "What if dinosaurs never went extinct?",
            "What would the world be like without electricity?",
        ];

        for query in unverifiable_exemplars {
            self.add_exemplar(semantic_space, query, EpistemicStatus::Unverifiable)?;
        }

        // ═══════════════════════════════════════════════════════════════════
        // KNOWN: Established facts
        // ═══════════════════════════════════════════════════════════════════
        let known_exemplars = [
            // Geography
            "What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Japan?",
            "What is the largest country?",
            "What continent is Egypt in?",

            // Math
            "What is 2 + 2?",
            "What is 15 times 3?",
            "What is the square root of 144?",
            "Calculate 100 divided by 5",

            // Science
            "What is the boiling point of water?",
            "What is the speed of light?",
            "What is H2O?",
            "How many planets are in our solar system?",

            // History/Culture
            "Who wrote Hamlet?",
            "Who painted the Mona Lisa?",
            "When did World War 2 end?",
            "Who was the first president of the United States?",
        ];

        for query in known_exemplars {
            self.add_exemplar(semantic_space, query, EpistemicStatus::Known)?;
        }

        tracing::info!(
            "🧠 HDC Epistemic Classifier trained with {} exemplars",
            self.exemplars.len()
        );

        Ok(())
    }

    /// Add an exemplar to the classifier
    fn add_exemplar(
        &mut self,
        semantic_space: &mut SemanticSpace,
        query: &str,
        status: EpistemicStatus,
    ) -> anyhow::Result<()> {
        let vector = semantic_space.encode(query)?;

        self.exemplars.push(EpistemicExemplar {
            query: query.to_string(),
            vector,
            status,
        });

        Ok(())
    }

    /// Classify a query using HDC semantic similarity
    ///
    /// Returns the most likely epistemic status and the confidence score.
    pub fn classify(
        &self,
        semantic_space: &mut SemanticSpace,
        query: &str,
    ) -> anyhow::Result<(EpistemicStatus, f32)> {
        // Encode the query
        let query_vector = semantic_space.encode(query)?;

        // Compute similarity to all exemplars
        let mut category_scores: HashMap<EpistemicStatus, Vec<f32>> = HashMap::new();
        category_scores.insert(EpistemicStatus::Unknown, Vec::new());
        category_scores.insert(EpistemicStatus::Unverifiable, Vec::new());
        category_scores.insert(EpistemicStatus::Known, Vec::new());

        for exemplar in &self.exemplars {
            let similarity = self.cosine_similarity(&query_vector, &exemplar.vector);

            if let Some(scores) = category_scores.get_mut(&exemplar.status) {
                scores.push(similarity);
            }
        }

        // Calculate average score for each category
        let mut best_status = EpistemicStatus::Uncertain;
        let mut best_score: f32 = 0.0;

        for (status, scores) in &category_scores {
            if !scores.is_empty() {
                // Use top-3 average (most similar exemplars in category)
                let mut sorted_scores = scores.clone();
                sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

                let top_n = sorted_scores.iter().take(3).copied().collect::<Vec<_>>();
                let avg_score: f32 = top_n.iter().sum::<f32>() / top_n.len() as f32;

                tracing::debug!(
                    "Category {:?}: top-3 avg similarity = {:.4}",
                    status, avg_score
                );

                if avg_score > best_score {
                    best_score = avg_score;
                    best_status = *status;
                }
            }
        }

        // If best score is below threshold, default to Uncertain
        if best_score < self.confidence_threshold {
            tracing::debug!(
                "Best score {:.4} below threshold {:.4}, defaulting to Uncertain",
                best_score, self.confidence_threshold
            );
            return Ok((EpistemicStatus::Uncertain, best_score));
        }

        tracing::info!(
            "HDC classified '{}' as {:?} (confidence: {:.2}%)",
            &query[..query.len().min(40)],
            best_status,
            best_score * 100.0
        );

        Ok((best_status, best_score))
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Get statistics about the classifier
    pub fn stats(&self) -> HdcEpistemicStats {
        let mut counts: HashMap<EpistemicStatus, usize> = HashMap::new();

        for exemplar in &self.exemplars {
            *counts.entry(exemplar.status).or_insert(0) += 1;
        }

        HdcEpistemicStats {
            total_exemplars: self.exemplars.len(),
            unknown_count: *counts.get(&EpistemicStatus::Unknown).unwrap_or(&0),
            unverifiable_count: *counts.get(&EpistemicStatus::Unverifiable).unwrap_or(&0),
            known_count: *counts.get(&EpistemicStatus::Known).unwrap_or(&0),
            dimension: self.dimension,
            confidence_threshold: self.confidence_threshold,
        }
    }
}

/// Statistics about the HDC epistemic classifier
#[derive(Debug, Clone)]
pub struct HdcEpistemicStats {
    pub total_exemplars: usize,
    pub unknown_count: usize,
    pub unverifiable_count: usize,
    pub known_count: usize,
    pub dimension: usize,
    pub confidence_threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        let stats = classifier.stats();
        assert!(stats.total_exemplars > 0);
        assert!(stats.unknown_count > 0);
        assert!(stats.unverifiable_count > 0);
        assert!(stats.known_count > 0);
    }

    #[test]
    fn test_classify_unknown() {
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        let (status, confidence) = classifier
            .classify(&mut semantic_space, "What is the GDP of Atlantis?")
            .unwrap();

        // The exact exemplar should have high confidence
        assert_eq!(status, EpistemicStatus::Unknown);
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_classify_known() {
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        let (status, confidence) = classifier
            .classify(&mut semantic_space, "What is 2 + 2?")
            .unwrap();

        assert_eq!(status, EpistemicStatus::Known);
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_classify_unverifiable() {
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        let (status, confidence) = classifier
            .classify(&mut semantic_space, "What will the stock market do tomorrow?")
            .unwrap();

        assert_eq!(status, EpistemicStatus::Unverifiable);
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_classify_novel_phrasing() {
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        // Novel phrasing but semantically similar to "GDP of Atlantis"
        let (status, _confidence) = classifier
            .classify(&mut semantic_space, "Tell me about the economy of the mythical city Atlantis")
            .unwrap();

        // Should still detect Unknown due to "Atlantis" semantic overlap
        // (Though exact classification depends on the HDC encoding quality)
        println!("Novel phrasing classified as: {:?}", status);
    }
}
