//! HDC-Enhanced Epistemic Classification
//!
//! Uses Hyperdimensional Computing semantic similarity to classify
//! queries into epistemic categories. This is more robust than pattern
//! matching because it handles novel phrasings, typos, and semantic
//! equivalence.
//!
//! ## Encoding Modes
//!
//! The classifier supports two encoding backends:
//!
//! 1. **N-gram HDC** (Original): Fast, character-based encoding via SemanticSpace
//! 2. **Semantic BGE** (Enhanced): Deep semantic understanding via SemanticEncoder
//!
//! The Semantic BGE mode provides better novel phrasing handling by using
//! dense 768D embeddings projected to HDC space.
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
//!
//! ## HAM Architecture Integration
//!
//! This classifier is part of the Holographic Associative Memory architecture:
//! - **Sensation**: Query → SemanticEncoder → (DenseVector, Hypervector)
//! - **Perception**: Hypervector → Epistemic Classification
//! - **Memory**: DenseVector stored for future retrieval
//! - **Cognition**: Active inference loop (Surprise = Input - Prediction)

use crate::hdc::SemanticSpace;
use super::structured_thought::EpistemicStatus;
use super::semantic_encoder::{SemanticEncoder, DenseVector};
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
    ///
    /// EXPANDED: Includes synonyms, fragments, and casual phrasings to fill
    /// gaps in the vector space for better novel query handling.
    fn train_exemplars(&mut self, semantic_space: &mut SemanticSpace) -> anyhow::Result<()> {
        // ═══════════════════════════════════════════════════════════════════
        // UNKNOWN: Things that don't exist or are nonsensical
        // ═══════════════════════════════════════════════════════════════════
        let unknown_exemplars = [
            // Fictional places - full questions
            "What is the GDP of Atlantis?",
            "What is the population of Hogwarts?",
            "What is the capital of Mordor?",
            "What is the weather in Narnia?",
            "How do I get to El Dorado?",
            "What is the currency in Wakanda?",
            "Who rules Westeros?",

            // Fictional places - EXPANDED: synonyms and fragments
            "Atlantis GDP",
            "Atlantis economy",
            "Atlantis economic output",
            "Atlantis statistics",
            "Atlantis data",
            "Tell me about Atlantis GDP",
            "Give me Atlantis economic data",
            "Hogwarts enrollment numbers",
            "Hogwarts student population",
            "Population of Hogwarts",
            "Mordor geography",
            "Narnia weather forecast",
            "El Dorado location",
            "El Dorado coordinates",
            "Wakanda currency exchange rate",
            "Westeros political system",

            // Fictional places - EXPANDED: more fictional entities
            "What is the GDP of Avalon?",
            "What is the population of Rivendell?",
            "What is the capital of Gondor?",
            "Middle Earth statistics",
            "Neverland demographics",
            "Oz economic data",
            "Camelot population",
            "Shangri-La location",

            // Fictional places - EXPANDED: mythical descriptors
            "Tell me the economic output of mythical Atlantis",
            "Statistics about the legendary city",
            "Data on fictional places",
            "Information about imaginary countries",
            "Facts about made-up kingdoms",

            // Nonsensical questions
            "What is the color of happiness?",
            "What does Tuesday smell like?",
            "What is the weight of love?",
            "How loud is purple?",
            "What is the taste of mathematics?",

            // Nonsensical - EXPANDED: more abstract nonsense
            "How much does silence weigh?",
            "What color is the number seven?",
            "How fast is yesterday?",
            "What shape is freedom?",
            "How tall is infinity?",

            // Impossible/contradictory
            "Draw a square circle",
            "Who is the married bachelor?",
            "What is north of the North Pole?",

            // Impossible - EXPANDED
            "Find the largest prime number",
            "Count all the integers",
            "Describe the edge of the universe",
        ];

        for query in unknown_exemplars {
            self.add_exemplar(semantic_space, query, EpistemicStatus::Unknown)?;
        }

        // ═══════════════════════════════════════════════════════════════════
        // UNVERIFIABLE: Future, subjective, hypothetical
        // ═══════════════════════════════════════════════════════════════════
        let unverifiable_exemplars = [
            // Future predictions - full questions
            "What will the stock market do tomorrow?",
            "What are the lottery numbers?",
            "What will happen next year?",
            "When will I die?",
            "Will it rain next week?",
            "Who will win the election?",

            // Future - EXPANDED: fragments and synonyms
            "Tomorrow's stock prices",
            "Tomorrow stock market",
            "Future stock prices",
            "Future market predictions",
            "Upcoming market trends",
            "Next week weather",
            "Next year predictions",
            "Future lottery numbers",
            "Winning lottery numbers",
            "Predict the lottery",
            "Predict tomorrow",
            "Forecast next month",

            // Future - EXPANDED: more temporal markers
            "What happens next",
            "What will be",
            "Future events",
            "Upcoming events",
            "Things that will happen",
            "Tomorrow's news",
            "Next week's headlines",
            "Future technology",
            "Predictions for 2030",

            // Subjective experience
            "What am I thinking right now?",
            "Read my mind",
            "What is my favorite color?",
            "How do I feel?",
            "What do I want for dinner?",

            // Subjective - EXPANDED
            "Tell me my thoughts",
            "Guess what I'm thinking",
            "What do I believe?",
            "My personal preferences",
            "What I should do with my life",
            "What makes me happy",
            "My secret desires",

            // Counterfactuals
            "What if Napoleon had won at Waterloo?",
            "What would have happened if Hitler died young?",
            "What if dinosaurs never went extinct?",
            "What would the world be like without electricity?",

            // Counterfactuals - EXPANDED
            "Alternate history scenarios",
            "If history had been different",
            "Parallel universe outcomes",
            "What could have been",
            "Hypothetical scenarios",
        ];

        for query in unverifiable_exemplars {
            self.add_exemplar(semantic_space, query, EpistemicStatus::Unverifiable)?;
        }

        // ═══════════════════════════════════════════════════════════════════
        // KNOWN: Established facts
        // ═══════════════════════════════════════════════════════════════════
        let known_exemplars = [
            // Geography - full questions
            "What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Japan?",
            "What is the largest country?",
            "What continent is Egypt in?",

            // Geography - EXPANDED: fragments and variations
            "Capital of France",
            "France capital",
            "France capital city",
            "French capital",
            "Paris capital",
            "Germany capital",
            "German capital city",
            "Japan capital",
            "Tokyo capital",
            "Largest country in the world",
            "Biggest country",
            "Egypt continent",
            "Where is Egypt",
            "Location of France",

            // Geography - EXPANDED: more countries
            "Capital of Spain",
            "Capital of Italy",
            "Capital of China",
            "Capital of Brazil",
            "Capital of Australia",
            "Capital of Canada",
            "What country is Paris in",
            "What country is London in",

            // Math - full questions
            "What is 2 + 2?",
            "What is 15 times 3?",
            "What is the square root of 144?",
            "Calculate 100 divided by 5",

            // Math - EXPANDED: fragments and variations
            "2 + 2",
            "2 plus 2",
            "Two plus two",
            "15 times 3",
            "15 * 3",
            "Square root of 144",
            "100 / 5",
            "100 divided by 5",
            "Simple math",
            "Basic arithmetic",
            "Calculate this",
            "Do the math",

            // Science - full questions
            "What is the boiling point of water?",
            "What is the speed of light?",
            "What is H2O?",
            "How many planets are in our solar system?",

            // Science - EXPANDED: fragments and variations
            "Boiling point water",
            "Water boiling temperature",
            "Speed of light",
            "Light speed",
            "H2O formula",
            "Water chemical formula",
            "Number of planets",
            "Planets in solar system",
            "Solar system planets",

            // Science - EXPANDED: more facts
            "Atomic number of carbon",
            "Chemical symbol for gold",
            "Freezing point of water",
            "Earth's circumference",
            "Distance to the moon",
            "How many moons does Jupiter have",

            // History/Culture - full questions
            "Who wrote Hamlet?",
            "Who painted the Mona Lisa?",
            "When did World War 2 end?",
            "Who was the first president of the United States?",

            // History/Culture - EXPANDED: fragments and variations
            "Hamlet author",
            "Shakespeare plays",
            "Mona Lisa painter",
            "Mona Lisa artist",
            "Leonardo da Vinci paintings",
            "World War 2 end date",
            "WW2 ended",
            "First US president",
            "George Washington president",
            "American history facts",

            // General knowledge - EXPANDED
            "Tell me about rust programming",
            "Explain Python language",
            "What is NixOS",
            "Define computer science",
            "Explain how the internet works",
            "What is DNA",
            "How does gravity work",
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

// ═══════════════════════════════════════════════════════════════════════════
// SEMANTIC EPISTEMIC CLASSIFIER (HAM Architecture)
// ═══════════════════════════════════════════════════════════════════════════

/// Exemplar for semantic epistemic classification
#[derive(Debug, Clone)]
pub struct SemanticExemplar {
    /// The query text
    pub query: String,
    /// Dense vector representation (768D from BGE)
    pub dense: DenseVector,
    /// The epistemic category
    pub status: EpistemicStatus,
}

/// Semantic Epistemic Classifier - HAM Architecture Integration
///
/// Uses SemanticEncoder (BGE embeddings) for superior novel phrasing handling.
/// This is the "Perception" layer of the Holographic Associative Memory.
///
/// ## Advantages over N-gram HDC
///
/// - **True semantic understanding**: "French capital" ≈ "capital of France"
/// - **Multilingual potential**: BGE supports 100+ languages
/// - **Typo tolerance**: Semantic meaning survives typos
/// - **Contextual awareness**: Dense embeddings capture nuance
///
/// ## Usage
///
/// ```ignore
/// let encoder = SemanticEncoder::new()?;
/// let classifier = SemanticEpistemicClassifier::new(encoder)?;
/// let (status, confidence) = classifier.classify("What is the GDP of Atlantis?")?;
/// ```
pub struct SemanticEpistemicClassifier {
    /// Semantic encoder (BGE + HDC bridge)
    encoder: SemanticEncoder,

    /// Encoded exemplars
    exemplars: Vec<SemanticExemplar>,

    /// Minimum similarity threshold for confident classification
    confidence_threshold: f32,
}

impl SemanticEpistemicClassifier {
    /// Create a new semantic epistemic classifier
    pub fn new(encoder: SemanticEncoder) -> anyhow::Result<Self> {
        let mut classifier = Self {
            encoder,
            exemplars: Vec::new(),
            confidence_threshold: 0.5, // Higher threshold for semantic similarity
        };

        classifier.train_exemplars()?;

        Ok(classifier)
    }

    /// Train with exemplars (same as HdcEpistemicClassifier but using semantic encoding)
    fn train_exemplars(&mut self) -> anyhow::Result<()> {
        // Unknown exemplars
        let unknown = vec![
            "What is the GDP of Atlantis?",
            "What is the population of Hogwarts?",
            "Atlantis GDP",
            "Atlantis economy",
            "Tell me the economic output of mythical Atlantis",
            "Hogwarts enrollment numbers",
            "What is the color of happiness?",
            "What does Tuesday smell like?",
            "How loud is purple?",
            "Draw a square circle",
        ];

        for query in unknown {
            self.add_exemplar(query, EpistemicStatus::Unknown)?;
        }

        // Unverifiable exemplars
        let unverifiable = vec![
            "What will the stock market do tomorrow?",
            "Tomorrow's stock prices",
            "Future market predictions",
            "What am I thinking right now?",
            "Read my mind",
            "What if Napoleon had won at Waterloo?",
            "Predict the lottery",
            "Next week's headlines",
        ];

        for query in unverifiable {
            self.add_exemplar(query, EpistemicStatus::Unverifiable)?;
        }

        // Known exemplars
        let known = vec![
            "What is the capital of France?",
            "Capital of France",
            "France capital city",
            "French capital",
            "What is 2 + 2?",
            "2 + 2",
            "What is the boiling point of water?",
            "Who wrote Hamlet?",
            "What is the speed of light?",
        ];

        for query in known {
            self.add_exemplar(query, EpistemicStatus::Known)?;
        }

        tracing::info!(
            "🧠 Semantic Epistemic Classifier trained with {} exemplars",
            self.exemplars.len()
        );

        Ok(())
    }

    /// Add an exemplar using semantic encoding
    fn add_exemplar(&mut self, query: &str, status: EpistemicStatus) -> anyhow::Result<()> {
        let dense = self.encoder.encode_dense(query)?;

        self.exemplars.push(SemanticExemplar {
            query: query.to_string(),
            dense,
            status,
        });

        Ok(())
    }

    /// Classify a query using semantic similarity
    ///
    /// Returns the most likely epistemic status and confidence score.
    pub fn classify(&self, query: &str) -> anyhow::Result<(EpistemicStatus, f32)> {
        // Encode the query
        let query_dense = self.encoder.encode_dense(query)?;

        // Compute similarity to all exemplars
        let mut category_scores: HashMap<EpistemicStatus, Vec<f32>> = HashMap::new();
        category_scores.insert(EpistemicStatus::Unknown, Vec::new());
        category_scores.insert(EpistemicStatus::Unverifiable, Vec::new());
        category_scores.insert(EpistemicStatus::Known, Vec::new());

        for exemplar in &self.exemplars {
            let similarity = query_dense.similarity(&exemplar.dense);

            if let Some(scores) = category_scores.get_mut(&exemplar.status) {
                scores.push(similarity);
            }
        }

        // Calculate average score for each category (top-3)
        let mut best_status = EpistemicStatus::Uncertain;
        let mut best_score: f32 = 0.0;

        for (status, scores) in &category_scores {
            if !scores.is_empty() {
                let mut sorted_scores = scores.clone();
                sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

                let top_n = sorted_scores.iter().take(3).copied().collect::<Vec<_>>();
                let avg_score: f32 = top_n.iter().sum::<f32>() / top_n.len() as f32;

                tracing::debug!(
                    "Semantic {:?}: top-3 avg similarity = {:.4}",
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
                "Semantic best score {:.4} below threshold {:.4}, defaulting to Uncertain",
                best_score, self.confidence_threshold
            );
            return Ok((EpistemicStatus::Uncertain, best_score));
        }

        tracing::info!(
            "Semantic classified '{}' as {:?} (confidence: {:.2}%)",
            &query[..query.len().min(40)],
            best_status,
            best_score * 100.0
        );

        Ok((best_status, best_score))
    }

    /// Get the underlying encoder (for HAM integration)
    pub fn encoder(&self) -> &SemanticEncoder {
        &self.encoder
    }

    /// Get mutable encoder (for temporal context)
    pub fn encoder_mut(&mut self) -> &mut SemanticEncoder {
        &mut self.encoder
    }

    /// Get statistics
    pub fn stats(&self) -> SemanticEpistemicStats {
        let mut counts: HashMap<EpistemicStatus, usize> = HashMap::new();

        for exemplar in &self.exemplars {
            *counts.entry(exemplar.status).or_insert(0) += 1;
        }

        SemanticEpistemicStats {
            total_exemplars: self.exemplars.len(),
            unknown_count: *counts.get(&EpistemicStatus::Unknown).unwrap_or(&0),
            unverifiable_count: *counts.get(&EpistemicStatus::Unverifiable).unwrap_or(&0),
            known_count: *counts.get(&EpistemicStatus::Known).unwrap_or(&0),
            encoder_stats: self.encoder.stats().clone(),
        }
    }
}

/// Statistics for the semantic epistemic classifier
#[derive(Debug, Clone)]
pub struct SemanticEpistemicStats {
    pub total_exemplars: usize,
    pub unknown_count: usize,
    pub unverifiable_count: usize,
    pub known_count: usize,
    pub encoder_stats: super::semantic_encoder::EncoderStats,
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

        // The exact exemplar should have high confidence (>70% with expanded exemplars)
        assert_eq!(status, EpistemicStatus::Unknown);
        assert!(confidence > 0.7, "Expected confidence > 0.7, got {}", confidence);
    }

    #[test]
    fn test_classify_known() {
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        let (status, confidence) = classifier
            .classify(&mut semantic_space, "What is 2 + 2?")
            .unwrap();

        assert_eq!(status, EpistemicStatus::Known);
        assert!(confidence > 0.7, "Expected confidence > 0.7, got {}", confidence);
    }

    #[test]
    fn test_classify_unverifiable() {
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        let (status, confidence) = classifier
            .classify(&mut semantic_space, "What will the stock market do tomorrow?")
            .unwrap();

        assert_eq!(status, EpistemicStatus::Unverifiable);
        assert!(confidence > 0.6, "Expected confidence > 0.6, got {}", confidence);
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

    // ═══════════════════════════════════════════════════════════════════
    // SEMANTIC EPISTEMIC CLASSIFIER TESTS
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_semantic_classifier_creation() {
        let encoder = SemanticEncoder::new().unwrap();
        let classifier = SemanticEpistemicClassifier::new(encoder).unwrap();

        let stats = classifier.stats();
        assert!(stats.total_exemplars > 0);
        assert!(stats.unknown_count > 0);
        assert!(stats.unverifiable_count > 0);
        assert!(stats.known_count > 0);

        println!("Semantic classifier stats: {:?}", stats);
    }

    #[test]
    fn test_semantic_classify_unknown() {
        let encoder = SemanticEncoder::new().unwrap();
        let classifier = SemanticEpistemicClassifier::new(encoder).unwrap();

        let (status, confidence) = classifier
            .classify("What is the GDP of Atlantis?")
            .unwrap();

        println!("Atlantis query: {:?} ({:.2}%)", status, confidence * 100.0);
        // Note: With stub embedder, results depend on hash-based similarity
    }

    #[test]
    fn test_semantic_classify_known() {
        let encoder = SemanticEncoder::new().unwrap();
        let classifier = SemanticEpistemicClassifier::new(encoder).unwrap();

        let (status, confidence) = classifier
            .classify("What is the capital of France?")
            .unwrap();

        println!("France capital query: {:?} ({:.2}%)", status, confidence * 100.0);
    }

    #[test]
    fn test_semantic_vs_hdc_comparison() {
        // Create both classifiers
        let mut semantic_space = SemanticSpace::new(512).unwrap();
        let hdc_classifier = HdcEpistemicClassifier::new(&mut semantic_space).unwrap();

        let encoder = SemanticEncoder::new().unwrap();
        let semantic_classifier = SemanticEpistemicClassifier::new(encoder).unwrap();

        let test_queries = [
            "What is the GDP of Atlantis?",
            "Tomorrow's stock prices",
            "Capital of France",
        ];

        println!("\n📊 Comparing HDC vs Semantic Classification:\n");
        println!("{:40} | {:20} | {:20}", "Query", "HDC", "Semantic");
        println!("{}", "-".repeat(85));

        for query in test_queries {
            let (hdc_status, hdc_conf) = hdc_classifier
                .classify(&mut semantic_space, query)
                .unwrap();
            let (sem_status, sem_conf) = semantic_classifier
                .classify(query)
                .unwrap();

            println!(
                "{:40} | {:?} ({:.1}%) | {:?} ({:.1}%)",
                &query[..query.len().min(38)],
                hdc_status, hdc_conf * 100.0,
                sem_status, sem_conf * 100.0
            );
        }
    }
}
