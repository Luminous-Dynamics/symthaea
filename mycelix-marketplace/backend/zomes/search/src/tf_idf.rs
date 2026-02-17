use std::collections::HashMap;
use crate::SearchIndexEntry;

/// TF-IDF (Term Frequency-Inverse Document Frequency) calculator
///
/// Calculates relevance scores for documents based on term frequency
/// and inverse document frequency.
///
/// Formula:
/// TF-IDF(term, doc) = TF(term, doc) * IDF(term)
///
/// where:
/// - TF(term, doc) = (frequency of term in doc) / (total terms in doc)
/// - IDF(term) = log(total documents / documents containing term)
pub struct TfIdfCalculator {
    /// Total number of documents in corpus
    total_docs: usize,

    /// Document frequency: how many documents contain each term
    document_frequency: HashMap<String, usize>,

    /// IDF scores for each term
    idf_scores: HashMap<String, f64>,
}

impl TfIdfCalculator {
    /// Create a new TF-IDF calculator from a corpus of documents
    pub fn new(corpus: &[SearchIndexEntry]) -> Self {
        let total_docs = corpus.len();

        // Calculate document frequency for each term
        let mut document_frequency: HashMap<String, usize> = HashMap::new();

        for doc in corpus {
            // Get unique terms in this document
            let unique_terms: std::collections::HashSet<String> =
                doc.processed_terms.iter().cloned().collect();

            for term in unique_terms {
                *document_frequency.entry(term).or_insert(0) += 1;
            }
        }

        // Calculate IDF scores
        let mut idf_scores = HashMap::new();
        for (term, df) in &document_frequency {
            let idf = ((total_docs as f64) / (*df as f64)).ln();
            idf_scores.insert(term.clone(), idf);
        }

        Self {
            total_docs,
            document_frequency,
            idf_scores,
        }
    }

    /// Calculate TF-IDF score for a document given query terms
    ///
    /// Returns a relevance score between 0.0 and 1.0 (normalized)
    pub fn calculate_score(&self, doc: &SearchIndexEntry, query_terms: &[String]) -> f64 {
        let mut total_score = 0.0;
        let doc_length = doc.processed_terms.len() as f64;

        if doc_length == 0.0 {
            return 0.0;
        }

        for query_term in query_terms {
            // Get term frequency in document
            let tf = *doc.term_frequencies.get(query_term).unwrap_or(&0) as f64 / doc_length;

            // Get IDF score
            let idf = *self.idf_scores.get(query_term).unwrap_or(&0.0);

            // TF-IDF = TF * IDF
            let tfidf = tf * idf;
            total_score += tfidf;
        }

        // Normalize score to [0, 1] range
        // Use query length as normalization factor
        let max_possible_score = query_terms.len() as f64 * 5.0; // Heuristic max
        let normalized_score = (total_score / max_possible_score).min(1.0);

        normalized_score
    }

    /// Get IDF score for a term
    pub fn get_idf(&self, term: &str) -> f64 {
        *self.idf_scores.get(term).unwrap_or(&0.0)
    }

    /// Get document frequency for a term
    pub fn get_document_frequency(&self, term: &str) -> usize {
        *self.document_frequency.get(term).unwrap_or(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EntityType, SearchIndexEntry};
    use hdk::prelude::*;

    fn create_test_entry(terms: Vec<String>) -> SearchIndexEntry {
        let term_frequencies = terms.iter().fold(HashMap::new(), |mut acc, term| {
            *acc.entry(term.clone()).or_insert(0) += 1;
            acc
        });

        SearchIndexEntry {
            index_id: "test".to_string(),
            entity_type: EntityType::Listing,
            entity_hash: ActionHash::from_raw_39(vec![0; 39]).unwrap(),
            title: "Test".to_string(),
            description: "Test".to_string(),
            tags: vec![],
            category: "Test".to_string(),
            creator: AgentPubKey::from_raw_39(vec![0; 39]).unwrap(),
            creator_matl_score: 0.5,
            created_at: 0,
            updated_at: 0,
            view_count: 0,
            engagement_score: 0.0,
            price_cents: None,
            location: None,
            status: "Active".to_string(),
            processed_terms: terms,
            term_frequencies,
        }
    }

    #[test]
    fn test_idf_calculation() {
        let corpus = vec![
            create_test_entry(vec!["laptop".to_string(), "macbook".to_string()]),
            create_test_entry(vec!["laptop".to_string(), "dell".to_string()]),
            create_test_entry(vec!["phone".to_string(), "iphone".to_string()]),
        ];

        let calculator = TfIdfCalculator::new(&corpus);

        // "laptop" appears in 2/3 documents
        let laptop_idf = calculator.get_idf("laptop");
        assert!(laptop_idf > 0.0);

        // "macbook" appears in 1/3 documents (higher IDF)
        let macbook_idf = calculator.get_idf("macbook");
        assert!(macbook_idf > laptop_idf);

        // "nonexistent" doesn't appear (IDF = 0)
        let nonexistent_idf = calculator.get_idf("nonexistent");
        assert_eq!(nonexistent_idf, 0.0);
    }

    #[test]
    fn test_tfidf_scoring() {
        let corpus = vec![
            create_test_entry(vec!["laptop".to_string(), "laptop".to_string(), "macbook".to_string()]),
            create_test_entry(vec!["laptop".to_string(), "dell".to_string()]),
        ];

        let calculator = TfIdfCalculator::new(&corpus);

        // Query for "laptop macbook"
        let query_terms = vec!["laptop".to_string(), "macbook".to_string()];

        // First document should score higher (has both terms, "laptop" appears twice)
        let score1 = calculator.calculate_score(&corpus[0], &query_terms);
        let score2 = calculator.calculate_score(&corpus[1], &query_terms);

        assert!(score1 > score2);
        assert!(score1 > 0.0);
        assert!(score1 <= 1.0);
    }
}
