use unicode_segmentation::UnicodeSegmentation;
use rust_stemmers::{Algorithm, Stemmer};

/// Text processor for search indexing
///
/// Handles tokenization, normalization, stop word removal, and stemming.
pub struct TextProcessor {
    stemmer: Stemmer,
    stop_words: Vec<String>,
}

impl TextProcessor {
    pub fn new() -> Self {
        Self {
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: Self::default_stop_words(),
        }
    }

    /// Process text into searchable terms
    ///
    /// Steps:
    /// 1. Normalize (lowercase, remove punctuation)
    /// 2. Tokenize (split into words)
    /// 3. Remove stop words
    /// 4. Stem words
    pub fn process_text(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize(text);
        let tokens = self.tokenize(&normalized);
        let filtered = self.remove_stop_words(&tokens);
        self.stem_words(&filtered)
    }

    /// Normalize text to lowercase and remove punctuation
    pub fn normalize(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
            .collect()
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.unicode_words()
            .map(|s| s.to_string())
            .collect()
    }

    /// Remove common stop words
    fn remove_stop_words(&self, words: &[String]) -> Vec<String> {
        words.iter()
            .filter(|word| !self.stop_words.contains(word))
            .cloned()
            .collect()
    }

    /// Stem words to their root form
    fn stem_words(&self, words: &[String]) -> Vec<String> {
        words.iter()
            .map(|word| self.stemmer.stem(word).to_string())
            .collect()
    }

    /// Default English stop words
    fn default_stop_words() -> Vec<String> {
        vec![
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "will", "with",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let processor = TextProcessor::new();
        assert_eq!(
            processor.normalize("Hello, World!"),
            "hello  world "
        );
    }

    #[test]
    fn test_tokenize() {
        let processor = TextProcessor::new();
        let tokens = processor.tokenize("hello world test");
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_remove_stop_words() {
        let processor = TextProcessor::new();
        let words = vec!["the".to_string(), "quick".to_string(), "brown".to_string()];
        let filtered = processor.remove_stop_words(&words);
        assert_eq!(filtered, vec!["quick", "brown"]);
    }

    #[test]
    fn test_stem_words() {
        let processor = TextProcessor::new();
        let words = vec!["running".to_string(), "runs".to_string(), "ran".to_string()];
        let stemmed = processor.stem_words(&words);
        // All should stem to "run"
        assert_eq!(stemmed.len(), 3);
        assert!(stemmed.iter().all(|w| w.starts_with("run")));
    }

    #[test]
    fn test_process_text() {
        let processor = TextProcessor::new();
        let processed = processor.process_text("The quick brown fox is running!");
        // Should remove "the", "is" and stem "running" to "run"
        assert!(!processed.contains(&"the".to_string()));
        assert!(!processed.contains(&"is".to_string()));
        assert!(processed.contains(&"quick".to_string()));
        assert!(processed.contains(&"brown".to_string()));
        assert!(processed.contains(&"fox".to_string()));
    }
}
