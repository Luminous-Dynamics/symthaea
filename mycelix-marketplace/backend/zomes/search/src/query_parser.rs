use regex::Regex;
use crate::text_processor::TextProcessor;

/// Parsed query with extracted terms and operators
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// Standard search terms
    pub terms: Vec<String>,

    /// Terms that must be present (AND operator)
    pub required_terms: Vec<String>,

    /// Terms that must not be present (NOT operator)
    pub excluded_terms: Vec<String>,

    /// Exact phrase matches
    pub phrase_terms: Vec<String>,
}

/// Query parser for natural language search
///
/// Supports:
/// - Basic terms: "laptop macbook"
/// - AND operator: "laptop AND macbook"
/// - OR operator: "laptop OR desktop" (default behavior)
/// - NOT operator: "-refurbished"
/// - Exact phrases: "\"macbook pro\""
pub struct QueryParser {
    text_processor: TextProcessor,
}

impl QueryParser {
    pub fn new() -> Self {
        Self {
            text_processor: TextProcessor::new(),
        }
    }

    /// Parse a search query string
    pub fn parse(&self, query: &str) -> ParsedQuery {
        let mut terms = Vec::new();
        let mut required_terms = Vec::new();
        let mut excluded_terms = Vec::new();
        let mut phrase_terms = Vec::new();

        // Extract exact phrases first
        let phrase_re = Regex::new(r#""([^"]+)""#).unwrap();
        let mut remaining_query = query.to_string();

        for cap in phrase_re.captures_iter(query) {
            if let Some(phrase) = cap.get(1) {
                phrase_terms.push(phrase.as_str().to_string());
                // Remove from remaining query
                remaining_query = remaining_query.replace(&format!("\"{}\"", phrase.as_str()), "");
            }
        }

        // Process remaining query
        let words: Vec<&str> = remaining_query.split_whitespace().collect();
        let mut i = 0;

        while i < words.len() {
            let word = words[i];

            // Check for excluded terms (starts with -)
            if word.starts_with('-') {
                let term = self.text_processor.normalize(&word[1..]);
                if !term.is_empty() {
                    excluded_terms.push(term);
                }
                i += 1;
                continue;
            }

            // Check for AND operator
            if word.to_uppercase() == "AND" && i + 1 < words.len() {
                let next_word = words[i + 1];
                let term = self.text_processor.normalize(next_word);
                if !term.is_empty() {
                    required_terms.push(term);
                }
                i += 2;
                continue;
            }

            // Check for OR operator (default behavior, just skip)
            if word.to_uppercase() == "OR" {
                i += 1;
                continue;
            }

            // Regular term
            let term = self.text_processor.normalize(word);
            if !term.is_empty() {
                // Process with stemming
                let processed = self.text_processor.process_text(&term);
                terms.extend(processed);
            }

            i += 1;
        }

        ParsedQuery {
            terms,
            required_terms,
            excluded_terms,
            phrase_terms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_query() {
        let parser = QueryParser::new();
        let parsed = parser.parse("laptop macbook");

        assert!(parsed.terms.len() >= 2);
        assert!(parsed.required_terms.is_empty());
        assert!(parsed.excluded_terms.is_empty());
        assert!(parsed.phrase_terms.is_empty());
    }

    #[test]
    fn test_and_operator() {
        let parser = QueryParser::new();
        let parsed = parser.parse("laptop AND macbook");

        assert!(!parsed.required_terms.is_empty());
    }

    #[test]
    fn test_excluded_terms() {
        let parser = QueryParser::new();
        let parsed = parser.parse("laptop -refurbished");

        assert!(!parsed.excluded_terms.is_empty());
        assert!(parsed.excluded_terms.iter().any(|t| t.contains("refurbish")));
    }

    #[test]
    fn test_exact_phrases() {
        let parser = QueryParser::new();
        let parsed = parser.parse("\"macbook pro\" laptop");

        assert!(!parsed.phrase_terms.is_empty());
        assert_eq!(parsed.phrase_terms[0], "macbook pro");
    }

    #[test]
    fn test_complex_query() {
        let parser = QueryParser::new();
        let parsed = parser.parse("\"macbook pro\" laptop AND 16inch -refurbished");

        assert!(!parsed.terms.is_empty());
        assert!(!parsed.required_terms.is_empty());
        assert!(!parsed.excluded_terms.is_empty());
        assert!(!parsed.phrase_terms.is_empty());
    }
}
