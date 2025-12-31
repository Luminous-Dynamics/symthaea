//! Math Language Adapter - Natural Language Math Processing
//!
//! Connects natural language math queries to the HDC arithmetic engine.
//! Routes questions like "What is 7 times 8?" to true mathematical cognition.
//!
//! ## Features
//!
//! - **Math Query Detection**: Identifies mathematical questions
//! - **Operation Extraction**: Parses numbers and operations
//! - **Proof Generation**: Returns results with mathematical proofs
//! - **Φ Tracking**: Measures consciousness of mathematical understanding
//!
//! ## Example Queries
//!
//! - "What is 7 times 8?"
//! - "Calculate 123 + 456"
//! - "What's the GCD of 24 and 36?"
//! - "Is 17 prime?"
//! - "Does 3 divide 12?"

use crate::hdc::arithmetic_engine::{
    HybridArithmeticEngine, HybridResult, MathReasoningBridge,
    MathAssertion, MathRelation, AbstractProof,
};
use crate::hdc::binary_hv::HV16;
use std::collections::HashMap;
use regex::Regex;

/// Result of processing a math query
#[derive(Debug, Clone)]
pub struct MathQueryResult {
    /// The original query
    pub query: String,
    /// The answer as a string
    pub answer: String,
    /// Numerical result if applicable
    pub numeric_result: Option<u64>,
    /// Explanation of how we got the answer
    pub explanation: Vec<String>,
    /// Abstract proof if available
    pub proof: Option<AbstractProof>,
    /// Total Φ from the computation
    pub phi: f64,
    /// Whether this was handled by the math engine
    pub handled: bool,
    /// Confidence in the result (1.0 for mathematical certainty)
    pub confidence: f64,
    /// Query type detected
    pub query_type: MathQueryType,
}

/// Type of math query
#[derive(Debug, Clone, PartialEq)]
pub enum MathQueryType {
    /// Basic arithmetic: add, subtract, multiply, divide
    Arithmetic,
    /// Comparison: equals, less than, greater than
    Comparison,
    /// Number theory: prime, gcd, divisibility
    NumberTheory,
    /// Theorem proving: commutativity, associativity
    TheoremProving,
    /// General math question
    GeneralMath,
    /// Not a math query
    NotMath,
}

/// Math language adapter - bridges natural language to arithmetic engine
pub struct MathLanguageAdapter {
    /// The reasoning bridge for computations
    bridge: MathReasoningBridge,
    /// Patterns for detecting math queries
    patterns: MathPatterns,
    /// Statistics
    stats: MathAdapterStats,
}

/// Patterns for detecting and parsing math queries
struct MathPatterns {
    // Basic arithmetic patterns
    add_pattern: Regex,
    subtract_pattern: Regex,
    multiply_pattern: Regex,
    divide_pattern: Regex,
    modulo_pattern: Regex,
    power_pattern: Regex,

    // Number theory patterns
    gcd_pattern: Regex,
    prime_pattern: Regex,
    divides_pattern: Regex,
    coprime_pattern: Regex,
    factorial_pattern: Regex,

    // General patterns
    number_pattern: Regex,
    calculate_pattern: Regex,
}

/// Statistics for the adapter
#[derive(Debug, Clone, Default)]
pub struct MathAdapterStats {
    pub queries_processed: usize,
    pub math_queries: usize,
    pub non_math_queries: usize,
    pub total_phi: f64,
    pub avg_phi: f64,
}

impl MathPatterns {
    fn new() -> Self {
        Self {
            // Arithmetic patterns
            add_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute)?\s*(\d+)\s*(?:\+|plus|and|added to)\s*(\d+)"
            ).unwrap(),
            subtract_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute)?\s*(\d+)\s*(?:-|minus|subtract(?:ed)?)\s*(\d+)"
            ).unwrap(),
            multiply_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute)?\s*(\d+)\s*(?:\*|×|times|multiplied by)\s*(\d+)"
            ).unwrap(),
            divide_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute)?\s*(\d+)\s*(?:/|÷|divided by)\s*(\d+)"
            ).unwrap(),
            modulo_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute)?\s*(\d+)\s*(?:mod(?:ulo)?|%|remainder)\s*(\d+)"
            ).unwrap(),
            power_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute)?\s*(\d+)\s*(?:\^|to the power(?: of)?|raised to)\s*(\d+)"
            ).unwrap(),

            // Number theory patterns
            gcd_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)(?: the)?|find|calculate|compute)?\s*(?:gcd|greatest common divisor|gcf|hcf)\s*(?:of)?\s*(\d+)\s*(?:and|,)?\s*(\d+)"
            ).unwrap(),
            prime_pattern: Regex::new(
                r"(?i)is\s*(\d+)\s*(?:a\s*)?prime(?:\s*number)?"
            ).unwrap(),
            divides_pattern: Regex::new(
                r"(?i)does\s*(\d+)\s*divide\s*(\d+)|is\s*(\d+)\s*divisible\s*by\s*(\d+)"
            ).unwrap(),
            coprime_pattern: Regex::new(
                r"(?i)(?:are|is)\s*(\d+)\s*(?:and)?\s*(\d+)\s*coprime|(?:are|is)\s*(\d+)\s*(?:and)?\s*(\d+)\s*relatively prime"
            ).unwrap(),
            factorial_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute)?\s*(\d+)(?:\s*factorial|!)"
            ).unwrap(),

            // General patterns
            number_pattern: Regex::new(r"\d+").unwrap(),
            calculate_pattern: Regex::new(
                r"(?i)(?:what(?:'s| is)|calculate|compute|find|evaluate)"
            ).unwrap(),
        }
    }
}

impl MathLanguageAdapter {
    /// Create a new math language adapter
    pub fn new() -> Self {
        Self {
            bridge: MathReasoningBridge::new(),
            patterns: MathPatterns::new(),
            stats: MathAdapterStats::default(),
        }
    }

    /// Process a natural language query
    pub fn process(&mut self, query: &str) -> MathQueryResult {
        self.stats.queries_processed += 1;

        let query_type = self.detect_query_type(query);

        if query_type == MathQueryType::NotMath {
            self.stats.non_math_queries += 1;
            return MathQueryResult {
                query: query.to_string(),
                answer: String::new(),
                numeric_result: None,
                explanation: vec![],
                proof: None,
                phi: 0.0,
                handled: false,
                confidence: 0.0,
                query_type,
            };
        }

        self.stats.math_queries += 1;

        match query_type {
            MathQueryType::Arithmetic => self.process_arithmetic(query),
            MathQueryType::NumberTheory => self.process_number_theory(query),
            MathQueryType::TheoremProving => self.process_theorem(query),
            MathQueryType::Comparison => self.process_comparison(query),
            MathQueryType::GeneralMath => self.process_general_math(query),
            MathQueryType::NotMath => unreachable!(),
        }
    }

    /// Detect the type of math query
    fn detect_query_type(&self, query: &str) -> MathQueryType {
        let query_lower = query.to_lowercase();

        // Check for number theory queries first (more specific)
        if self.patterns.gcd_pattern.is_match(query) ||
           self.patterns.prime_pattern.is_match(query) ||
           self.patterns.divides_pattern.is_match(query) ||
           self.patterns.coprime_pattern.is_match(query) ||
           self.patterns.factorial_pattern.is_match(query) {
            return MathQueryType::NumberTheory;
        }

        // Check for basic arithmetic
        if self.patterns.add_pattern.is_match(query) ||
           self.patterns.subtract_pattern.is_match(query) ||
           self.patterns.multiply_pattern.is_match(query) ||
           self.patterns.divide_pattern.is_match(query) ||
           self.patterns.modulo_pattern.is_match(query) ||
           self.patterns.power_pattern.is_match(query) {
            return MathQueryType::Arithmetic;
        }

        // Check for theorem proving
        if query_lower.contains("commutative") ||
           query_lower.contains("associative") ||
           query_lower.contains("distributive") ||
           query_lower.contains("prove") ||
           query_lower.contains("theorem") {
            return MathQueryType::TheoremProving;
        }

        // Check for comparison
        if query_lower.contains("equal") ||
           query_lower.contains("same") ||
           query_lower.contains("greater") ||
           query_lower.contains("less") ||
           query_lower.contains("compare") {
            return MathQueryType::Comparison;
        }

        // Check if it's general math (has numbers and math words)
        if self.patterns.number_pattern.is_match(query) &&
           self.patterns.calculate_pattern.is_match(query) {
            return MathQueryType::GeneralMath;
        }

        MathQueryType::NotMath
    }

    /// Process arithmetic queries
    fn process_arithmetic(&mut self, query: &str) -> MathQueryResult {
        // Try each arithmetic pattern
        if let Some(caps) = self.patterns.add_pattern.captures(query) {
            let a: u64 = caps[1].parse().unwrap_or(0);
            let b: u64 = caps[2].parse().unwrap_or(0);
            let assertion = self.bridge.assert_equality(a, b, "+");
            return self.build_result(query, &assertion, MathQueryType::Arithmetic,
                format!("{} + {} = {}", a, b, assertion.object));
        }

        if let Some(caps) = self.patterns.subtract_pattern.captures(query) {
            let a: u64 = caps[1].parse().unwrap_or(0);
            let b: u64 = caps[2].parse().unwrap_or(0);
            let assertion = self.bridge.assert_equality(a, b, "-");
            return self.build_result(query, &assertion, MathQueryType::Arithmetic,
                format!("{} - {} = {}", a, b, assertion.object));
        }

        if let Some(caps) = self.patterns.multiply_pattern.captures(query) {
            let a: u64 = caps[1].parse().unwrap_or(0);
            let b: u64 = caps[2].parse().unwrap_or(0);
            let assertion = self.bridge.assert_equality(a, b, "*");
            return self.build_result(query, &assertion, MathQueryType::Arithmetic,
                format!("{} × {} = {}", a, b, assertion.object));
        }

        if let Some(caps) = self.patterns.divide_pattern.captures(query) {
            let a: u64 = caps[1].parse().unwrap_or(0);
            let b: u64 = caps[2].parse().unwrap_or(0);
            let assertion = self.bridge.assert_equality(a, b, "/");
            return self.build_result(query, &assertion, MathQueryType::Arithmetic,
                format!("{} ÷ {} = {}", a, b, assertion.object));
        }

        if let Some(caps) = self.patterns.modulo_pattern.captures(query) {
            let a: u64 = caps[1].parse().unwrap_or(0);
            let b: u64 = caps[2].parse().unwrap_or(0);
            let assertion = self.bridge.assert_equality(a, b, "%");
            return self.build_result(query, &assertion, MathQueryType::Arithmetic,
                format!("{} mod {} = {}", a, b, assertion.object));
        }

        if let Some(caps) = self.patterns.power_pattern.captures(query) {
            let a: u64 = caps[1].parse().unwrap_or(0);
            let b: u64 = caps[2].parse().unwrap_or(0);
            let assertion = self.bridge.assert_equality(a, b, "^");
            return self.build_result(query, &assertion, MathQueryType::Arithmetic,
                format!("{}^{} = {}", a, b, assertion.object));
        }

        self.not_handled(query, MathQueryType::Arithmetic)
    }

    /// Process number theory queries
    fn process_number_theory(&mut self, query: &str) -> MathQueryResult {
        // GCD
        if let Some(caps) = self.patterns.gcd_pattern.captures(query) {
            let a: u64 = caps[1].parse().unwrap_or(0);
            let b: u64 = caps[2].parse().unwrap_or(0);
            let assertion = self.bridge.assert_equality(a, b, "gcd");
            return self.build_result(query, &assertion, MathQueryType::NumberTheory,
                format!("gcd({}, {}) = {}", a, b, assertion.object));
        }

        // Factorial
        if let Some(caps) = self.patterns.factorial_pattern.captures(query) {
            let n: u64 = caps[1].parse().unwrap_or(0);
            let result = self.bridge.engine().factorial(n);
            let assertion = MathAssertion {
                subject: format!("{}!", n),
                relation: MathRelation::Equals,
                object: result.value.to_string(),
                confidence: 1.0,
                phi: result.phi,
                proof_source: result.abstract_proof.clone(),
            };
            return self.build_result(query, &assertion, MathQueryType::NumberTheory,
                format!("{}! = {}", n, result.value));
        }

        // Prime
        if let Some(caps) = self.patterns.prime_pattern.captures(query) {
            let n: u64 = caps[1].parse().unwrap_or(0);
            let assertion = self.bridge.assert_prime(n);
            let answer = if assertion.object == "Prime" {
                format!("Yes, {} is prime", n)
            } else {
                format!("No, {} is composite (not prime)", n)
            };
            return self.build_result(query, &assertion, MathQueryType::NumberTheory, answer);
        }

        // Divisibility
        if let Some(caps) = self.patterns.divides_pattern.captures(query) {
            // Handle both "does a divide b" and "is b divisible by a"
            let (a, b) = if caps.get(1).is_some() {
                (caps[1].parse().unwrap_or(0), caps[2].parse().unwrap_or(0))
            } else {
                (caps[4].parse().unwrap_or(0), caps[3].parse().unwrap_or(0))
            };
            let assertion = self.bridge.assert_divides(a, b);
            let answer = if assertion.confidence > 0.5 {
                format!("Yes, {} divides {} evenly", a, b)
            } else {
                format!("No, {} does not divide {} evenly", a, b)
            };
            return self.build_result(query, &assertion, MathQueryType::NumberTheory, answer);
        }

        // Coprime
        if let Some(caps) = self.patterns.coprime_pattern.captures(query) {
            let (a, b) = if caps.get(1).is_some() {
                (caps[1].parse().unwrap_or(0), caps[2].parse().unwrap_or(0))
            } else {
                (caps[3].parse().unwrap_or(0), caps[4].parse().unwrap_or(0))
            };
            let assertion = self.bridge.assert_coprime(a, b);
            let answer = if assertion.confidence > 0.5 {
                format!("Yes, {} and {} are coprime (gcd = 1)", a, b)
            } else {
                format!("No, {} and {} are not coprime (gcd > 1)", a, b)
            };
            return self.build_result(query, &assertion, MathQueryType::NumberTheory, answer);
        }

        self.not_handled(query, MathQueryType::NumberTheory)
    }

    /// Process theorem proving queries
    fn process_theorem(&mut self, query: &str) -> MathQueryResult {
        let query_lower = query.to_lowercase();

        // Extract numbers for theorem proving
        let numbers: Vec<u64> = self.patterns.number_pattern
            .find_iter(query)
            .filter_map(|m| m.as_str().parse().ok())
            .collect();

        if query_lower.contains("commutative") && query_lower.contains("add") {
            if numbers.len() >= 2 {
                if let Some(assertion) = self.bridge.prove_theorem("commutativity_add", &numbers) {
                    return self.build_result(query, &assertion, MathQueryType::TheoremProving,
                        format!("Proven: {} + {} = {} + {} (commutative)",
                            numbers[0], numbers[1], numbers[1], numbers[0]));
                }
            }
        }

        if query_lower.contains("commutative") && (query_lower.contains("mul") || query_lower.contains("times")) {
            if numbers.len() >= 2 {
                if let Some(assertion) = self.bridge.prove_theorem("commutativity_mul", &numbers) {
                    return self.build_result(query, &assertion, MathQueryType::TheoremProving,
                        format!("Proven: {} × {} = {} × {} (commutative)",
                            numbers[0], numbers[1], numbers[1], numbers[0]));
                }
            }
        }

        if query_lower.contains("associative") {
            if numbers.len() >= 3 {
                if let Some(assertion) = self.bridge.prove_theorem("associativity", &numbers) {
                    return self.build_result(query, &assertion, MathQueryType::TheoremProving,
                        format!("Proven: ({} + {}) + {} = {} + ({} + {}) (associative)",
                            numbers[0], numbers[1], numbers[2],
                            numbers[0], numbers[1], numbers[2]));
                }
            }
        }

        if query_lower.contains("distributive") {
            if numbers.len() >= 3 {
                if let Some(assertion) = self.bridge.prove_theorem("distributive", &numbers) {
                    return self.build_result(query, &assertion, MathQueryType::TheoremProving,
                        format!("Proven: {} × ({} + {}) = ({} × {}) + ({} × {}) (distributive)",
                            numbers[0], numbers[1], numbers[2],
                            numbers[0], numbers[1], numbers[0], numbers[2]));
                }
            }
        }

        self.not_handled(query, MathQueryType::TheoremProving)
    }

    /// Process comparison queries
    fn process_comparison(&mut self, query: &str) -> MathQueryResult {
        // Extract expressions and compare them
        let numbers: Vec<u64> = self.patterns.number_pattern
            .find_iter(query)
            .filter_map(|m| m.as_str().parse().ok())
            .collect();

        if numbers.len() >= 2 {
            let a = numbers[0];
            let b = numbers[1];

            let answer = if a == b {
                format!("{} equals {}", a, b)
            } else if a < b {
                format!("{} is less than {}", a, b)
            } else {
                format!("{} is greater than {}", a, b)
            };

            return MathQueryResult {
                query: query.to_string(),
                answer: answer.clone(),
                numeric_result: Some(if a == b { 1 } else { 0 }),
                explanation: vec![answer],
                proof: None,
                phi: 0.1,
                handled: true,
                confidence: 1.0,
                query_type: MathQueryType::Comparison,
            };
        }

        self.not_handled(query, MathQueryType::Comparison)
    }

    /// Process general math queries
    fn process_general_math(&mut self, query: &str) -> MathQueryResult {
        // Try to extract and compute any arithmetic expression
        let numbers: Vec<u64> = self.patterns.number_pattern
            .find_iter(query)
            .filter_map(|m| m.as_str().parse().ok())
            .collect();

        if numbers.len() >= 2 {
            // Default to addition if we can't determine operation
            let assertion = self.bridge.assert_equality(numbers[0], numbers[1], "+");
            return self.build_result(query, &assertion, MathQueryType::GeneralMath,
                format!("{} + {} = {}", numbers[0], numbers[1], assertion.object));
        }

        self.not_handled(query, MathQueryType::GeneralMath)
    }

    /// Build a result from an assertion
    fn build_result(&mut self, query: &str, assertion: &MathAssertion,
                    query_type: MathQueryType, answer: String) -> MathQueryResult {
        self.stats.total_phi += assertion.phi;
        self.stats.avg_phi = self.stats.total_phi / self.stats.math_queries as f64;

        let numeric_result = assertion.object.parse().ok();

        let mut explanation = vec![answer.clone()];
        if assertion.phi > 0.0 {
            explanation.push(format!("Mathematical understanding Φ = {:.4}", assertion.phi));
        }
        if assertion.proof_source.is_some() {
            explanation.push("Proof available".to_string());
        }

        MathQueryResult {
            query: query.to_string(),
            answer,
            numeric_result,
            explanation,
            proof: assertion.proof_source.clone(),
            phi: assertion.phi,
            handled: true,
            confidence: assertion.confidence,
            query_type,
        }
    }

    /// Return a not-handled result
    fn not_handled(&self, query: &str, query_type: MathQueryType) -> MathQueryResult {
        MathQueryResult {
            query: query.to_string(),
            answer: "Could not process this math query".to_string(),
            numeric_result: None,
            explanation: vec!["Query type detected but could not extract parameters".to_string()],
            proof: None,
            phi: 0.0,
            handled: false,
            confidence: 0.0,
            query_type,
        }
    }

    /// Check if a query is mathematical
    pub fn is_math_query(&self, query: &str) -> bool {
        self.detect_query_type(query) != MathQueryType::NotMath
    }

    /// Get adapter statistics
    pub fn stats(&self) -> &MathAdapterStats {
        &self.stats
    }

    /// Get total Φ from all computations
    pub fn total_phi(&self) -> f64 {
        self.bridge.total_phi()
    }

    /// Access the underlying reasoning bridge
    pub fn bridge(&mut self) -> &mut MathReasoningBridge {
        &mut self.bridge
    }
}

impl Default for MathLanguageAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = MathLanguageAdapter::new();
        assert_eq!(adapter.stats().queries_processed, 0);
    }

    #[test]
    fn test_addition_query() {
        let mut adapter = MathLanguageAdapter::new();

        let result = adapter.process("What is 7 plus 8?");
        assert!(result.handled);
        assert_eq!(result.numeric_result, Some(15));
        assert!(result.phi > 0.0);
    }

    #[test]
    fn test_multiplication_query() {
        let mut adapter = MathLanguageAdapter::new();

        let result = adapter.process("calculate 6 times 7");
        assert!(result.handled);
        assert_eq!(result.numeric_result, Some(42));
    }

    #[test]
    fn test_gcd_query() {
        let mut adapter = MathLanguageAdapter::new();

        let result = adapter.process("What is the GCD of 24 and 36?");
        assert!(result.handled);
        assert_eq!(result.numeric_result, Some(12));
        assert_eq!(result.query_type, MathQueryType::NumberTheory);
    }

    #[test]
    fn test_prime_query() {
        let mut adapter = MathLanguageAdapter::new();

        let result = adapter.process("Is 17 prime?");
        assert!(result.handled);
        assert!(result.answer.contains("Yes"));

        let result = adapter.process("Is 15 prime?");
        assert!(result.handled);
        assert!(result.answer.contains("No"));
    }

    #[test]
    fn test_divisibility_query() {
        let mut adapter = MathLanguageAdapter::new();

        let result = adapter.process("Does 3 divide 12?");
        assert!(result.handled);
        assert!(result.answer.contains("Yes"));

        let result = adapter.process("Is 12 divisible by 5?");
        assert!(result.handled);
        assert!(result.answer.contains("No"));
    }

    #[test]
    fn test_factorial_query() {
        let mut adapter = MathLanguageAdapter::new();

        let result = adapter.process("Calculate 5 factorial");
        assert!(result.handled);
        assert_eq!(result.numeric_result, Some(120));
    }

    #[test]
    fn test_non_math_query() {
        let mut adapter = MathLanguageAdapter::new();

        let result = adapter.process("What is the weather today?");
        assert!(!result.handled);
        assert_eq!(result.query_type, MathQueryType::NotMath);
    }

    #[test]
    fn test_stats_accumulation() {
        let mut adapter = MathLanguageAdapter::new();

        adapter.process("5 + 3");
        adapter.process("7 times 8");
        adapter.process("Hello world");

        let stats = adapter.stats();
        assert_eq!(stats.queries_processed, 3);
        assert_eq!(stats.math_queries, 2);
        assert_eq!(stats.non_math_queries, 1);
    }

    #[test]
    fn test_is_math_query() {
        let adapter = MathLanguageAdapter::new();

        assert!(adapter.is_math_query("What is 5 + 3?"));
        assert!(adapter.is_math_query("Calculate 10 times 20"));
        assert!(adapter.is_math_query("Is 17 prime?"));
        assert!(!adapter.is_math_query("Hello world"));
    }
}
