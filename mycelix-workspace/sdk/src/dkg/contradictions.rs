//! Contradiction Detection for the DKG Truth Engine
//!
//! Detects when multiple triples make conflicting claims about the same
//! subject-predicate pair and adjusts confidence accordingly.
//!
//! # Contradiction Types
//!
//! 1. **Direct contradiction**: Same subject+predicate, incompatible values
//!    - "sky" + "color" = "blue" vs "sky" + "color" = "green"
//!
//! 2. **Logical contradiction**: Values that cannot both be true
//!    - "temperature" = "hot" vs "temperature" = "cold"
//!
//! 3. **Temporal contradiction**: Claims that conflict at same time
//!    - "location" = "Paris" at T vs "location" = "Tokyo" at T
//!
//! # Confidence Impact
//!
//! When contradictions are detected, both claims have reduced confidence
//! proportional to the opposing claim's attestation strength.

use super::{StoredTriple, TripleValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of contradiction analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContradictionAnalysis {
    /// The triple being analyzed
    pub triple_hash: String,
    /// Detected contradictions
    pub contradictions: Vec<Contradiction>,
    /// Overall contradiction weight (0.0 = no contradictions, 1.0 = fully contradicted)
    pub contradiction_weight: f64,
    /// Suggested confidence penalty
    pub confidence_penalty: f64,
}

/// A detected contradiction between two triples
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contradiction {
    /// Hash of the contradicting triple
    pub contradicting_hash: String,
    /// Type of contradiction
    pub contradiction_type: ContradictionType,
    /// Strength of the contradiction (0.0-1.0)
    pub strength: f64,
    /// The conflicting value
    pub conflicting_value: String,
    /// Attestation weight of the contradicting claim
    pub opposing_weight: f64,
}

/// Types of contradictions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContradictionType {
    /// Same subject+predicate with different object values
    DirectValue,
    /// Logically incompatible values (e.g., "hot" vs "cold")
    LogicalIncompatibility,
    /// Same entity claimed to be in different states at same time
    TemporalConflict,
    /// Mutually exclusive categories
    CategoryExclusion,
}

/// Configuration for contradiction detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContradictionConfig {
    /// Minimum attestation weight difference to consider significant
    pub min_weight_difference: f64,
    /// Whether to detect logical contradictions (requires semantic analysis)
    pub detect_logical: bool,
    /// Time window (seconds) for temporal contradiction detection
    pub temporal_window_secs: u64,
    /// Maximum confidence penalty from contradictions
    pub max_penalty: f64,
}

impl Default for ContradictionConfig {
    fn default() -> Self {
        Self {
            min_weight_difference: 0.1,
            detect_logical: true,
            temporal_window_secs: 3600, // 1 hour
            max_penalty: 0.5,
        }
    }
}

/// Contradiction detector for the DKG
pub struct ContradictionDetector {
    config: ContradictionConfig,
    /// Known logical opposites for semantic contradiction detection
    logical_opposites: HashMap<String, Vec<String>>,
}

impl ContradictionDetector {
    /// Create a new detector with default config
    pub fn new() -> Self {
        let mut detector = Self {
            config: ContradictionConfig::default(),
            logical_opposites: HashMap::new(),
        };
        detector.init_logical_opposites();
        detector
    }

    /// Create with custom config
    pub fn with_config(config: ContradictionConfig) -> Self {
        let mut detector = Self {
            config,
            logical_opposites: HashMap::new(),
        };
        detector.init_logical_opposites();
        detector
    }

    /// Initialize known logical opposites
    fn init_logical_opposites(&mut self) {
        // Temperature
        self.add_opposites("hot", &["cold", "freezing"]);
        self.add_opposites("cold", &["hot", "warm", "burning"]);
        self.add_opposites("warm", &["cold", "freezing"]);

        // Boolean-like
        self.add_opposites("true", &["false"]);
        self.add_opposites("false", &["true"]);
        self.add_opposites("yes", &["no"]);
        self.add_opposites("no", &["yes"]);

        // States
        self.add_opposites("alive", &["dead", "deceased"]);
        self.add_opposites("dead", &["alive", "living"]);
        self.add_opposites("open", &["closed", "shut"]);
        self.add_opposites("closed", &["open"]);

        // Colors (typically mutually exclusive for same property)
        self.add_opposites("blue", &["red", "green", "yellow", "orange", "purple"]);
        self.add_opposites("red", &["blue", "green", "yellow"]);
        self.add_opposites("green", &["blue", "red", "purple"]);

        // Size
        self.add_opposites("large", &["small", "tiny"]);
        self.add_opposites("small", &["large", "huge", "enormous"]);

        // Direction
        self.add_opposites("north", &["south"]);
        self.add_opposites("south", &["north"]);
        self.add_opposites("east", &["west"]);
        self.add_opposites("west", &["east"]);
    }

    fn add_opposites(&mut self, value: &str, opposites: &[&str]) {
        self.logical_opposites.insert(
            value.to_lowercase(),
            opposites.iter().map(|s| s.to_lowercase()).collect(),
        );
    }

    /// Detect contradictions for a triple against a set of existing triples
    pub fn detect_contradictions(
        &self,
        triple: &StoredTriple,
        all_triples: &[StoredTriple],
    ) -> ContradictionAnalysis {
        let mut contradictions = Vec::new();

        for other in all_triples {
            // Skip self
            if other.hash == triple.hash {
                continue;
            }

            // Check for same subject+predicate
            if other.triple.subject == triple.triple.subject
                && other.triple.predicate.url == triple.triple.predicate.url
            {
                if let Some(contradiction) = self.check_value_contradiction(triple, other) {
                    contradictions.push(contradiction);
                }
            }
        }

        // Calculate overall contradiction weight
        let contradiction_weight = self.calculate_contradiction_weight(&contradictions);
        let confidence_penalty =
            (contradiction_weight * self.config.max_penalty).min(self.config.max_penalty);

        ContradictionAnalysis {
            triple_hash: triple.hash.clone(),
            contradictions,
            contradiction_weight,
            confidence_penalty,
        }
    }

    /// Check if two triples with same subject+predicate have contradicting values
    fn check_value_contradiction(
        &self,
        triple: &StoredTriple,
        other: &StoredTriple,
    ) -> Option<Contradiction> {
        let value1 = &triple.triple.object;
        let value2 = &other.triple.object;

        // Check for direct value difference
        if !values_equal(value1, value2) {
            let contradiction_type = if self.config.detect_logical {
                self.classify_contradiction(value1, value2)
            } else {
                ContradictionType::DirectValue
            };

            let strength = self.calculate_contradiction_strength(value1, value2);

            return Some(Contradiction {
                contradicting_hash: other.hash.clone(),
                contradiction_type,
                strength,
                conflicting_value: format_value(value2),
                opposing_weight: other.cached_confidence,
            });
        }

        None
    }

    /// Classify the type of contradiction
    fn classify_contradiction(&self, v1: &TripleValue, v2: &TripleValue) -> ContradictionType {
        // Check for logical opposites
        if let (Some(s1), Some(s2)) = (value_to_string(v1), value_to_string(v2)) {
            let s1_lower = s1.to_lowercase();
            let s2_lower = s2.to_lowercase();

            if let Some(opposites) = self.logical_opposites.get(&s1_lower) {
                if opposites.contains(&s2_lower) {
                    return ContradictionType::LogicalIncompatibility;
                }
            }
        }

        // Check for boolean contradiction
        match (v1, v2) {
            (TripleValue::Boolean(b1), TripleValue::Boolean(b2)) if b1 != b2 => {
                return ContradictionType::LogicalIncompatibility;
            }
            _ => {}
        }

        ContradictionType::DirectValue
    }

    /// Calculate how strong the contradiction is (0.0-1.0)
    fn calculate_contradiction_strength(&self, v1: &TripleValue, v2: &TripleValue) -> f64 {
        match (v1, v2) {
            // Boolean contradictions are absolute
            (TripleValue::Boolean(b1), TripleValue::Boolean(b2)) if b1 != b2 => 1.0,

            // Numeric values: strength based on relative difference
            (TripleValue::Float(f1), TripleValue::Float(f2)) => {
                let diff = (f1 - f2).abs();
                let max = f1.abs().max(f2.abs()).max(1.0);
                (diff / max).min(1.0)
            }
            (TripleValue::Integer(i1), TripleValue::Integer(i2)) => {
                let diff = (i1 - i2).abs() as f64;
                let max = (i1.abs().max(i2.abs()) as f64).max(1.0);
                (diff / max).min(1.0)
            }

            // String values: check for logical opposites
            (TripleValue::String(s1), TripleValue::String(s2)) => {
                let s1_lower = s1.to_lowercase();
                let s2_lower = s2.to_lowercase();

                if let Some(opposites) = self.logical_opposites.get(&s1_lower) {
                    if opposites.contains(&s2_lower) {
                        return 1.0; // Logical opposite
                    }
                }
                0.8 // Different string, moderate contradiction
            }

            // Different types: moderate contradiction
            _ => 0.7,
        }
    }

    /// Calculate overall contradiction weight from all detected contradictions
    fn calculate_contradiction_weight(&self, contradictions: &[Contradiction]) -> f64 {
        if contradictions.is_empty() {
            return 0.0;
        }

        // Weight by both strength and opposing attestation weight
        let total_weight: f64 = contradictions
            .iter()
            .map(|c| c.strength * c.opposing_weight)
            .sum();

        // Normalize to 0-1 range
        (total_weight / contradictions.len() as f64).min(1.0)
    }

    /// Find all contradicting pairs in a set of triples
    pub fn find_all_contradictions(
        &self,
        triples: &[StoredTriple],
    ) -> Vec<(String, String, Contradiction)> {
        let mut pairs = Vec::new();

        for (i, t1) in triples.iter().enumerate() {
            for t2 in triples.iter().skip(i + 1) {
                // Check for same subject+predicate
                if t1.triple.subject == t2.triple.subject
                    && t1.triple.predicate.url == t2.triple.predicate.url
                {
                    if let Some(contradiction) = self.check_value_contradiction(t1, t2) {
                        pairs.push((t1.hash.clone(), t2.hash.clone(), contradiction));
                    }
                }
            }
        }

        pairs
    }

    /// Get contradiction weight for confidence calculation
    pub fn get_contradiction_weight_for_confidence(
        &self,
        triple: &StoredTriple,
        all_triples: &[StoredTriple],
    ) -> f64 {
        let analysis = self.detect_contradictions(triple, all_triples);
        analysis.contradiction_weight
    }
}

impl Default for ContradictionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if two TripleValues are equal
fn values_equal(v1: &TripleValue, v2: &TripleValue) -> bool {
    match (v1, v2) {
        (TripleValue::String(s1), TripleValue::String(s2)) => s1 == s2,
        (TripleValue::Float(f1), TripleValue::Float(f2)) => (f1 - f2).abs() < 1e-10,
        (TripleValue::Integer(i1), TripleValue::Integer(i2)) => i1 == i2,
        (TripleValue::Boolean(b1), TripleValue::Boolean(b2)) => b1 == b2,
        (TripleValue::Reference(r1), TripleValue::Reference(r2)) => r1 == r2,
        _ => false,
    }
}

/// Convert TripleValue to string for comparison
fn value_to_string(v: &TripleValue) -> Option<String> {
    match v {
        TripleValue::String(s) => Some(s.clone()),
        TripleValue::Boolean(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Format a value for display
fn format_value(v: &TripleValue) -> String {
    match v {
        TripleValue::String(s) => s.clone(),
        TripleValue::Float(f) => f.to_string(),
        TripleValue::Integer(i) => i.to_string(),
        TripleValue::Boolean(b) => b.to_string(),
        TripleValue::Reference(r) => format!("ref:{}", r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dkg::VerifiableTriple;

    fn make_stored(subject: &str, predicate: &str, value: TripleValue, hash: &str) -> StoredTriple {
        StoredTriple {
            triple: VerifiableTriple::new(subject, predicate, value),
            hash: hash.to_string(),
            attestation_count: 1,
            cached_confidence: 0.5,
            confidence_computed_at: 0,
        }
    }

    #[test]
    fn test_direct_value_contradiction() {
        let detector = ContradictionDetector::new();

        let t1 = make_stored("sky", "color", TripleValue::String("blue".into()), "h1");
        let t2 = make_stored("sky", "color", TripleValue::String("green".into()), "h2");

        let analysis = detector.detect_contradictions(&t1, &[t1.clone(), t2]);

        assert_eq!(analysis.contradictions.len(), 1);
        assert!(analysis.contradiction_weight > 0.0);
    }

    #[test]
    fn test_boolean_contradiction() {
        let detector = ContradictionDetector::new();

        let t1 = make_stored("claim", "valid", TripleValue::Boolean(true), "h1");
        let t2 = make_stored("claim", "valid", TripleValue::Boolean(false), "h2");

        let analysis = detector.detect_contradictions(&t1, &[t1.clone(), t2]);

        assert_eq!(analysis.contradictions.len(), 1);
        assert_eq!(
            analysis.contradictions[0].contradiction_type,
            ContradictionType::LogicalIncompatibility
        );
        assert!((analysis.contradictions[0].strength - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_logical_opposites() {
        let detector = ContradictionDetector::new();

        let t1 = make_stored(
            "room",
            "temperature",
            TripleValue::String("hot".into()),
            "h1",
        );
        let t2 = make_stored(
            "room",
            "temperature",
            TripleValue::String("cold".into()),
            "h2",
        );

        let analysis = detector.detect_contradictions(&t1, &[t1.clone(), t2]);

        assert_eq!(analysis.contradictions.len(), 1);
        assert_eq!(
            analysis.contradictions[0].contradiction_type,
            ContradictionType::LogicalIncompatibility
        );
    }

    #[test]
    fn test_no_contradiction_same_value() {
        let detector = ContradictionDetector::new();

        let t1 = make_stored("sky", "color", TripleValue::String("blue".into()), "h1");
        let t2 = make_stored("sky", "color", TripleValue::String("blue".into()), "h2");

        let analysis = detector.detect_contradictions(&t1, &[t1.clone(), t2]);

        assert_eq!(analysis.contradictions.len(), 0);
        assert!((analysis.contradiction_weight - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_no_contradiction_different_subject() {
        let detector = ContradictionDetector::new();

        let t1 = make_stored("sky", "color", TripleValue::String("blue".into()), "h1");
        let t2 = make_stored("grass", "color", TripleValue::String("green".into()), "h2");

        let analysis = detector.detect_contradictions(&t1, &[t1.clone(), t2]);

        assert_eq!(analysis.contradictions.len(), 0);
    }

    #[test]
    fn test_numeric_contradiction_strength() {
        let detector = ContradictionDetector::new();

        // Large difference
        let t1 = make_stored("temp", "value", TripleValue::Float(100.0), "h1");
        let t2 = make_stored("temp", "value", TripleValue::Float(0.0), "h2");

        let analysis = detector.detect_contradictions(&t1, &[t1.clone(), t2]);
        assert!(analysis.contradictions[0].strength > 0.9);

        // Small difference
        let t3 = make_stored("temp", "value", TripleValue::Float(100.0), "h3");
        let t4 = make_stored("temp", "value", TripleValue::Float(99.0), "h4");

        let analysis2 = detector.detect_contradictions(&t3, &[t3.clone(), t4]);
        assert!(analysis2.contradictions[0].strength < 0.1);
    }

    #[test]
    fn test_find_all_contradictions() {
        let detector = ContradictionDetector::new();

        let triples = vec![
            make_stored("sky", "color", TripleValue::String("blue".into()), "h1"),
            make_stored("sky", "color", TripleValue::String("green".into()), "h2"),
            make_stored("grass", "color", TripleValue::String("green".into()), "h3"),
        ];

        let pairs = detector.find_all_contradictions(&triples);

        // Only sky/color has a contradiction
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].0 == "h1" || pairs[0].1 == "h1");
        assert!(pairs[0].0 == "h2" || pairs[0].1 == "h2");
    }
}
