//! Mycelix WASM SDK
//!
//! WebAssembly bindings for the Mycelix SDK, providing MATL, Epistemic,
//! and Bridge functionality directly in the browser.
//!
//! ## Usage
//!
//! ```javascript
//! import init, {
//!   createPoGQ,
//!   createReputation,
//!   createClaim,
//!   calculateAggregateReputation
//! } from '@mycelix/wasm';
//!
//! await init();
//!
//! const pogq = createPoGQ(0.9, 0.85, 0.1);
//! console.log(`Quality: ${pogq.quality}`);
//! ```

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ============================================================================
// MATL - Mycelix Adaptive Trust Layer
// ============================================================================

/// Proof of Gradient Quality measurement
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct PoGQ {
    quality: f64,
    consistency: f64,
    entropy: f64,
    timestamp: f64,
}

#[wasm_bindgen]
impl PoGQ {
    #[wasm_bindgen(getter)]
    pub fn quality(&self) -> f64 {
        self.quality
    }

    #[wasm_bindgen(getter)]
    pub fn consistency(&self) -> f64 {
        self.consistency
    }

    #[wasm_bindgen(getter)]
    pub fn entropy(&self) -> f64 {
        self.entropy
    }

    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }
}

/// Create a new PoGQ measurement
#[wasm_bindgen(js_name = createPoGQ)]
pub fn create_pogq(quality: f64, consistency: f64, entropy: f64) -> PoGQ {
    PoGQ {
        quality: clamp(quality),
        consistency: clamp(consistency),
        entropy: clamp(entropy),
        timestamp: current_timestamp(),
    }
}

/// Reputation score with Bayesian smoothing
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct Reputation {
    agent_id: String,
    positive_count: u32,
    negative_count: u32,
    last_update: f64,
}

#[wasm_bindgen]
impl Reputation {
    #[wasm_bindgen(getter, js_name = agentId)]
    pub fn agent_id(&self) -> String {
        self.agent_id.clone()
    }

    #[wasm_bindgen(getter, js_name = positiveCount)]
    pub fn positive_count(&self) -> u32 {
        self.positive_count
    }

    #[wasm_bindgen(getter, js_name = negativeCount)]
    pub fn negative_count(&self) -> u32 {
        self.negative_count
    }

    /// Calculate reputation value using Bayesian estimation
    pub fn value(&self) -> f64 {
        let alpha = self.positive_count as f64;
        let beta = self.negative_count as f64;
        alpha / (alpha + beta)
    }

    /// Record a positive interaction
    #[wasm_bindgen(js_name = recordPositive)]
    pub fn record_positive(&self) -> Reputation {
        Reputation {
            agent_id: self.agent_id.clone(),
            positive_count: self.positive_count + 1,
            negative_count: self.negative_count,
            last_update: current_timestamp(),
        }
    }

    /// Record a negative interaction
    #[wasm_bindgen(js_name = recordNegative)]
    pub fn record_negative(&self) -> Reputation {
        Reputation {
            agent_id: self.agent_id.clone(),
            positive_count: self.positive_count,
            negative_count: self.negative_count + 1,
            last_update: current_timestamp(),
        }
    }
}

/// Create a new reputation tracker
#[wasm_bindgen(js_name = createReputation)]
pub fn create_reputation(agent_id: &str) -> Reputation {
    Reputation {
        agent_id: agent_id.to_string(),
        positive_count: 1, // Prior
        negative_count: 1, // Prior
        last_update: current_timestamp(),
    }
}

/// Composite trust score
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct CompositeScore {
    final_score: f64,
    confidence: f64,
    timestamp: f64,
}

#[wasm_bindgen]
impl CompositeScore {
    #[wasm_bindgen(getter, js_name = finalScore)]
    pub fn final_score(&self) -> f64 {
        self.final_score
    }

    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Check if score indicates Byzantine behavior
    #[wasm_bindgen(js_name = isByzantine)]
    pub fn is_byzantine(&self, threshold: Option<f64>) -> bool {
        let thresh = threshold.unwrap_or(0.5);
        self.final_score < thresh
    }

    /// Check if trustworthy
    #[wasm_bindgen(js_name = isTrustworthy)]
    pub fn is_trustworthy(&self, threshold: Option<f64>, min_confidence: Option<f64>) -> bool {
        let thresh = threshold.unwrap_or(0.5);
        let min_conf = min_confidence.unwrap_or(0.5);
        self.final_score >= thresh && self.confidence >= min_conf
    }
}

/// Calculate composite score from PoGQ and reputation
#[wasm_bindgen(js_name = calculateComposite)]
pub fn calculate_composite(pogq: &PoGQ, reputation: &Reputation) -> CompositeScore {
    const QUALITY_WEIGHT: f64 = 0.4;
    const CONSISTENCY_WEIGHT: f64 = 0.3;
    const REPUTATION_WEIGHT: f64 = 0.3;

    let rep_value = reputation.value();
    let quality_contribution = QUALITY_WEIGHT * pogq.quality;
    let consistency_contribution = CONSISTENCY_WEIGHT * pogq.consistency;
    let reputation_contribution = REPUTATION_WEIGHT * rep_value;
    let entropy_penalty = 0.1 * pogq.entropy;

    let final_score = clamp(
        quality_contribution + consistency_contribution + reputation_contribution - entropy_penalty,
    );

    // Confidence based on agreement between components
    let pogq_avg = (pogq.quality + pogq.consistency) / 2.0;
    let deviation = (pogq_avg - rep_value).abs();
    let confidence = clamp(1.0 - deviation);

    CompositeScore {
        final_score,
        confidence,
        timestamp: current_timestamp(),
    }
}

/// Adaptive threshold for anomaly detection
#[wasm_bindgen]
#[derive(Clone)]
pub struct AdaptiveThreshold {
    node_id: String,
    history: Vec<f64>,
    window_size: usize,
    min_threshold: f64,
    sigma_multiplier: f64,
}

#[wasm_bindgen]
impl AdaptiveThreshold {
    /// Add an observation
    pub fn observe(&mut self, score: f64) {
        self.history.push(clamp(score));
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }

    /// Get current threshold
    #[wasm_bindgen(js_name = getThreshold)]
    pub fn get_threshold(&self) -> f64 {
        if self.history.is_empty() {
            return 0.5;
        }

        let mean: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance: f64 = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.history.len() as f64;
        let std_dev = variance.sqrt();

        f64::max(self.min_threshold, mean - self.sigma_multiplier * std_dev)
    }

    /// Check if score is anomalous
    #[wasm_bindgen(js_name = isAnomaly)]
    pub fn is_anomaly(&self, score: f64) -> bool {
        score < self.get_threshold()
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 {
        if self.history.is_empty() {
            return 0.5;
        }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    #[wasm_bindgen(getter, js_name = stdDev)]
    pub fn std_dev(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.history.len() as f64;
        variance.sqrt()
    }
}

/// Create an adaptive threshold tracker
#[wasm_bindgen(js_name = createAdaptiveThreshold)]
pub fn create_adaptive_threshold(
    node_id: &str,
    window_size: Option<usize>,
    min_threshold: Option<f64>,
    sigma_multiplier: Option<f64>,
) -> AdaptiveThreshold {
    AdaptiveThreshold {
        node_id: node_id.to_string(),
        history: Vec::new(),
        window_size: window_size.unwrap_or(100),
        min_threshold: min_threshold.unwrap_or(0.5),
        sigma_multiplier: sigma_multiplier.unwrap_or(2.0),
    }
}

// ============================================================================
// Epistemic Charter
// ============================================================================

/// Empirical verification levels
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EmpiricalLevel {
    E0Unverified = 0,
    E1Testimonial = 1,
    E2PrivateVerify = 2,
    E3Cryptographic = 3,
    E4Consensus = 4,
}

/// Normative scope levels
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NormativeLevel {
    N0Personal = 0,
    N1Communal = 1,
    N2Network = 2,
    N3Universal = 3,
}

/// Materiality levels
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaterialityLevel {
    M0Ephemeral = 0,
    M1Temporal = 1,
    M2Persistent = 2,
    M3Immutable = 3,
}

/// Epistemic claim with 3D classification
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct EpistemicClaim {
    id: String,
    content: String,
    empirical: EmpiricalLevel,
    normative: NormativeLevel,
    materiality: MaterialityLevel,
    issuer: String,
    issued_at: f64,
}

#[wasm_bindgen]
impl EpistemicClaim {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn content(&self) -> String {
        self.content.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn empirical(&self) -> EmpiricalLevel {
        self.empirical
    }

    #[wasm_bindgen(getter)]
    pub fn normative(&self) -> NormativeLevel {
        self.normative
    }

    #[wasm_bindgen(getter)]
    pub fn materiality(&self) -> MaterialityLevel {
        self.materiality
    }

    /// Get classification code (e.g., "E3-N2-M2")
    #[wasm_bindgen(js_name = classificationCode)]
    pub fn classification_code(&self) -> String {
        format!(
            "E{}-N{}-M{}",
            self.empirical as u8, self.normative as u8, self.materiality as u8
        )
    }

    /// Check if claim meets minimum requirements
    #[wasm_bindgen(js_name = meetsMinimum)]
    pub fn meets_minimum(
        &self,
        min_e: EmpiricalLevel,
        min_n: NormativeLevel,
        min_m: Option<MaterialityLevel>,
    ) -> bool {
        let m = min_m.unwrap_or(MaterialityLevel::M0Ephemeral);
        self.empirical >= min_e && self.normative >= min_n && self.materiality >= m
    }
}

/// Claim builder for fluent API
#[wasm_bindgen]
pub struct ClaimBuilder {
    content: String,
    empirical: EmpiricalLevel,
    normative: NormativeLevel,
    materiality: MaterialityLevel,
    issuer: String,
}

#[wasm_bindgen]
impl ClaimBuilder {
    #[wasm_bindgen(js_name = withEmpirical)]
    pub fn with_empirical(mut self, level: EmpiricalLevel) -> ClaimBuilder {
        self.empirical = level;
        self
    }

    #[wasm_bindgen(js_name = withNormative)]
    pub fn with_normative(mut self, level: NormativeLevel) -> ClaimBuilder {
        self.normative = level;
        self
    }

    #[wasm_bindgen(js_name = withMateriality)]
    pub fn with_materiality(mut self, level: MaterialityLevel) -> ClaimBuilder {
        self.materiality = level;
        self
    }

    #[wasm_bindgen(js_name = withIssuer)]
    pub fn with_issuer(mut self, issuer: &str) -> ClaimBuilder {
        self.issuer = issuer.to_string();
        self
    }

    pub fn build(self) -> EpistemicClaim {
        EpistemicClaim {
            id: generate_id("claim"),
            content: self.content,
            empirical: self.empirical,
            normative: self.normative,
            materiality: self.materiality,
            issuer: self.issuer,
            issued_at: current_timestamp(),
        }
    }
}

/// Create a claim builder
#[wasm_bindgen(js_name = createClaim)]
pub fn create_claim(content: &str) -> ClaimBuilder {
    ClaimBuilder {
        content: content.to_string(),
        empirical: EmpiricalLevel::E0Unverified,
        normative: NormativeLevel::N0Personal,
        materiality: MaterialityLevel::M0Ephemeral,
        issuer: String::new(),
    }
}

// ============================================================================
// Bridge - Cross-hApp Communication
// ============================================================================

/// hApp reputation score for aggregation
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct HappScore {
    happ: String,
    score: f64,
    weight: f64,
}

#[wasm_bindgen]
impl HappScore {
    #[wasm_bindgen(constructor)]
    pub fn new(happ: &str, score: f64, weight: f64) -> HappScore {
        HappScore {
            happ: happ.to_string(),
            score: clamp(score),
            weight: f64::max(0.0, weight),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn happ(&self) -> String {
        self.happ.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.score
    }

    #[wasm_bindgen(getter)]
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

/// Calculate weighted aggregate reputation
#[wasm_bindgen(js_name = calculateAggregateReputation)]
pub fn calculate_aggregate_reputation(scores: Vec<HappScore>) -> f64 {
    if scores.is_empty() {
        return 0.5;
    }

    let total_weight: f64 = scores.iter().map(|s| s.weight).sum();
    if total_weight == 0.0 {
        return 0.5;
    }

    let weighted_sum: f64 = scores.iter().map(|s| s.score * s.weight).sum();
    weighted_sum / total_weight
}

// ============================================================================
// Utilities
// ============================================================================

fn clamp(value: f64) -> f64 {
    f64::max(0.0, f64::min(1.0, value))
}

/// Get current timestamp. Uses js_sys in WASM, mock in native tests.
#[cfg(target_arch = "wasm32")]
fn current_timestamp() -> f64 {
    current_timestamp()
}

#[cfg(not(target_arch = "wasm32"))]
fn current_timestamp() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as f64)
        .unwrap_or(0.0)
}

/// Generate a random number. Uses js_sys in WASM, mock in native tests.
#[cfg(target_arch = "wasm32")]
fn random_f64() -> f64 {
    js_sys::Math::random()
}

#[cfg(not(target_arch = "wasm32"))]
fn random_f64() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    (hasher.finish() % 1_000_000) as f64 / 1_000_000.0
}

fn generate_id(prefix: &str) -> String {
    let timestamp = current_timestamp() as u64;
    let random: u32 = (random_f64() * 1_000_000.0) as u32;
    format!("{}_{:x}_{:x}", prefix, timestamp, random)
}

/// Get SDK version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PoGQ Tests
    // =========================================================================

    #[test]
    fn test_pogq_creation() {
        let pogq = create_pogq(0.9, 0.85, 0.1);
        assert_eq!(pogq.quality, 0.9);
        assert_eq!(pogq.consistency, 0.85);
        assert_eq!(pogq.entropy, 0.1);
    }

    #[test]
    fn test_pogq_clamps_values() {
        // Values above 1 should be clamped
        let pogq = create_pogq(1.5, 2.0, 3.0);
        assert_eq!(pogq.quality, 1.0);
        assert_eq!(pogq.consistency, 1.0);
        assert_eq!(pogq.entropy, 1.0);

        // Values below 0 should be clamped
        let pogq = create_pogq(-0.5, -1.0, -0.1);
        assert_eq!(pogq.quality, 0.0);
        assert_eq!(pogq.consistency, 0.0);
        assert_eq!(pogq.entropy, 0.0);
    }

    #[test]
    fn test_pogq_getters() {
        let pogq = create_pogq(0.7, 0.8, 0.05);
        assert_eq!(pogq.quality(), 0.7);
        assert_eq!(pogq.consistency(), 0.8);
        assert_eq!(pogq.entropy(), 0.05);
        assert!(pogq.timestamp() > 0.0);
    }

    // =========================================================================
    // Reputation Tests
    // =========================================================================

    #[test]
    fn test_reputation() {
        let rep = create_reputation("agent_1");
        assert_eq!(rep.value(), 0.5); // 1/(1+1)

        let rep = rep.record_positive();
        assert!(rep.value() > 0.5);

        let rep = rep.record_negative();
        assert!(rep.value() > 0.0);
    }

    #[test]
    fn test_reputation_bayesian_convergence() {
        let mut rep = create_reputation("agent_test");

        // Record many positive interactions
        for _ in 0..100 {
            rep = rep.record_positive();
        }

        // Should converge close to 1.0
        let value = rep.value();
        assert!(value > 0.95, "Expected > 0.95, got {}", value);
    }

    #[test]
    fn test_reputation_agent_id_preserved() {
        let rep = create_reputation("alice");
        assert_eq!(rep.agent_id(), "alice");

        let rep = rep.record_positive();
        assert_eq!(rep.agent_id(), "alice");

        let rep = rep.record_negative();
        assert_eq!(rep.agent_id(), "alice");
    }

    #[test]
    fn test_reputation_counts() {
        let rep = create_reputation("bob");
        assert_eq!(rep.positive_count(), 1); // Prior
        assert_eq!(rep.negative_count(), 1); // Prior

        let rep = rep.record_positive().record_positive();
        assert_eq!(rep.positive_count(), 3);
        assert_eq!(rep.negative_count(), 1);
    }

    // =========================================================================
    // Composite Score Tests
    // =========================================================================

    #[test]
    fn test_composite_score_calculation() {
        let pogq = create_pogq(0.9, 0.85, 0.1);
        let rep = create_reputation("agent_composite");

        let composite = calculate_composite(&pogq, &rep);

        // Should be a valid score
        assert!(composite.final_score() >= 0.0);
        assert!(composite.final_score() <= 1.0);
        assert!(composite.confidence() >= 0.0);
        assert!(composite.confidence() <= 1.0);
    }

    #[test]
    fn test_composite_is_byzantine() {
        let pogq = create_pogq(0.2, 0.1, 0.8); // Poor quality, high entropy
        let rep = create_reputation("malicious");

        let composite = calculate_composite(&pogq, &rep);

        // With default threshold of 0.5
        assert!(composite.is_byzantine(None));

        // With lower threshold
        assert!(!composite.is_byzantine(Some(0.1)));
    }

    #[test]
    fn test_composite_is_trustworthy() {
        let pogq = create_pogq(0.95, 0.92, 0.05);
        let mut rep = create_reputation("trusted");

        // Build up reputation
        for _ in 0..10 {
            rep = rep.record_positive();
        }

        let composite = calculate_composite(&pogq, &rep);

        // Should be trustworthy with default thresholds
        assert!(composite.is_trustworthy(None, None));
    }

    #[test]
    fn test_composite_high_entropy_penalty() {
        let low_entropy = create_pogq(0.9, 0.9, 0.1);
        let high_entropy = create_pogq(0.9, 0.9, 0.9);
        let rep = create_reputation("test");

        let low_score = calculate_composite(&low_entropy, &rep);
        let high_score = calculate_composite(&high_entropy, &rep);

        // High entropy should result in lower score
        assert!(low_score.final_score() > high_score.final_score());
    }

    // =========================================================================
    // Adaptive Threshold Tests
    // =========================================================================

    #[test]
    fn test_adaptive_threshold_initial() {
        let threshold = create_adaptive_threshold("node_1", None, None, None);
        // With no history, should return default 0.5
        assert_eq!(threshold.get_threshold(), 0.5);
    }

    #[test]
    fn test_adaptive_threshold_observe() {
        let mut threshold = create_adaptive_threshold("node_2", Some(10), None, None);

        // Add observations
        for i in 0..5 {
            threshold.observe(0.7 + (i as f64 * 0.01));
        }

        let th = threshold.get_threshold();
        assert!(th > 0.0);
        assert!(th < 1.0);
    }

    #[test]
    fn test_adaptive_threshold_anomaly_detection() {
        let mut threshold = create_adaptive_threshold("node_3", Some(10), Some(0.3), Some(2.0));

        // Add consistent good observations
        for _ in 0..10 {
            threshold.observe(0.9);
        }

        // With mean=0.9, std=0, threshold = max(0.3, 0.9 - 2*0) = 0.9
        // So 0.85 would be anomaly with this logic. Let's add some variance.
        let mut threshold2 = create_adaptive_threshold("node_3b", Some(10), Some(0.3), Some(2.0));
        for i in 0..10 {
            threshold2.observe(0.7 + (i as f64 * 0.04)); // 0.7 to 1.06, avg ~0.88
        }

        // Very low score should be anomaly
        assert!(threshold2.is_anomaly(0.2));

        // Score close to mean should not be anomaly (threshold is mean - 2*std)
        let th = threshold2.get_threshold();
        assert!(threshold2.get_threshold() < 0.9); // Should have a threshold below mean
    }

    #[test]
    fn test_adaptive_threshold_statistics() {
        let mut threshold = create_adaptive_threshold("node_4", Some(5), None, None);

        threshold.observe(0.8);
        threshold.observe(0.8);
        threshold.observe(0.8);

        // Mean should be 0.8
        assert!((threshold.mean() - 0.8).abs() < 0.001);

        // Std dev should be 0 for identical values
        assert!((threshold.std_dev() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_threshold_window_size() {
        let mut threshold = create_adaptive_threshold("node_5", Some(3), None, None);

        threshold.observe(0.5);
        threshold.observe(0.5);
        threshold.observe(0.5);
        threshold.observe(0.9); // This should push out first 0.5

        // Mean should be close to (0.5 + 0.5 + 0.9) / 3 = 0.633...
        let expected = (0.5 + 0.5 + 0.9) / 3.0;
        assert!((threshold.mean() - expected).abs() < 0.01);
    }

    // =========================================================================
    // Epistemic Claim Tests
    // =========================================================================

    #[test]
    fn test_claim_builder() {
        let claim = create_claim("Test claim")
            .with_empirical(EmpiricalLevel::E3Cryptographic)
            .with_normative(NormativeLevel::N2Network)
            .build();

        assert_eq!(claim.empirical, EmpiricalLevel::E3Cryptographic);
        assert_eq!(claim.normative, NormativeLevel::N2Network);
    }

    #[test]
    fn test_claim_classification_code() {
        let claim = create_claim("Verified transaction")
            .with_empirical(EmpiricalLevel::E4Consensus)
            .with_normative(NormativeLevel::N3Universal)
            .with_materiality(MaterialityLevel::M3Immutable)
            .build();

        assert_eq!(claim.classification_code(), "E4-N3-M3");
    }

    #[test]
    fn test_claim_meets_minimum() {
        let claim = create_claim("Network claim")
            .with_empirical(EmpiricalLevel::E3Cryptographic)
            .with_normative(NormativeLevel::N2Network)
            .with_materiality(MaterialityLevel::M2Persistent)
            .build();

        // Should meet lower requirements
        assert!(claim.meets_minimum(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N1Communal,
            Some(MaterialityLevel::M1Temporal)
        ));

        // Should meet exact requirements
        assert!(claim.meets_minimum(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            Some(MaterialityLevel::M2Persistent)
        ));

        // Should not meet higher requirements
        assert!(!claim.meets_minimum(
            EmpiricalLevel::E4Consensus,
            NormativeLevel::N2Network,
            None
        ));
    }

    #[test]
    fn test_claim_default_levels() {
        let claim = create_claim("Simple claim").build();

        assert_eq!(claim.empirical, EmpiricalLevel::E0Unverified);
        assert_eq!(claim.normative, NormativeLevel::N0Personal);
        assert_eq!(claim.materiality, MaterialityLevel::M0Ephemeral);
    }

    #[test]
    fn test_claim_content_and_issuer() {
        let claim = create_claim("Important statement")
            .with_issuer("did:mycelix:alice")
            .build();

        assert_eq!(claim.content(), "Important statement");
        assert!(!claim.id().is_empty());
    }

    #[test]
    fn test_empirical_level_ordering() {
        assert!(EmpiricalLevel::E0Unverified < EmpiricalLevel::E1Testimonial);
        assert!(EmpiricalLevel::E1Testimonial < EmpiricalLevel::E2PrivateVerify);
        assert!(EmpiricalLevel::E2PrivateVerify < EmpiricalLevel::E3Cryptographic);
        assert!(EmpiricalLevel::E3Cryptographic < EmpiricalLevel::E4Consensus);
    }

    #[test]
    fn test_normative_level_ordering() {
        assert!(NormativeLevel::N0Personal < NormativeLevel::N1Communal);
        assert!(NormativeLevel::N1Communal < NormativeLevel::N2Network);
        assert!(NormativeLevel::N2Network < NormativeLevel::N3Universal);
    }

    #[test]
    fn test_materiality_level_ordering() {
        assert!(MaterialityLevel::M0Ephemeral < MaterialityLevel::M1Temporal);
        assert!(MaterialityLevel::M1Temporal < MaterialityLevel::M2Persistent);
        assert!(MaterialityLevel::M2Persistent < MaterialityLevel::M3Immutable);
    }

    // =========================================================================
    // Bridge / Aggregate Reputation Tests
    // =========================================================================

    #[test]
    fn test_aggregate_reputation() {
        let scores = vec![
            HappScore::new("h1", 0.9, 1.0),
            HappScore::new("h2", 0.7, 1.0),
        ];
        let aggregate = calculate_aggregate_reputation(scores);
        assert!((aggregate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_aggregate_empty_scores() {
        let scores: Vec<HappScore> = vec![];
        let aggregate = calculate_aggregate_reputation(scores);
        assert_eq!(aggregate, 0.5); // Default for empty
    }

    #[test]
    fn test_aggregate_weighted() {
        let scores = vec![
            HappScore::new("high_weight", 0.9, 3.0),
            HappScore::new("low_weight", 0.3, 1.0),
        ];
        let aggregate = calculate_aggregate_reputation(scores);

        // (0.9 * 3 + 0.3 * 1) / 4 = 0.75
        assert!((aggregate - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_aggregate_zero_weight() {
        let scores = vec![
            HappScore::new("zero", 0.9, 0.0),
            HappScore::new("also_zero", 0.3, 0.0),
        ];
        let aggregate = calculate_aggregate_reputation(scores);
        assert_eq!(aggregate, 0.5); // Default when total weight is 0
    }

    #[test]
    fn test_happ_score_clamps() {
        let score = HappScore::new("test", 1.5, -1.0);
        assert_eq!(score.score(), 1.0); // Clamped to 1
        assert_eq!(score.weight(), 0.0); // Clamped to 0
    }

    #[test]
    fn test_happ_score_getters() {
        let score = HappScore::new("identity", 0.85, 2.0);
        assert_eq!(score.happ(), "identity");
        assert_eq!(score.score(), 0.85);
        assert_eq!(score.weight(), 2.0);
    }

    // =========================================================================
    // Utility Tests
    // =========================================================================

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(0.5), 0.5);
        assert_eq!(clamp(1.5), 1.0);
        assert_eq!(clamp(-0.5), 0.0);
        assert_eq!(clamp(0.0), 0.0);
        assert_eq!(clamp(1.0), 1.0);
    }

    #[test]
    fn test_generate_id() {
        let id1 = generate_id("test");
        let id2 = generate_id("test");

        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
        // IDs should be unique (technically could collide but extremely unlikely)
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
        // Should be semver format
        assert!(v.contains('.'));
    }
}
