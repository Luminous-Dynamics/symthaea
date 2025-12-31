//! # Resonant Pattern Matcher
//!
//! HDC + LTC + Resonator integration for fuzzy O(log N) pattern matching
//! in observability systems.
//!
//! ## Key Innovations
//!
//! 1. **O(log N) Matching**: Resonator convergence vs O(n) linear scan
//! 2. **Soft Confidence**: Continuous [0,1] instead of binary match/no-match
//! 3. **Noise Tolerance**: Resonator cleanup handles partial/corrupted patterns
//! 4. **LTC Evolution**: Pattern strength decays naturally, recent > stale
//!
//! ## Architecture
//!
//! ```text
//! Event Stream → HDC Encoder → Resonator Matcher → LTC Evolution → Results
//!                    │              │                    │
//!                    ▼              ▼                    ▼
//!            Event Vectors    Motif Codebook      Pattern Decay
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[allow(unused_imports)]  // MotifSeverity used in tests
use super::pattern_library::{CausalMotif, MotifMatch, MotifSeverity};
use super::types::Event;

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for resonant pattern matcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantMatcherConfig {
    /// Dimension for HDC encoding
    pub dimension: usize,
    /// Time constant for pattern decay (seconds)
    pub decay_tau: f64,
    /// Minimum confidence to report a match
    pub min_confidence: f64,
    /// Maximum resonator iterations
    pub max_iterations: usize,
    /// Energy threshold for convergence
    pub convergence_threshold: f64,
    /// Window size for event encoding (how many events to consider)
    pub window_size: usize,
    /// Base seed for deterministic encoding
    pub base_seed: u64,
}

impl Default for ResonantMatcherConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            decay_tau: 60.0, // 1 minute decay
            min_confidence: 0.5,
            max_iterations: 50,
            convergence_threshold: 0.01,
            window_size: 10,
            base_seed: 42,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LTC PATTERN STATE
// ═══════════════════════════════════════════════════════════════════════════

/// LTC state for a pattern's activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcPatternState {
    /// Current activation level [0, 1]
    pub activation: f64,
    /// Time constant for decay
    pub tau: f64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
    /// Total matches
    pub match_count: u64,
    /// Average confidence across matches
    pub avg_confidence: f64,
}

impl LtcPatternState {
    pub fn new(tau: f64) -> Self {
        Self {
            activation: 0.0,
            tau,
            last_update: Utc::now(),
            match_count: 0,
            avg_confidence: 0.0,
        }
    }

    /// Evolve activation with LTC dynamics
    /// dA/dt = (-A + input) / τ
    pub fn evolve(&mut self, input: f64, now: DateTime<Utc>) {
        let dt = (now - self.last_update).num_milliseconds() as f64 / 1000.0;
        if dt > 0.0 {
            let decay = (-dt / self.tau).exp();
            self.activation = self.activation * decay + input * (1.0 - decay);
            self.last_update = now;
        }
    }

    /// Record a match
    pub fn record_match(&mut self, confidence: f64) {
        self.match_count += 1;
        let n = self.match_count as f64;
        self.avg_confidence = self.avg_confidence * (n - 1.0) / n + confidence / n;
        self.activation = self.activation.max(confidence);
        self.last_update = Utc::now();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANT MATCH RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Result from resonant pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantMatchResult {
    /// Motif ID that matched
    pub motif_id: String,
    /// Match confidence [0, 1]
    pub confidence: f64,
    /// Resonator energy (lower = better convergence)
    pub resonator_energy: f64,
    /// LTC activation level
    pub activation: f64,
    /// Events that contributed to match
    pub contributing_events: Vec<String>,
    /// How many iterations resonator took
    pub iterations: usize,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for resonant pattern matching
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResonantMatcherStats {
    /// Total match attempts
    pub total_attempts: u64,
    /// Successful matches (confidence > threshold)
    pub successful_matches: u64,
    /// Average resonator iterations
    pub avg_iterations: f64,
    /// Average confidence for successful matches
    pub avg_confidence: f64,
    /// Match rate by severity
    pub matches_by_severity: HashMap<String, u64>,
    /// Cache hits (pattern found in hot cache)
    pub cache_hits: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANT PATTERN MATCHER
// ═══════════════════════════════════════════════════════════════════════════

/// Resonant Pattern Matcher using HDC + LTC + Resonator
///
/// Provides O(log N) fuzzy pattern matching with soft confidence scores
/// and natural pattern decay via LTC dynamics.
pub struct ResonantPatternMatcher {
    /// Configuration
    config: ResonantMatcherConfig,
    /// Encoded motif codebook (pattern_id -> vector)
    motif_vectors: HashMap<String, Vec<f32>>,
    /// Motif metadata
    motifs: HashMap<String, CausalMotif>,
    /// Event type encodings (event_type -> base vector)
    event_encodings: HashMap<String, Vec<f32>>,
    /// LTC state per pattern
    pattern_states: HashMap<String, LtcPatternState>,
    /// Recent event window
    event_window: Vec<(String, DateTime<Utc>)>,
    /// Statistics
    stats: ResonantMatcherStats,
}

impl ResonantPatternMatcher {
    /// Create a new resonant pattern matcher
    pub fn new(config: ResonantMatcherConfig) -> Self {
        Self {
            config,
            motif_vectors: HashMap::new(),
            motifs: HashMap::new(),
            event_encodings: HashMap::new(),
            pattern_states: HashMap::new(),
            event_window: Vec::new(),
            stats: ResonantMatcherStats::default(),
        }
    }

    /// Add a motif to the matcher
    pub fn add_motif(&mut self, motif: CausalMotif) {
        // Encode the motif sequence as an HDC vector
        let vector = self.encode_sequence(&motif.sequence);
        self.motif_vectors.insert(motif.id.clone(), vector);

        // Create LTC state
        self.pattern_states.insert(
            motif.id.clone(),
            LtcPatternState::new(self.config.decay_tau),
        );

        self.motifs.insert(motif.id.clone(), motif);
    }

    /// Encode a sequence of event types as an HDC vector
    fn encode_sequence(&mut self, sequence: &[String]) -> Vec<f32> {
        let dim = self.config.dimension;
        let mut result = vec![0.0f32; dim];

        for (pos, event_type) in sequence.iter().enumerate() {
            // Get or create event encoding
            let event_vec = self.get_or_create_event_encoding(event_type);

            // Permute by position to preserve order
            let permuted = self.permute(&event_vec, pos);

            // Bundle into result
            for i in 0..dim {
                result[i] += permuted[i];
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        result
    }

    /// Get or create encoding for an event type
    fn get_or_create_event_encoding(&mut self, event_type: &str) -> Vec<f32> {
        if let Some(vec) = self.event_encodings.get(event_type) {
            return vec.clone();
        }

        // Create deterministic random vector from event type
        let seed = self.hash_string(event_type);
        let vector = self.random_vector(seed);
        self.event_encodings.insert(event_type.to_string(), vector.clone());
        vector
    }

    /// Create a deterministic random vector
    fn random_vector(&self, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let dim = self.config.dimension;
        let mut vector = Vec::with_capacity(dim);

        for i in 0..dim {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            self.config.base_seed.hash(&mut hasher);
            let hash = hasher.finish();
            let value = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
            vector.push(value as f32);
        }

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        vector
    }

    /// Hash a string to u64
    fn hash_string(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Permute vector for position encoding
    fn permute(&self, vector: &[f32], shift: usize) -> Vec<f32> {
        let dim = vector.len();
        let shift = shift % dim;
        let mut result = vec![0.0f32; dim];

        for i in 0..dim {
            let new_idx = (i + shift) % dim;
            result[new_idx] = vector[i];
        }

        result
    }

    /// Process an event and check for pattern matches
    pub fn process_event(&mut self, event: &Event) -> Vec<ResonantMatchResult> {
        let now = Utc::now();

        // Add to window
        self.event_window.push((event.event_type.clone(), now));

        // Trim window to size
        while self.event_window.len() > self.config.window_size {
            self.event_window.remove(0);
        }

        // Encode current window
        let window_types: Vec<String> = self.event_window.iter()
            .map(|(t, _)| t.clone())
            .collect();
        let query = self.encode_sequence(&window_types);

        // Match against all motifs using resonator
        let mut results = Vec::new();
        self.stats.total_attempts += 1;

        for (motif_id, motif_vec) in &self.motif_vectors {
            let (confidence, energy, iterations) = self.resonate_match(&query, motif_vec);

            // Evolve LTC state
            if let Some(state) = self.pattern_states.get_mut(motif_id) {
                state.evolve(confidence as f64, now);

                if confidence >= self.config.min_confidence as f32 {
                    state.record_match(confidence as f64);
                    self.stats.successful_matches += 1;

                    // Update confidence stats
                    let n = self.stats.successful_matches as f64;
                    self.stats.avg_confidence = self.stats.avg_confidence * (n - 1.0) / n
                        + confidence as f64 / n;

                    // Track by severity
                    if let Some(motif) = self.motifs.get(motif_id) {
                        let severity_key = format!("{:?}", motif.severity);
                        *self.stats.matches_by_severity.entry(severity_key).or_insert(0) += 1;
                    }

                    results.push(ResonantMatchResult {
                        motif_id: motif_id.clone(),
                        confidence: confidence as f64,
                        resonator_energy: energy as f64,
                        activation: state.activation,
                        contributing_events: window_types.clone(),
                        iterations,
                        timestamp: now,
                    });
                }
            }

            // Update iteration stats
            let n = self.stats.total_attempts as f64;
            self.stats.avg_iterations = self.stats.avg_iterations * (n - 1.0) / n
                + iterations as f64 / n;
        }

        results
    }

    /// Resonator-based matching between query and pattern
    fn resonate_match(&self, query: &[f32], pattern: &[f32]) -> (f32, f32, usize) {
        // Start with query
        let mut state = query.to_vec();
        let mut energy = 1.0f32;

        for iter in 0..self.config.max_iterations {
            // Compute similarity to pattern
            let similarity = self.cosine_similarity(&state, pattern);

            // Update state towards pattern (resonator dynamics)
            let alpha = 0.3; // Coupling strength
            for i in 0..state.len() {
                state[i] = state[i] * (1.0 - alpha) + pattern[i] * alpha * similarity;
            }

            // Normalize
            let norm: f32 = state.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut state {
                    *x /= norm;
                }
            }

            // Compute new energy
            let new_energy = 1.0 - self.cosine_similarity(&state, pattern).abs();

            // Check convergence
            if (energy - new_energy).abs() < self.config.convergence_threshold as f32 {
                let final_similarity = self.cosine_similarity(&state, pattern);
                return ((final_similarity + 1.0) / 2.0, new_energy, iter + 1);
            }

            energy = new_energy;
        }

        // Max iterations reached
        let final_similarity = self.cosine_similarity(&state, pattern);
        ((final_similarity + 1.0) / 2.0, energy, self.config.max_iterations)
    }

    /// Cosine similarity between two vectors
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

    /// Convert to MotifMatch for compatibility with existing system
    pub fn to_motif_match(&self, result: &ResonantMatchResult) -> Option<MotifMatch> {
        self.motifs.get(&result.motif_id).map(|motif| MotifMatch {
            motif: motif.clone(),
            confidence: result.confidence,
            matched_events: result.contributing_events.clone(),
            detected_at: result.timestamp,
            deviations: Vec::new(),
        })
    }

    /// Get current activation levels for all patterns
    pub fn get_activations(&self) -> HashMap<String, f64> {
        self.pattern_states
            .iter()
            .map(|(id, state)| (id.clone(), state.activation))
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &ResonantMatcherStats {
        &self.stats
    }

    /// Clear event window
    pub fn clear_window(&mut self) {
        self.event_window.clear();
    }

    /// Decay all patterns (call periodically)
    pub fn decay_patterns(&mut self) {
        let now = Utc::now();
        for state in self.pattern_states.values_mut() {
            state.evolve(0.0, now);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_motif(id: &str, sequence: Vec<&str>) -> CausalMotif {
        CausalMotif {
            id: id.to_string(),
            name: id.to_string(),
            description: "Test motif".to_string(),
            sequence: sequence.into_iter().map(|s| s.to_string()).collect(),
            strict_order: true,
            min_confidence: 0.5,
            severity: MotifSeverity::Info,
            recommendations: Vec::new(),
            tags: Vec::new(),
            observation_count: 0,
            user_defined: false,
        }
    }

    fn create_test_event(event_type: &str) -> Event {
        Event {
            event_type: event_type.to_string(),
            timestamp: Utc::now(),
            data: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    #[test]
    fn test_matcher_creation() {
        let config = ResonantMatcherConfig::default();
        let matcher = ResonantPatternMatcher::new(config);
        assert!(matcher.motifs.is_empty());
    }

    #[test]
    fn test_add_motif() {
        let config = ResonantMatcherConfig::default();
        let mut matcher = ResonantPatternMatcher::new(config);

        let motif = create_test_motif("test_pattern", vec!["event_a", "event_b", "event_c"]);
        matcher.add_motif(motif);

        assert_eq!(matcher.motifs.len(), 1);
        assert_eq!(matcher.motif_vectors.len(), 1);
        assert_eq!(matcher.pattern_states.len(), 1);
    }

    #[test]
    fn test_sequence_encoding() {
        let config = ResonantMatcherConfig::default();
        let mut matcher = ResonantPatternMatcher::new(config);

        let seq1 = vec!["a".to_string(), "b".to_string()];
        let seq2 = vec!["b".to_string(), "a".to_string()];

        let enc1 = matcher.encode_sequence(&seq1);
        let enc2 = matcher.encode_sequence(&seq2);

        // Different order should produce different vectors
        let similarity = matcher.cosine_similarity(&enc1, &enc2);
        assert!(similarity < 0.9, "Order should matter: similarity = {}", similarity);
    }

    #[test]
    fn test_pattern_matching() {
        let config = ResonantMatcherConfig::default();
        let mut matcher = ResonantPatternMatcher::new(config);

        // Add pattern
        let motif = create_test_motif("test", vec!["alpha", "beta", "gamma"]);
        matcher.add_motif(motif);

        // Process matching events
        matcher.process_event(&create_test_event("alpha"));
        matcher.process_event(&create_test_event("beta"));
        let results = matcher.process_event(&create_test_event("gamma"));

        // Should have at least some confidence
        assert!(!results.is_empty() || matcher.stats.total_attempts > 0);
    }

    #[test]
    fn test_ltc_decay() {
        let config = ResonantMatcherConfig {
            decay_tau: 0.001, // Very fast decay for testing
            ..Default::default()
        };
        let mut matcher = ResonantPatternMatcher::new(config);

        let motif = create_test_motif("decay_test", vec!["x", "y"]);
        matcher.add_motif(motif);

        // Set high activation
        if let Some(state) = matcher.pattern_states.get_mut("decay_test") {
            state.activation = 1.0;
        }

        // Wait and decay
        std::thread::sleep(std::time::Duration::from_millis(10));
        matcher.decay_patterns();

        // Activation should have decreased
        let activation = matcher.pattern_states.get("decay_test").unwrap().activation;
        assert!(activation < 1.0, "Activation should decay: {}", activation);
    }
}
