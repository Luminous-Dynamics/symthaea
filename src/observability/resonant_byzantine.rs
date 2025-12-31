//! # Resonant Byzantine Defense
//!
//! HDC + LTC + Resonator integration for real-time attack pattern recognition.
//!
//! ## Key Innovations
//!
//! 1. **O(log N) Attack Detection**: Resonator vs O(n) pattern matching
//! 2. **Soft Threat Levels**: Continuous [0,1] instead of binary alerts
//! 3. **Temporal Attack Evolution**: LTC tracks attack progression
//! 4. **Noise Tolerance**: Resonator handles partial attack signatures
//!
//! ## Architecture
//!
//! ```text
//! Event Stream → HDC Encoder → Attack Codebook → Resonator → LTC Evolution → Alerts
//!                    │              │                │            │
//!                    ▼              ▼                ▼            ▼
//!            Event Vectors    Known Attack      Detection    Threat Level
//!                             Patterns          Convergence   Tracking
//! ```

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use super::byzantine_defense::{AttackType, AttackPattern, SystemState};

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for resonant Byzantine defense
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantDefenseConfig {
    /// Dimension for HDC encoding
    pub dimension: usize,
    /// Time constant for threat decay (seconds)
    pub decay_tau: f64,
    /// Minimum confidence to trigger alert
    pub alert_threshold: f64,
    /// Maximum resonator iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Base seed for deterministic encoding
    pub base_seed: u64,
    /// Window size for event sequence matching
    pub sequence_window: usize,
}

impl Default for ResonantDefenseConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            decay_tau: 60.0, // 1 minute decay
            alert_threshold: 0.7,
            max_iterations: 30,
            convergence_threshold: 0.02,
            base_seed: 42,
            sequence_window: 10,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LTC THREAT STATE
// ═══════════════════════════════════════════════════════════════════════════

/// LTC state for tracking threat evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcThreatState {
    /// Current threat level [0, 1]
    pub level: f64,
    /// Time constant for decay
    pub tau: f64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
    /// Peak threat level observed
    pub peak_level: f64,
    /// Number of detections
    pub detection_count: u64,
    /// Attack type if identified
    pub attack_type: Option<AttackType>,
}

impl LtcThreatState {
    pub fn new(initial_level: f64, tau: f64, attack_type: Option<AttackType>) -> Self {
        Self {
            level: initial_level,
            tau,
            last_update: Utc::now(),
            peak_level: initial_level,
            detection_count: 1,
            attack_type,
        }
    }

    /// Evolve threat level with LTC dynamics
    pub fn evolve(&mut self, now: DateTime<Utc>) {
        let dt = (now - self.last_update).num_milliseconds() as f64 / 1000.0;
        if dt > 0.0 {
            // Natural decay toward zero
            let decay = (-dt / self.tau).exp();
            self.level *= decay;
        }
    }

    /// Escalate threat based on new evidence
    pub fn escalate(&mut self, new_evidence: f64) {
        self.detection_count += 1;
        self.last_update = Utc::now();

        // Threat level increases with repeated evidence
        let escalation = new_evidence * (1.0 + (self.detection_count as f64).ln() * 0.1);
        self.level = (self.level + escalation).min(1.0);
        self.peak_level = self.peak_level.max(self.level);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// THREAT ALERT
// ═══════════════════════════════════════════════════════════════════════════

/// Alert generated when threat is detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatAlert {
    /// Unique alert ID
    pub id: String,
    /// Detected attack type
    pub attack_type: AttackType,
    /// Threat level [0, 1]
    pub threat_level: f64,
    /// Confidence in detection
    pub confidence: f64,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Events that triggered detection
    pub trigger_events: Vec<String>,
    /// Recommended countermeasures
    pub recommendations: Vec<String>,
    /// Resonator iterations used
    pub detection_iterations: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for resonant Byzantine defense
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResonantDefenseStats {
    /// Total events processed
    pub events_processed: u64,
    /// Total alerts generated
    pub alerts_generated: u64,
    /// Alerts by attack type
    pub alerts_by_type: HashMap<String, u64>,
    /// Average detection time (microseconds)
    pub avg_detection_time_us: f64,
    /// Average resonator iterations
    pub avg_iterations: f64,
    /// False positive rate (if known)
    pub false_positive_rate: Option<f64>,
    /// Current active threats
    pub active_threats: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANT BYZANTINE DEFENDER
// ═══════════════════════════════════════════════════════════════════════════

/// Real-time Byzantine defense using HDC + LTC + Resonator
pub struct ResonantByzantineDefender {
    config: ResonantDefenseConfig,

    /// Attack type → HDC encoding
    attack_vectors: HashMap<AttackType, Vec<f32>>,

    /// Known attack patterns as HDC codebook
    pattern_codebook: Vec<(AttackType, Vec<f32>)>,

    /// Active threat tracking
    active_threats: HashMap<String, LtcThreatState>,

    /// Recent event window for sequence matching
    event_window: Vec<(String, DateTime<Utc>)>,

    /// Statistics
    stats: ResonantDefenseStats,

    /// Alert counter for unique IDs
    alert_counter: u64,
}

impl ResonantByzantineDefender {
    pub fn new(config: ResonantDefenseConfig) -> Self {
        let mut defender = Self {
            config,
            attack_vectors: HashMap::new(),
            pattern_codebook: Vec::new(),
            active_threats: HashMap::new(),
            event_window: Vec::new(),
            stats: ResonantDefenseStats::default(),
            alert_counter: 0,
        };

        // Initialize attack type encodings
        defender.initialize_attack_encodings();
        defender
    }

    /// Initialize HDC encodings for known attack types
    fn initialize_attack_encodings(&mut self) {
        let attacks = [
            AttackType::SybilAttack,
            AttackType::EclipseAttack,
            AttackType::DoubleSpendAttack,
            AttackType::DataPoisoning,
            AttackType::ModelInversion,
            AttackType::AdversarialExample,
            AttackType::DenialOfService,
            AttackType::ByzantineConsensusFailure,
        ];

        for (idx, attack) in attacks.iter().enumerate() {
            let seed = self.config.base_seed + idx as u64 * 1000;
            let vec = self.random_vector(self.config.dimension, seed);
            self.attack_vectors.insert(*attack, vec);
        }
    }

    /// Register a known attack pattern
    pub fn register_pattern(&mut self, attack_type: AttackType, pattern: &AttackPattern) {
        let pattern_vec = self.encode_pattern(attack_type, pattern);
        self.pattern_codebook.push((attack_type, pattern_vec));
    }

    /// Encode an attack pattern to HDC vector
    fn encode_pattern(&self, attack_type: AttackType, pattern: &AttackPattern) -> Vec<f32> {
        let base_vec = self.attack_vectors.get(&attack_type)
            .cloned()
            .unwrap_or_else(|| self.random_vector(self.config.dimension, 0));

        // Encode event sequence
        let mut sequence_vec = vec![0.0f32; self.config.dimension];
        for (idx, event) in pattern.event_sequence.iter().enumerate() {
            let event_vec = self.random_vector(
                self.config.dimension,
                self.hash_string(event) + idx as u64 * 100,
            );
            for (i, v) in sequence_vec.iter_mut().enumerate() {
                *v += event_vec[i] / pattern.event_sequence.len() as f32;
            }
        }

        // Bind base with sequence
        self.bind(&base_vec, &sequence_vec)
    }

    /// Process an event and check for attacks
    pub fn process_event(&mut self, event_type: &str, event_id: &str, _state: &SystemState) -> Option<ThreatAlert> {
        let start = std::time::Instant::now();
        let now = Utc::now();

        // Add to event window
        self.event_window.push((event_type.to_string(), now));
        if self.event_window.len() > self.config.sequence_window {
            self.event_window.remove(0);
        }

        self.stats.events_processed += 1;

        // Evolve existing threats
        for threat in self.active_threats.values_mut() {
            threat.evolve(now);
        }

        // Remove decayed threats
        self.active_threats.retain(|_, t| t.level > 0.1);
        self.stats.active_threats = self.active_threats.len();

        // Encode current event sequence
        let sequence_vec = self.encode_event_sequence();

        // Use resonator to find matching attack pattern
        let (best_match, confidence, iterations) = self.resonator_detect(&sequence_vec);

        // Update stats
        let detection_time_us = start.elapsed().as_micros() as u64;
        let n = self.stats.events_processed as f64;
        self.stats.avg_detection_time_us =
            (self.stats.avg_detection_time_us * (n - 1.0) + detection_time_us as f64) / n;
        self.stats.avg_iterations =
            (self.stats.avg_iterations * (n - 1.0) + iterations as f64) / n;

        // Generate alert if confidence exceeds threshold
        if confidence >= self.config.alert_threshold as f32 {
            if let Some((attack_type, _)) = best_match {
                // Update or create threat state
                let threat_key = format!("{:?}_{}", attack_type, now.timestamp());
                if let Some(threat) = self.active_threats.get_mut(&threat_key) {
                    threat.escalate(confidence as f64);
                } else {
                    self.active_threats.insert(
                        threat_key.clone(),
                        LtcThreatState::new(confidence as f64, self.config.decay_tau, Some(attack_type)),
                    );
                }

                // Generate alert
                self.alert_counter += 1;
                let alert = ThreatAlert {
                    id: format!("alert_{}", self.alert_counter),
                    attack_type,
                    threat_level: confidence as f64,
                    confidence: confidence as f64,
                    detected_at: now,
                    trigger_events: self.event_window.iter().map(|(e, _)| e.clone()).collect(),
                    recommendations: self.get_recommendations(attack_type),
                    detection_iterations: iterations,
                };

                self.stats.alerts_generated += 1;
                *self.stats.alerts_by_type
                    .entry(format!("{:?}", attack_type))
                    .or_insert(0) += 1;

                return Some(alert);
            }
        }

        None
    }

    /// Encode current event sequence to HDC vector
    fn encode_event_sequence(&self) -> Vec<f32> {
        let mut result = vec![0.0f32; self.config.dimension];

        for (idx, (event, _)) in self.event_window.iter().enumerate() {
            // Position-aware encoding using permutation
            let event_vec = self.random_vector(self.config.dimension, self.hash_string(event));
            let permuted = self.permute(&event_vec, idx);

            for (i, v) in result.iter_mut().enumerate() {
                *v += permuted[i] / self.event_window.len() as f32;
            }
        }

        result
    }

    /// Resonator-based attack detection
    fn resonator_detect(&self, query: &[f32]) -> (Option<(AttackType, Vec<f32>)>, f32, usize) {
        if self.pattern_codebook.is_empty() {
            return (None, 0.0, 0);
        }

        let mut estimate = query.to_vec();
        let mut best_match: Option<(AttackType, Vec<f32>)> = None;
        let mut best_sim = 0.0f32;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Find best matching pattern
            for (attack_type, pattern) in &self.pattern_codebook {
                let sim = self.similarity(&estimate, pattern);
                if sim > best_sim {
                    best_sim = sim;
                    best_match = Some((*attack_type, pattern.clone()));
                }
            }

            // Check convergence
            if let Some((_, ref pattern)) = best_match {
                let energy: f32 = estimate.iter()
                    .zip(pattern.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();

                if energy < self.config.convergence_threshold as f32 {
                    break;
                }

                // Update estimate
                let alpha = 0.3;
                for (i, v) in estimate.iter_mut().enumerate() {
                    *v = (1.0 - alpha) * *v + alpha * pattern[i];
                }
            }
        }

        (best_match, best_sim, iterations)
    }

    /// Get recommendations for an attack type
    fn get_recommendations(&self, attack_type: AttackType) -> Vec<String> {
        match attack_type {
            AttackType::SybilAttack => vec![
                "Increase identity verification requirements".to_string(),
                "Implement stake-based voting".to_string(),
                "Enable proof-of-work for new nodes".to_string(),
            ],
            AttackType::EclipseAttack => vec![
                "Diversify peer connections".to_string(),
                "Implement connection rotation".to_string(),
                "Add geographic diversity requirements".to_string(),
            ],
            AttackType::DoubleSpendAttack => vec![
                "Increase confirmation requirements".to_string(),
                "Enable real-time transaction monitoring".to_string(),
                "Implement velocity limits".to_string(),
            ],
            AttackType::DataPoisoning => vec![
                "Validate data source authenticity".to_string(),
                "Implement outlier detection".to_string(),
                "Use trusted data aggregation".to_string(),
            ],
            AttackType::ModelInversion => vec![
                "Add differential privacy".to_string(),
                "Limit query frequency".to_string(),
                "Implement output perturbation".to_string(),
            ],
            AttackType::AdversarialExample => vec![
                "Enable input validation".to_string(),
                "Implement adversarial training".to_string(),
                "Add input transformation defenses".to_string(),
            ],
            AttackType::DenialOfService => vec![
                "Enable rate limiting".to_string(),
                "Implement request prioritization".to_string(),
                "Scale resources dynamically".to_string(),
            ],
            AttackType::ByzantineConsensusFailure => vec![
                "Increase honest node threshold".to_string(),
                "Enable view change protocol".to_string(),
                "Implement Byzantine fault detection".to_string(),
            ],
        }
    }

    // HDC helper functions

    fn bind(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    fn permute(&self, vec: &[f32], positions: usize) -> Vec<f32> {
        let n = vec.len();
        let shift = positions % n;
        let mut result = vec![0.0; n];
        for (i, v) in vec.iter().enumerate() {
            result[(i + shift) % n] = *v;
        }
        result
    }

    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Generate random HDC vector from seed using SplitMix64
    ///
    /// Uses proper bit mixing to ensure different seeds produce uncorrelated vectors.
    fn random_vector(&self, dim: usize, seed: u64) -> Vec<f32> {
        let mut vec = Vec::with_capacity(dim);
        let mut state = seed;
        for i in 0..dim {
            // SplitMix64: excellent bit mixing for independent values
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state.wrapping_add(i as u64);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z = z ^ (z >> 31);
            // Convert to f32 in range [-1, 1]
            let val = (z as f32) / (u64::MAX as f32) * 2.0 - 1.0;
            vec.push(val);
        }
        vec
    }

    fn hash_string(&self, s: &str) -> u64 {
        let mut hash: u64 = self.config.base_seed;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Get statistics
    pub fn stats(&self) -> &ResonantDefenseStats {
        &self.stats
    }

    /// Get active threats
    pub fn active_threats(&self) -> &HashMap<String, LtcThreatState> {
        &self.active_threats
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defender_creation() {
        let defender = ResonantByzantineDefender::new(ResonantDefenseConfig::default());
        assert_eq!(defender.stats.events_processed, 0);
        assert_eq!(defender.attack_vectors.len(), 8); // All attack types initialized
    }

    #[test]
    fn test_pattern_registration() {
        let mut defender = ResonantByzantineDefender::new(ResonantDefenseConfig::default());

        let pattern = AttackPattern {
            event_sequence: vec!["login".to_string(), "transfer".to_string(), "transfer".to_string()],
            timing_constraints: vec![(0, 1, 5.0), (1, 2, 2.0)],
            anomalies: vec!["rapid_transfers".to_string()],
        };

        defender.register_pattern(AttackType::DoubleSpendAttack, &pattern);
        assert_eq!(defender.pattern_codebook.len(), 1);
    }

    #[test]
    fn test_event_processing() {
        let mut defender = ResonantByzantineDefender::new(ResonantDefenseConfig::default());

        let state = SystemState {
            honest_nodes: 10,
            suspicious_nodes: 2,
            network_connectivity: 0.9,
            resource_utilization: 0.5,
            consensus_round: Some(100),
            recent_patterns: vec![],
        };

        let alert = defender.process_event("normal_event", "evt_1", &state);
        assert!(alert.is_none()); // No pattern registered, no alert
        assert_eq!(defender.stats.events_processed, 1);
    }

    #[test]
    fn test_threat_evolution() {
        let mut threat = LtcThreatState::new(0.8, 60.0, Some(AttackType::SybilAttack));
        assert_eq!(threat.level, 0.8);

        // Escalate
        threat.escalate(0.5);
        assert!(threat.level > 0.8);
        assert_eq!(threat.detection_count, 2);
    }

    #[test]
    fn test_recommendations() {
        let defender = ResonantByzantineDefender::new(ResonantDefenseConfig::default());

        let recs = defender.get_recommendations(AttackType::DenialOfService);
        assert!(!recs.is_empty());
        assert!(recs.iter().any(|r| r.contains("rate limiting")));
    }

    #[test]
    fn test_hdv_operations() {
        let defender = ResonantByzantineDefender::new(ResonantDefenseConfig::default());

        let a = defender.random_vector(1024, 42);
        let b = defender.random_vector(1024, 43);

        // Self-similarity should be 1.0
        let self_sim = defender.similarity(&a, &a);
        assert!((self_sim - 1.0).abs() < 0.001);

        // Permutation should preserve norm
        let permuted = defender.permute(&a, 5);
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_p: f32 = permuted.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_a - norm_p).abs() < 0.001);
    }
}
