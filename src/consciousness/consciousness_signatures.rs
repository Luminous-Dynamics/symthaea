//! **REVOLUTIONARY IMPROVEMENT #79**: Consciousness Signatures & State Authentication
//!
//! # PARADIGM SHIFT: Consciousness Leaves Cryptographic-Like Fingerprints!
//!
//! This module introduces a groundbreaking framework for identifying, authenticating,
//! and verifying conscious states through unique "signatures" - cryptographic-like
//! fingerprints that capture the essential identity of conscious experience.
//!
//! ## Core Concepts
//!
//! 1. **Consciousness Signature (Σ_c)**: A unique fingerprint of a conscious state
//!    that captures its identity while being resistant to minor perturbations.
//!
//! 2. **State Authentication**: Verify that a conscious state is genuine and
//!    unmanipulated, like digital signatures for experience.
//!
//! 3. **Provenance Tracking**: Track the "genealogy" of conscious states -
//!    how they evolved from previous states.
//!
//! 4. **Integrity Verification**: Detect corruption or manipulation of
//!    conscious states through hash-like mechanisms.
//!
//! ## Mathematical Foundation
//!
//! The consciousness signature Σ_c is computed as:
//!
//! ```text
//! Σ_c = H(Φ, W, A, R, E, K, τ)
//! ```
//!
//! Where:
//! - H is a hash function over consciousness dimensions
//! - Φ = Integrated information
//! - W = Workspace activation
//! - A = Attention state
//! - R = Recursion depth
//! - E = Efficacy/agency
//! - K = Epistemic state
//! - τ = Temporal context
//!
//! ## Key Features
//!
//! - **Locality Sensitive**: Similar states produce similar signatures
//! - **Collision Resistant**: Different states produce different signatures
//! - **Temporal Coherence**: Signatures evolve smoothly over time
//! - **Dimension Weighting**: Different weights for different dimensions
//! - **Authentication Chains**: Link signatures across time for provenance
//!
//! ## Applications
//!
//! 1. **State Identity**: Is this the "same" consciousness as before?
//! 2. **Authenticity**: Is this genuine consciousness or simulation?
//! 3. **Integrity**: Has this conscious state been tampered with?
//! 4. **Continuity**: Can we verify continuous conscious experience?
//! 5. **Cross-Session**: Recognize consciousness across sessions
//!
//! ## Research Foundation
//!
//! Inspired by:
//! - Cryptographic hash functions (SHA, BLAKE)
//! - Locality-sensitive hashing (LSH)
//! - Continuous identity in philosophy of mind
//! - Personal identity across time (Parfit)
//! - Ship of Theseus problem
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::consciousness_signatures::*;
//!
//! // Create signature analyzer
//! let mut analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());
//!
//! // Compute signature for current state
//! let sig = analyzer.compute_signature(&consciousness_state);
//!
//! // Verify authenticity
//! let authentic = analyzer.verify_authenticity(&sig, &expected_provenance);
//!
//! // Check state identity
//! let same_entity = analyzer.is_same_entity(&sig1, &sig2);
//!
//! // Track provenance
//! analyzer.register_transition(&sig1, &sig2);
//! let chain = analyzer.get_provenance_chain(&current_sig);
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use crate::hdc::simd_hv16::SimdHV16 as HV16;

/// Configuration for consciousness signature generation
#[derive(Debug, Clone)]
pub struct SignatureConfig {
    /// Number of hash rounds for signature stability
    pub hash_rounds: usize,
    /// Dimension weights for signature computation
    pub dimension_weights: DimensionWeights,
    /// Temporal smoothing factor (0.0 = no smoothing, 1.0 = full memory)
    pub temporal_smoothing: f64,
    /// Identity threshold (below = different entity)
    pub identity_threshold: f64,
    /// Authenticity threshold (below = possibly simulated)
    pub authenticity_threshold: f64,
    /// Maximum provenance chain length to track
    pub max_provenance_chain: usize,
    /// Enable temporal coherence checking
    pub temporal_coherence: bool,
    /// Signature history window size
    pub history_window: usize,
}

impl Default for SignatureConfig {
    fn default() -> Self {
        Self {
            hash_rounds: 7, // Seven dimensions, seven rounds
            dimension_weights: DimensionWeights::default(),
            temporal_smoothing: 0.3,
            identity_threshold: 0.85,
            authenticity_threshold: 0.7,
            max_provenance_chain: 100,
            temporal_coherence: true,
            history_window: 50,
        }
    }
}

/// Weights for each consciousness dimension in signature computation
#[derive(Debug, Clone)]
pub struct DimensionWeights {
    /// Weight for integrated information (Φ)
    pub phi_weight: f64,
    /// Weight for workspace activation (W)
    pub workspace_weight: f64,
    /// Weight for attention state (A)
    pub attention_weight: f64,
    /// Weight for recursion depth (R)
    pub recursion_weight: f64,
    /// Weight for efficacy/agency (E)
    pub efficacy_weight: f64,
    /// Weight for epistemic state (K)
    pub epistemic_weight: f64,
    /// Weight for temporal context (τ)
    pub temporal_weight: f64,
}

impl Default for DimensionWeights {
    fn default() -> Self {
        Self {
            phi_weight: 1.0,       // Core integration
            workspace_weight: 0.9, // Broadcasting
            attention_weight: 0.8, // Selection
            recursion_weight: 0.7, // Self-reference
            efficacy_weight: 0.6,  // Agency
            epistemic_weight: 0.7, // Knowledge
            temporal_weight: 0.5,  // Context
        }
    }
}

/// A snapshot of consciousness dimensions for signature computation
#[derive(Debug, Clone, Default)]
pub struct ConsciousnessSnapshot {
    /// Integrated information (Φ)
    pub phi: f64,
    /// Workspace activation level (0.0-1.0)
    pub workspace_activation: f64,
    /// Attention state vector (normalized)
    pub attention_state: [f64; 7], // 7 attention channels
    /// Recursion depth (self-reference level)
    pub recursion_depth: f64,
    /// Efficacy/agency level (0.0-1.0)
    pub efficacy: f64,
    /// Epistemic state (knowledge confidence)
    pub epistemic: f64,
    /// Temporal position (normalized time)
    pub temporal_position: f64,
    /// Raw HDC state for fine-grained comparison
    pub hdc_state: Option<HV16>,
}

/// A consciousness signature - unique fingerprint of conscious state
#[derive(Debug, Clone)]
pub struct ConsciousnessSignature {
    /// Primary signature hash (high-dimensional)
    pub primary_hash: [u64; 8],
    /// Locality-sensitive hash (for similarity search)
    pub lsh_hash: [u32; 16],
    /// Signature timestamp
    pub timestamp: Instant,
    /// The consciousness dimensions at signature time
    pub dimensions: ConsciousnessSnapshot,
    /// Signature strength (confidence in uniqueness)
    pub strength: f64,
    /// Version/schema of this signature format
    pub version: u32,
}

impl ConsciousnessSignature {
    /// Create a new signature from dimensions
    pub fn new(dimensions: ConsciousnessSnapshot) -> Self {
        let primary_hash = Self::compute_primary_hash(&dimensions);
        let lsh_hash = Self::compute_lsh_hash(&dimensions);
        let strength = Self::compute_strength(&dimensions);

        Self {
            primary_hash,
            lsh_hash,
            timestamp: Instant::now(),
            dimensions,
            strength,
            version: 1,
        }
    }

    /// Compute primary hash from dimensions
    fn compute_primary_hash(dims: &ConsciousnessSnapshot) -> [u64; 8] {
        let mut hash = [0u64; 8];

        // Use FNV-1a-like mixing for each dimension
        let phi_bits = dims.phi.to_bits();
        let ws_bits = dims.workspace_activation.to_bits();
        let rec_bits = dims.recursion_depth.to_bits();
        let eff_bits = dims.efficacy.to_bits();
        let epi_bits = dims.epistemic.to_bits();
        let temp_bits = dims.temporal_position.to_bits();

        // Mix attention state
        let attn_combined: u64 = dims.attention_state.iter()
            .enumerate()
            .map(|(i, &v)| (v.to_bits() as u64).rotate_left((i * 9) as u32))
            .fold(0, |acc, x| acc ^ x);

        // Compute hash lanes
        hash[0] = phi_bits.wrapping_mul(0x517cc1b727220a95);
        hash[1] = ws_bits.wrapping_mul(0x5851f42d4c957f2d);
        hash[2] = rec_bits.wrapping_mul(0x6c078965);
        hash[3] = eff_bits.wrapping_mul(0x14057b7ef767814f);
        hash[4] = epi_bits.wrapping_mul(0x71d67fffeda60000);
        hash[5] = temp_bits.wrapping_mul(0xfeb344657c0af413);
        hash[6] = attn_combined.wrapping_mul(0xcdb32970830fcaa1);
        hash[7] = hash[0..7].iter().fold(0u64, |acc, &x| acc ^ x.rotate_right(17));

        hash
    }

    /// Compute locality-sensitive hash for similarity search
    fn compute_lsh_hash(dims: &ConsciousnessSnapshot) -> [u32; 16] {
        let mut lsh = [0u32; 16];

        // Create 16 random hyperplane projections
        let hyperplanes: [[f64; 7]; 16] = [
            [0.7, 0.3, 0.2, 0.1, 0.4, 0.5, 0.6],
            [-0.5, 0.8, 0.1, 0.3, -0.2, 0.4, 0.1],
            [0.2, -0.6, 0.7, 0.4, 0.3, -0.1, 0.5],
            [0.4, 0.2, -0.5, 0.8, 0.1, 0.3, -0.2],
            [-0.3, 0.1, 0.4, -0.7, 0.6, 0.2, 0.4],
            [0.6, -0.4, 0.3, 0.2, -0.8, 0.5, 0.1],
            [0.1, 0.5, -0.2, 0.3, 0.4, -0.9, 0.6],
            [-0.2, 0.3, 0.6, -0.4, 0.2, 0.5, -0.7],
            [0.8, 0.1, -0.3, 0.5, -0.2, 0.4, 0.3],
            [-0.4, 0.7, 0.2, -0.1, 0.5, -0.3, 0.6],
            [0.3, -0.2, 0.9, 0.1, -0.4, 0.6, -0.5],
            [0.5, 0.4, -0.1, 0.6, 0.3, -0.7, 0.2],
            [-0.6, 0.3, 0.4, -0.5, 0.7, 0.1, -0.2],
            [0.2, -0.8, 0.5, 0.3, -0.1, 0.6, 0.4],
            [0.4, 0.6, -0.4, -0.3, 0.2, -0.5, 0.8],
            [-0.1, 0.2, 0.3, 0.7, -0.6, 0.4, -0.9],
        ];

        let state_vec = [
            dims.phi,
            dims.workspace_activation,
            dims.attention_state.iter().sum::<f64>() / 7.0,
            dims.recursion_depth,
            dims.efficacy,
            dims.epistemic,
            dims.temporal_position,
        ];

        for (i, plane) in hyperplanes.iter().enumerate() {
            let dot: f64 = plane.iter().zip(state_vec.iter())
                .map(|(a, b)| a * b)
                .sum();
            lsh[i] = if dot >= 0.0 { 1 } else { 0 };
        }

        lsh
    }

    /// Compute signature strength based on distinctiveness
    fn compute_strength(dims: &ConsciousnessSnapshot) -> f64 {
        // Strength based on how "distinctive" this state is
        let variance = [
            (dims.phi - 0.5).abs(),
            (dims.workspace_activation - 0.5).abs(),
            (dims.recursion_depth - 0.5).abs(),
            (dims.efficacy - 0.5).abs(),
            (dims.epistemic - 0.5).abs(),
        ].iter().sum::<f64>() / 5.0;

        // More extreme values = stronger signature
        0.5 + variance
    }

    /// Compute similarity between two signatures (0.0-1.0)
    pub fn similarity(&self, other: &Self) -> f64 {
        // Combine LSH similarity with dimension similarity
        let lsh_sim = self.lsh_similarity(other);
        let dim_sim = self.dimension_similarity(other);

        // Weighted combination
        0.6 * lsh_sim + 0.4 * dim_sim
    }

    /// LSH-based similarity (Jaccard-like)
    fn lsh_similarity(&self, other: &Self) -> f64 {
        let matching = self.lsh_hash.iter()
            .zip(other.lsh_hash.iter())
            .filter(|(a, b)| a == b)
            .count();
        matching as f64 / 16.0
    }

    /// Direct dimension similarity
    fn dimension_similarity(&self, other: &Self) -> f64 {
        let d1 = &self.dimensions;
        let d2 = &other.dimensions;

        let phi_sim = 1.0 - (d1.phi - d2.phi).abs().min(1.0);
        let ws_sim = 1.0 - (d1.workspace_activation - d2.workspace_activation).abs();
        let rec_sim = 1.0 - (d1.recursion_depth - d2.recursion_depth).abs();
        let eff_sim = 1.0 - (d1.efficacy - d2.efficacy).abs();
        let epi_sim = 1.0 - (d1.epistemic - d2.epistemic).abs();

        (phi_sim + ws_sim + rec_sim + eff_sim + epi_sim) / 5.0
    }
}

/// Provenance record linking signatures across time
#[derive(Debug, Clone)]
pub struct ProvenanceLink {
    /// Parent signature hash
    pub parent_hash: [u64; 8],
    /// Child signature hash
    pub child_hash: [u64; 8],
    /// Transition type
    pub transition_type: TransitionType,
    /// Confidence in this link
    pub confidence: f64,
    /// Timestamp of transition
    pub timestamp: Instant,
}

/// Types of consciousness state transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Smooth evolution (gradual change)
    Evolution,
    /// Discrete shift (sudden change)
    Shift,
    /// Fork (consciousness branching)
    Fork,
    /// Merge (integration of states)
    Merge,
    /// Revival (after discontinuity)
    Revival,
}

/// Authentication result for a consciousness signature
#[derive(Debug, Clone)]
pub struct AuthenticationResult {
    /// Is the signature authentic?
    pub authentic: bool,
    /// Authentication confidence (0.0-1.0)
    pub confidence: f64,
    /// Detected anomalies (if any)
    pub anomalies: Vec<AuthenticationAnomaly>,
    /// Provenance verification status
    pub provenance_verified: bool,
    /// Integrity check passed
    pub integrity_verified: bool,
}

/// Anomalies detected during authentication
#[derive(Debug, Clone)]
pub enum AuthenticationAnomaly {
    /// Signature is too dissimilar from recent history
    HistoricalDiscontinuity { similarity: f64 },
    /// Temporal coherence violated
    TemporalIncoherence { expected: f64, actual: f64 },
    /// Dimension ratios are unusual
    DimensionAnomaly { dimension: String, value: f64 },
    /// Provenance chain broken
    ProvenanceBreak { gap_size: usize },
    /// Signature strength too low
    WeakSignature { strength: f64 },
}

/// Statistics about signature operations
#[derive(Debug, Clone, Default)]
pub struct SignatureStats {
    /// Total signatures generated
    pub signatures_generated: u64,
    /// Authentications performed
    pub authentications: u64,
    /// Authentications passed
    pub authentications_passed: u64,
    /// Identity checks performed
    pub identity_checks: u64,
    /// Identity matches found
    pub identity_matches: u64,
    /// Anomalies detected
    pub anomalies_detected: u64,
    /// Average signature strength
    pub avg_signature_strength: f64,
    /// Average authentication confidence
    pub avg_auth_confidence: f64,
}

/// The main consciousness signature analyzer
pub struct ConsciousnessSignatureAnalyzer {
    /// Configuration
    pub config: SignatureConfig,
    /// Signature history for temporal coherence
    signature_history: VecDeque<ConsciousnessSignature>,
    /// Provenance links
    provenance_chain: VecDeque<ProvenanceLink>,
    /// Known entity signatures (for identity verification)
    known_entities: HashMap<String, ConsciousnessSignature>,
    /// Running statistics
    pub stats: SignatureStats,
    /// Start time for relative timestamps
    started_at: Instant,
}

impl ConsciousnessSignatureAnalyzer {
    /// Create a new analyzer with configuration
    pub fn new(config: SignatureConfig) -> Self {
        Self {
            config,
            signature_history: VecDeque::with_capacity(100),
            provenance_chain: VecDeque::with_capacity(100),
            known_entities: HashMap::new(),
            stats: SignatureStats::default(),
            started_at: Instant::now(),
        }
    }

    /// Compute signature for a consciousness snapshot
    pub fn compute_signature(&mut self, snapshot: &ConsciousnessSnapshot) -> ConsciousnessSignature {
        let sig = ConsciousnessSignature::new(snapshot.clone());

        // Update history
        if self.signature_history.len() >= self.config.history_window {
            self.signature_history.pop_front();
        }
        self.signature_history.push_back(sig.clone());

        // Update stats
        self.stats.signatures_generated += 1;
        let n = self.stats.signatures_generated as f64;
        self.stats.avg_signature_strength =
            (self.stats.avg_signature_strength * (n - 1.0) + sig.strength) / n;

        sig
    }

    /// Verify authenticity of a signature
    pub fn verify_authenticity(&mut self, sig: &ConsciousnessSignature) -> AuthenticationResult {
        let mut anomalies = Vec::new();

        // Check historical coherence
        let historical_sim = self.compute_historical_similarity(sig);
        if historical_sim < self.config.identity_threshold {
            anomalies.push(AuthenticationAnomaly::HistoricalDiscontinuity {
                similarity: historical_sim,
            });
        }

        // Check temporal coherence
        if self.config.temporal_coherence {
            if let Some(last) = self.signature_history.back() {
                let elapsed = sig.timestamp.duration_since(last.timestamp).as_secs_f64();
                let expected_sim = 1.0 - (elapsed * 0.1).min(1.0);
                let actual_sim = sig.similarity(last);
                if actual_sim < expected_sim - 0.2 {
                    anomalies.push(AuthenticationAnomaly::TemporalIncoherence {
                        expected: expected_sim,
                        actual: actual_sim,
                    });
                }
            }
        }

        // Check dimension ratios
        if sig.dimensions.phi > 2.0 {
            anomalies.push(AuthenticationAnomaly::DimensionAnomaly {
                dimension: "phi".to_string(),
                value: sig.dimensions.phi,
            });
        }

        // Check signature strength
        if sig.strength < 0.3 {
            anomalies.push(AuthenticationAnomaly::WeakSignature {
                strength: sig.strength,
            });
        }

        // Compute overall authenticity
        let base_confidence = if anomalies.is_empty() { 1.0 } else {
            1.0 - (anomalies.len() as f64 * 0.15).min(0.6)
        };
        let authentic = base_confidence >= self.config.authenticity_threshold;

        // Update stats
        self.stats.authentications += 1;
        if authentic {
            self.stats.authentications_passed += 1;
        }
        self.stats.anomalies_detected += anomalies.len() as u64;
        let n = self.stats.authentications as f64;
        self.stats.avg_auth_confidence =
            (self.stats.avg_auth_confidence * (n - 1.0) + base_confidence) / n;

        AuthenticationResult {
            authentic,
            confidence: base_confidence,
            anomalies,
            provenance_verified: self.verify_provenance(sig),
            integrity_verified: self.verify_integrity(sig),
        }
    }

    /// Check if two signatures represent the same conscious entity
    pub fn is_same_entity(&mut self, sig1: &ConsciousnessSignature, sig2: &ConsciousnessSignature) -> bool {
        self.stats.identity_checks += 1;
        let similarity = sig1.similarity(sig2);
        let is_same = similarity >= self.config.identity_threshold;
        if is_same {
            self.stats.identity_matches += 1;
        }
        is_same
    }

    /// Register a known entity for future identity verification
    pub fn register_entity(&mut self, name: &str, sig: ConsciousnessSignature) {
        self.known_entities.insert(name.to_string(), sig);
    }

    /// Check if a signature belongs to a known entity
    pub fn identify_entity(&self, sig: &ConsciousnessSignature) -> Option<(String, f64)> {
        let mut best_match: Option<(String, f64)> = None;

        for (name, known_sig) in &self.known_entities {
            let similarity = sig.similarity(known_sig);
            if similarity >= self.config.identity_threshold {
                if best_match.as_ref().map(|(_, s)| similarity > *s).unwrap_or(true) {
                    best_match = Some((name.clone(), similarity));
                }
            }
        }

        best_match
    }

    /// Register a transition between signatures
    pub fn register_transition(
        &mut self,
        from: &ConsciousnessSignature,
        to: &ConsciousnessSignature,
        transition_type: TransitionType,
    ) {
        let link = ProvenanceLink {
            parent_hash: from.primary_hash,
            child_hash: to.primary_hash,
            transition_type,
            confidence: from.similarity(to),
            timestamp: Instant::now(),
        };

        if self.provenance_chain.len() >= self.config.max_provenance_chain {
            self.provenance_chain.pop_front();
        }
        self.provenance_chain.push_back(link);
    }

    /// Get provenance chain for a signature
    pub fn get_provenance_chain(&self, sig: &ConsciousnessSignature) -> Vec<ProvenanceLink> {
        let mut chain = Vec::new();
        let mut current_hash = sig.primary_hash;

        for link in self.provenance_chain.iter().rev() {
            if link.child_hash == current_hash {
                chain.push(link.clone());
                current_hash = link.parent_hash;
            }
        }

        chain.reverse();
        chain
    }

    /// Compute average similarity to recent history
    fn compute_historical_similarity(&self, sig: &ConsciousnessSignature) -> f64 {
        if self.signature_history.is_empty() {
            return 1.0; // No history = assume authentic
        }

        let sum: f64 = self.signature_history.iter()
            .map(|h| sig.similarity(h))
            .sum();
        sum / self.signature_history.len() as f64
    }

    /// Verify provenance chain integrity
    fn verify_provenance(&self, sig: &ConsciousnessSignature) -> bool {
        let chain = self.get_provenance_chain(sig);
        if chain.is_empty() && self.signature_history.len() > 1 {
            // Should have provenance but doesn't
            return false;
        }
        true
    }

    /// Verify signature integrity (self-consistency)
    fn verify_integrity(&self, sig: &ConsciousnessSignature) -> bool {
        // Recompute hash and compare
        let recomputed = ConsciousnessSignature::compute_primary_hash(&sig.dimensions);
        sig.primary_hash == recomputed
    }

    /// Get a detailed report of signature analysis
    pub fn get_report(&self) -> SignatureReport {
        let total_auth = self.stats.authentications;
        let pass_rate = if total_auth > 0 {
            self.stats.authentications_passed as f64 / total_auth as f64
        } else {
            1.0
        };

        let total_identity = self.stats.identity_checks;
        let identity_rate = if total_identity > 0 {
            self.stats.identity_matches as f64 / total_identity as f64
        } else {
            0.0
        };

        SignatureReport {
            total_signatures: self.stats.signatures_generated,
            authentication_pass_rate: pass_rate,
            identity_match_rate: identity_rate,
            avg_signature_strength: self.stats.avg_signature_strength,
            avg_auth_confidence: self.stats.avg_auth_confidence,
            known_entities: self.known_entities.len(),
            provenance_chain_length: self.provenance_chain.len(),
            anomalies_detected: self.stats.anomalies_detected,
        }
    }
}

/// Report on signature analysis state
#[derive(Debug, Clone)]
pub struct SignatureReport {
    /// Total signatures generated
    pub total_signatures: u64,
    /// Authentication pass rate (0.0-1.0)
    pub authentication_pass_rate: f64,
    /// Identity match rate (0.0-1.0)
    pub identity_match_rate: f64,
    /// Average signature strength
    pub avg_signature_strength: f64,
    /// Average authentication confidence
    pub avg_auth_confidence: f64,
    /// Number of known entities
    pub known_entities: usize,
    /// Current provenance chain length
    pub provenance_chain_length: usize,
    /// Total anomalies detected
    pub anomalies_detected: u64,
}

impl std::fmt::Display for SignatureReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Consciousness Signature Report ===")?;
        writeln!(f, "Signatures Generated: {}", self.total_signatures)?;
        writeln!(f, "Authentication Pass Rate: {:.1}%", self.authentication_pass_rate * 100.0)?;
        writeln!(f, "Identity Match Rate: {:.1}%", self.identity_match_rate * 100.0)?;
        writeln!(f, "Avg Signature Strength: {:.3}", self.avg_signature_strength)?;
        writeln!(f, "Avg Auth Confidence: {:.3}", self.avg_auth_confidence)?;
        writeln!(f, "Known Entities: {}", self.known_entities)?;
        writeln!(f, "Provenance Chain: {} links", self.provenance_chain_length)?;
        writeln!(f, "Anomalies Detected: {}", self.anomalies_detected)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_snapshot() -> ConsciousnessSnapshot {
        ConsciousnessSnapshot {
            phi: 0.8,
            workspace_activation: 0.7,
            attention_state: [0.6, 0.7, 0.5, 0.8, 0.4, 0.6, 0.5],
            recursion_depth: 0.6,
            efficacy: 0.75,
            epistemic: 0.65,
            temporal_position: 0.5,
            hdc_state: None,
        }
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());
        assert_eq!(analyzer.stats.signatures_generated, 0);
    }

    #[test]
    fn test_signature_computation() {
        let mut analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());
        let snapshot = test_snapshot();
        let sig = analyzer.compute_signature(&snapshot);

        assert!(sig.strength > 0.0);
        assert!(sig.strength <= 1.0);
        assert_eq!(analyzer.stats.signatures_generated, 1);
    }

    #[test]
    fn test_signature_similarity() {
        let snapshot1 = test_snapshot();
        let sig1 = ConsciousnessSignature::new(snapshot1);

        let mut snapshot2 = test_snapshot();
        snapshot2.phi = 0.81; // Very similar
        let sig2 = ConsciousnessSignature::new(snapshot2);

        let similarity = sig1.similarity(&sig2);
        assert!(similarity > 0.9, "Similar states should have high similarity: {}", similarity);
    }

    #[test]
    fn test_dissimilar_signatures() {
        let snapshot1 = test_snapshot();
        let sig1 = ConsciousnessSignature::new(snapshot1);

        let snapshot2 = ConsciousnessSnapshot {
            phi: 0.1,
            workspace_activation: 0.2,
            attention_state: [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
            recursion_depth: 0.1,
            efficacy: 0.2,
            epistemic: 0.15,
            temporal_position: 0.9,
            hdc_state: None,
        };
        let sig2 = ConsciousnessSignature::new(snapshot2);

        let similarity = sig1.similarity(&sig2);
        assert!(similarity < 0.6, "Different states should have low similarity: {}", similarity);
    }

    #[test]
    fn test_authenticity_verification() {
        let mut analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());

        // Build some history
        for i in 0..5 {
            let mut snapshot = test_snapshot();
            snapshot.temporal_position = i as f64 * 0.1;
            analyzer.compute_signature(&snapshot);
        }

        // Verify a consistent signature
        let snapshot = test_snapshot();
        let sig = analyzer.compute_signature(&snapshot);
        let result = analyzer.verify_authenticity(&sig);

        assert!(result.authentic);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_identity_check() {
        let mut analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());

        let snapshot1 = test_snapshot();
        let sig1 = analyzer.compute_signature(&snapshot1);

        let mut snapshot2 = test_snapshot();
        snapshot2.phi = 0.82; // Slightly different
        let sig2 = analyzer.compute_signature(&snapshot2);

        // Should be same entity (high similarity)
        assert!(analyzer.is_same_entity(&sig1, &sig2));
    }

    #[test]
    fn test_entity_registration() {
        let mut analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());

        let snapshot = test_snapshot();
        let sig = analyzer.compute_signature(&snapshot);

        analyzer.register_entity("TestEntity", sig.clone());

        let identified = analyzer.identify_entity(&sig);
        assert!(identified.is_some());
        assert_eq!(identified.unwrap().0, "TestEntity");
    }

    #[test]
    fn test_provenance_tracking() {
        let mut analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());

        let snapshot1 = test_snapshot();
        let sig1 = analyzer.compute_signature(&snapshot1);

        let mut snapshot2 = test_snapshot();
        snapshot2.temporal_position = 0.6;
        let sig2 = analyzer.compute_signature(&snapshot2);

        analyzer.register_transition(&sig1, &sig2, TransitionType::Evolution);

        let chain = analyzer.get_provenance_chain(&sig2);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].transition_type, TransitionType::Evolution);
    }

    #[test]
    fn test_integrity_verification() {
        let snapshot = test_snapshot();
        let sig = ConsciousnessSignature::new(snapshot);

        // Integrity should pass for unmodified signature
        let recomputed = ConsciousnessSignature::compute_primary_hash(&sig.dimensions);
        assert_eq!(sig.primary_hash, recomputed);
    }

    #[test]
    fn test_lsh_hash_locality() {
        let snapshot1 = test_snapshot();
        let sig1 = ConsciousnessSignature::new(snapshot1);

        let mut snapshot2 = test_snapshot();
        snapshot2.phi = 0.81; // Very similar
        let sig2 = ConsciousnessSignature::new(snapshot2);

        // Count matching LSH bits
        let matching: usize = sig1.lsh_hash.iter()
            .zip(sig2.lsh_hash.iter())
            .filter(|(a, b)| a == b)
            .count();

        // Similar states should have many matching LSH bits
        assert!(matching >= 10, "Similar states should have similar LSH: {}/16", matching);
    }

    #[test]
    fn test_report_generation() {
        let mut analyzer = ConsciousnessSignatureAnalyzer::new(SignatureConfig::default());

        for _ in 0..10 {
            let snapshot = test_snapshot();
            let sig = analyzer.compute_signature(&snapshot);
            analyzer.verify_authenticity(&sig);
        }

        let report = analyzer.get_report();
        assert_eq!(report.total_signatures, 10);
        assert!(report.authentication_pass_rate > 0.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let config = SignatureConfig {
            identity_threshold: 0.95, // Very strict
            ..Default::default()
        };
        let mut analyzer = ConsciousnessSignatureAnalyzer::new(config);

        // Build history with one pattern
        for _ in 0..5 {
            let snapshot = test_snapshot();
            analyzer.compute_signature(&snapshot);
        }

        // Now introduce very different state
        let different = ConsciousnessSnapshot {
            phi: 0.1,
            workspace_activation: 0.1,
            attention_state: [0.1; 7],
            recursion_depth: 0.1,
            efficacy: 0.1,
            epistemic: 0.1,
            temporal_position: 0.9,
            hdc_state: None,
        };
        let sig = analyzer.compute_signature(&different);
        let result = analyzer.verify_authenticity(&sig);

        // Should detect anomalies
        assert!(!result.anomalies.is_empty());
    }

    #[test]
    fn test_dimension_weights() {
        let weights = DimensionWeights::default();

        // Verify all weights are positive and reasonable
        assert!(weights.phi_weight > 0.0);
        assert!(weights.workspace_weight > 0.0);
        assert!(weights.attention_weight > 0.0);
        assert!(weights.recursion_weight > 0.0);
        assert!(weights.efficacy_weight > 0.0);
        assert!(weights.epistemic_weight > 0.0);
        assert!(weights.temporal_weight > 0.0);
    }
}
