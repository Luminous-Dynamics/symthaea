//! Byzantine↔Identity Bridge
//!
//! Connects the MATL Byzantine detection system to identity verification,
//! enabling trust assessments that combine reputation scores with Byzantine
//! fault tolerance analysis.
//!
//! # Architecture
//!
//! ```text
//! DID Activity → Identity Reputation
//!                     ↓
//! Aggregated Reputation (0.0-1.0)
//!                     ↓
//! MATL Engine.evaluate_node()
//!                     ↓
//! NodeEvaluation + NetworkEvaluation
//!                     ↓
//! Identity Trust Assessment
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut coordinator = ByzantineIdentityCoordinator::new();
//!
//! // Evaluate a DID's trustworthiness
//! let assessment = coordinator.evaluate_identity(
//!     "did:mycelix:abc123",
//!     &identity_reputation,
//! );
//!
//! if assessment.is_trustworthy() {
//!     // Allow high-stakes operations
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{CrossHappReputation, HappReputationScore};
use crate::matl::{
    KVector, MatlEngine, NetworkEvaluation, NetworkStatus, NodeEvaluation, ProofOfGradientQuality,
};

/// Trust level classification for identities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Highly trusted - passed all checks, high reputation
    Trusted,
    /// Conditionally trusted - minor concerns but generally acceptable
    Conditional,
    /// Suspicious - anomalies detected, requires verification
    Suspicious,
    /// Untrusted - Byzantine behavior detected
    Untrusted,
    /// Unknown - insufficient data for assessment
    Unknown,
}

impl TrustLevel {
    /// Check if this trust level allows high-stakes operations
    pub fn allows_high_stakes(&self) -> bool {
        matches!(self, TrustLevel::Trusted)
    }

    /// Check if this trust level allows standard operations
    pub fn allows_standard_ops(&self) -> bool {
        matches!(self, TrustLevel::Trusted | TrustLevel::Conditional)
    }

    /// Check if this trust level requires additional verification
    pub fn requires_verification(&self) -> bool {
        matches!(self, TrustLevel::Suspicious | TrustLevel::Unknown)
    }
}

/// Complete trust assessment for an identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityTrustAssessment {
    /// The DID being assessed
    pub did: String,
    /// Overall trust level
    pub trust_level: TrustLevel,
    /// Trust score (0.0-1.0)
    pub trust_score: f64,
    /// MATL node evaluation results
    pub matl_evaluation: MatlNodeResult,
    /// Network status at time of evaluation
    pub network_status: NetworkStatusSummary,
    /// Reasons for the trust determination
    pub reasons: Vec<String>,
    /// Timestamp of assessment
    pub assessed_at: u64,
}

impl IdentityTrustAssessment {
    /// Check if identity is trustworthy for the given operation type
    pub fn is_trustworthy(&self) -> bool {
        self.trust_level.allows_standard_ops()
    }

    /// Check if identity can perform high-stakes operations
    pub fn can_perform_high_stakes(&self) -> bool {
        self.trust_level.allows_high_stakes() && self.trust_score >= 0.8
    }
}

/// Serializable MATL node evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlNodeResult {
    /// Composite score from MATL
    pub composite_score: f64,
    /// Per-node adaptive threshold
    pub node_threshold: f64,
    /// Whether node is flagged as anomalous
    pub is_anomalous: bool,
    /// Whether node is in a suspicious cluster
    pub in_suspicious_cluster: bool,
}

impl From<&NodeEvaluation> for MatlNodeResult {
    fn from(eval: &NodeEvaluation) -> Self {
        Self {
            composite_score: eval.composite_score,
            node_threshold: eval.node_threshold,
            is_anomalous: eval.is_node_anomalous,
            in_suspicious_cluster: eval.is_in_suspicious_cluster,
        }
    }
}

/// Serializable network status summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatusSummary {
    /// Estimated Byzantine fraction
    pub byzantine_fraction: f64,
    /// Current Byzantine tolerance threshold
    pub tolerance_threshold: f64,
    /// Network health status
    pub health: NetworkHealth,
}

/// Network health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkHealth {
    /// Network is operating normally
    Healthy,
    /// Elevated Byzantine activity detected
    Warning,
    /// Network is under active attack
    UnderAttack,
}

impl From<NetworkStatus> for NetworkHealth {
    fn from(status: NetworkStatus) -> Self {
        match status {
            NetworkStatus::Healthy => NetworkHealth::Healthy,
            NetworkStatus::Warning => NetworkHealth::Warning,
            NetworkStatus::UnderAttack => NetworkHealth::UnderAttack,
        }
    }
}

impl From<&NetworkEvaluation> for NetworkStatusSummary {
    fn from(eval: &NetworkEvaluation) -> Self {
        Self {
            byzantine_fraction: eval.estimated_byzantine_fraction,
            tolerance_threshold: eval.adaptive_byzantine_threshold,
            health: eval.status.into(),
        }
    }
}

/// Configuration for the Byzantine↔Identity coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Minimum reputation score to be considered trusted
    pub min_trusted_reputation: f64,
    /// Minimum reputation score to be considered conditional
    pub min_conditional_reputation: f64,
    /// Minimum interactions required for assessment
    pub min_interactions: u64,
    /// Weight given to MATL score vs reputation (0.0-1.0)
    pub matl_weight: f64,
    /// MATL engine window size
    pub matl_window_size: usize,
    /// MATL hierarchy levels
    pub matl_hierarchy_levels: usize,
    /// MATL minimum cluster size
    pub matl_min_cluster_size: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            min_trusted_reputation: 0.8,
            min_conditional_reputation: 0.5,
            min_interactions: 5,
            matl_weight: 0.6, // 60% MATL, 40% reputation
            matl_window_size: 100,
            matl_hierarchy_levels: 3,
            matl_min_cluster_size: 3,
        }
    }
}

/// Coordinator bridging Byzantine detection and identity verification
pub struct ByzantineIdentityCoordinator {
    /// MATL engine for Byzantine detection
    matl_engine: MatlEngine,
    /// Configuration
    config: CoordinatorConfig,
    /// Mapping from DID to MATL node ID
    did_to_node: HashMap<String, String>,
    /// Cache of recent assessments
    assessment_cache: HashMap<String, IdentityTrustAssessment>,
    /// Cache TTL in seconds
    cache_ttl: u64,
}

impl ByzantineIdentityCoordinator {
    /// Create a new coordinator with default configuration
    pub fn new() -> Self {
        Self::with_config(CoordinatorConfig::default())
    }

    /// Create a new coordinator with custom configuration
    pub fn with_config(config: CoordinatorConfig) -> Self {
        Self {
            matl_engine: MatlEngine::new(
                config.matl_window_size,
                config.matl_hierarchy_levels,
                config.matl_min_cluster_size,
            ),
            config,
            did_to_node: HashMap::new(),
            assessment_cache: HashMap::new(),
            cache_ttl: 300, // 5 minutes default
        }
    }

    /// Get or create a node ID for a DID
    fn get_node_id(&mut self, did: &str) -> String {
        self.did_to_node
            .entry(did.to_string())
            .or_insert_with(|| format!("node:{}", did.replace(":", "_")))
            .clone()
    }

    /// Evaluate an identity's trustworthiness
    pub fn evaluate_identity(
        &mut self,
        did: &str,
        reputation: &CrossHappReputation,
    ) -> IdentityTrustAssessment {
        let now = current_timestamp();

        // Check cache
        if let Some(cached) = self.assessment_cache.get(did) {
            if now - cached.assessed_at < self.cache_ttl {
                return cached.clone();
            }
        }

        let node_id = self.get_node_id(did);
        let mut reasons = Vec::new();

        // Create PoGQ from reputation data
        let pogq = self.create_pogq_from_reputation(reputation);

        // Evaluate with MATL
        let (node_eval, net_eval) =
            self.matl_engine
                .evaluate_node(&node_id, &pogq, reputation.aggregate);

        // Determine trust level
        let trust_level =
            self.determine_trust_level(reputation, &node_eval, &net_eval, &mut reasons);

        // Compute combined trust score
        let trust_score = self.compute_trust_score(reputation, &node_eval);

        let assessment = IdentityTrustAssessment {
            did: did.to_string(),
            trust_level,
            trust_score,
            matl_evaluation: MatlNodeResult::from(&node_eval),
            network_status: NetworkStatusSummary::from(&net_eval),
            reasons,
            assessed_at: now,
        };

        // Cache the assessment
        self.assessment_cache
            .insert(did.to_string(), assessment.clone());

        assessment
    }

    /// Evaluate multiple identities efficiently
    pub fn evaluate_batch(
        &mut self,
        identities: &[(String, CrossHappReputation)],
    ) -> Vec<IdentityTrustAssessment> {
        identities
            .iter()
            .map(|(did, rep)| self.evaluate_identity(did, rep))
            .collect()
    }

    /// Create a PoGQ from reputation data
    fn create_pogq_from_reputation(
        &self,
        reputation: &CrossHappReputation,
    ) -> ProofOfGradientQuality {
        // Map reputation metrics to PoGQ fields
        // - quality_score: aggregate reputation
        // - consistency_score: variance across hApps (lower variance = higher consistency)
        // - attack_resistance: based on interaction count and score stability

        let quality_score = reputation.aggregate;

        // Compute consistency from score variance
        let consistency_score = if reputation.scores.len() > 1 {
            let mean = reputation.aggregate;
            let variance: f64 = reputation
                .scores
                .iter()
                .map(|s| (s.score - mean).powi(2))
                .sum::<f64>()
                / reputation.scores.len() as f64;
            let std_dev = variance.sqrt();
            // Lower std dev = higher consistency
            (1.0 - std_dev.min(0.5) * 2.0).max(0.0)
        } else {
            0.5 // Default for single source
        };

        // Attack resistance based on interaction volume
        let total_interactions = reputation.total_interactions();
        let attack_resistance = if total_interactions >= 100 {
            0.9
        } else if total_interactions >= 50 {
            0.7
        } else if total_interactions >= 10 {
            0.5
        } else {
            0.3
        };

        ProofOfGradientQuality::new(quality_score, consistency_score, attack_resistance)
    }

    /// Determine trust level based on all factors
    fn determine_trust_level(
        &self,
        reputation: &CrossHappReputation,
        node_eval: &NodeEvaluation,
        net_eval: &NetworkEvaluation,
        reasons: &mut Vec<String>,
    ) -> TrustLevel {
        // Check for Byzantine behavior
        if node_eval.is_in_suspicious_cluster {
            reasons.push("In suspicious cluster detected by MATL".to_string());
            return TrustLevel::Untrusted;
        }

        if node_eval.is_node_anomalous {
            reasons.push(format!(
                "Anomalous behavior: score {:.2} below threshold {:.2}",
                node_eval.composite_score, node_eval.node_threshold
            ));
            return TrustLevel::Suspicious;
        }

        // Check network conditions
        if net_eval.status == NetworkStatus::UnderAttack {
            reasons.push("Network under attack - elevated scrutiny".to_string());
            // Don't automatically distrust, but note the condition
        }

        // Check reputation thresholds
        let total_interactions = reputation.total_interactions();

        if total_interactions < self.config.min_interactions {
            reasons.push(format!(
                "Insufficient interactions: {} < {}",
                total_interactions, self.config.min_interactions
            ));
            return TrustLevel::Unknown;
        }

        if reputation.aggregate >= self.config.min_trusted_reputation {
            reasons.push(format!(
                "High reputation: {:.2} >= {:.2}",
                reputation.aggregate, self.config.min_trusted_reputation
            ));
            return TrustLevel::Trusted;
        }

        if reputation.aggregate >= self.config.min_conditional_reputation {
            reasons.push(format!(
                "Moderate reputation: {:.2} >= {:.2}",
                reputation.aggregate, self.config.min_conditional_reputation
            ));
            return TrustLevel::Conditional;
        }

        reasons.push(format!(
            "Low reputation: {:.2} < {:.2}",
            reputation.aggregate, self.config.min_conditional_reputation
        ));
        TrustLevel::Suspicious
    }

    /// Compute combined trust score
    fn compute_trust_score(
        &self,
        reputation: &CrossHappReputation,
        node_eval: &NodeEvaluation,
    ) -> f64 {
        let matl_score = node_eval.composite_score;
        let rep_score = reputation.aggregate;

        // Weighted combination
        let combined =
            self.config.matl_weight * matl_score + (1.0 - self.config.matl_weight) * rep_score;

        // Apply penalties
        let mut score = combined;

        if node_eval.is_node_anomalous {
            score *= 0.5; // 50% penalty for anomalous behavior
        }

        if node_eval.is_in_suspicious_cluster {
            score *= 0.2; // 80% penalty for suspicious cluster
        }

        score.clamp(0.0, 1.0)
    }

    /// Get current network status
    pub fn network_status(&self) -> NetworkStatusSummary {
        // Get from most recent evaluation or estimate
        let byzantine_fraction = 0.0; // Would need to track this
        let tolerance_threshold = 0.33; // Default

        NetworkStatusSummary {
            byzantine_fraction,
            tolerance_threshold,
            health: NetworkHealth::Healthy,
        }
    }

    /// Clear the assessment cache
    pub fn clear_cache(&mut self) {
        self.assessment_cache.clear();
    }

    /// Get the number of tracked identities
    pub fn tracked_identity_count(&self) -> usize {
        self.did_to_node.len()
    }

    /// Check if a DID has been evaluated
    pub fn has_assessment(&self, did: &str) -> bool {
        self.assessment_cache.contains_key(did)
    }

    /// Get a cached assessment if available
    pub fn get_cached_assessment(&self, did: &str) -> Option<&IdentityTrustAssessment> {
        self.assessment_cache.get(did)
    }

    /// Report suspicious activity for a DID
    pub fn report_suspicious(&mut self, did: &str, _reason: &str) {
        // Invalidate cache
        self.assessment_cache.remove(did);

        // Could also feed this into MATL as a negative signal
        // For now, just clear cache so next evaluation recalculates
    }

    /// Report positive activity for a DID
    pub fn report_positive(&mut self, did: &str) {
        // Invalidate cache to force recalculation
        self.assessment_cache.remove(did);
    }
}

impl Default for ByzantineIdentityCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Quick check if a reputation is trustworthy
pub fn is_reputation_trustworthy(reputation: &CrossHappReputation, threshold: f64) -> bool {
    reputation.aggregate >= threshold && reputation.total_interactions() >= 5
}

/// Compute a simple trust score from reputation alone
pub fn simple_trust_score(reputation: &CrossHappReputation) -> f64 {
    let base_score = reputation.aggregate;

    // Boost for high interaction count
    let interaction_boost = match reputation.total_interactions() {
        0..=10 => 0.0,
        11..=50 => 0.05,
        51..=100 => 0.1,
        _ => 0.15,
    };

    (base_score + interaction_boost).min(1.0)
}

// =============================================================================
// Trust Score Cache with Event-Based Invalidation (FIND-009)
// =============================================================================

/// Default invalidation threshold - invalidate if reputation changes by more than 10%
pub const DEFAULT_INVALIDATION_THRESHOLD: f64 = 0.1;

/// Event indicating a reputation change occurred
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationChangeEvent {
    /// The agent/DID whose reputation changed
    pub agent_id: String,
    /// Previous reputation score
    pub old_score: f64,
    /// New reputation score
    pub new_score: f64,
    /// Timestamp of the change
    pub timestamp: u64,
    /// Source of the change
    pub source: String,
}

impl ReputationChangeEvent {
    /// Create a new reputation change event
    pub fn new(
        agent_id: impl Into<String>,
        old_score: f64,
        new_score: f64,
        source: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            old_score,
            new_score,
            timestamp: current_timestamp(),
            source: source.into(),
        }
    }

    /// Get the magnitude of the score change
    pub fn change_magnitude(&self) -> f64 {
        (self.new_score - self.old_score).abs()
    }

    /// Check if this is a significant change (exceeds threshold)
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.change_magnitude() > threshold
    }
}

/// Cached trust score entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedScore {
    /// The cached assessment
    assessment: IdentityTrustAssessment,
    /// Last known reputation score for invalidation detection
    last_reputation: f64,
    /// Cache entry creation time
    cached_at: u64,
}

/// Trust score cache with event-based invalidation (FIND-009)
///
/// This cache extends time-based expiration with event-based invalidation:
/// - Cache entries expire after TTL
/// - Cache entries are invalidated when reputation changes significantly
/// - Significant change is defined by `invalidation_threshold`
#[derive(Debug)]
pub struct TrustScoreCache {
    /// Cached scores indexed by agent/DID
    cache: HashMap<String, CachedScore>,
    /// Time-to-live in seconds
    ttl: u64,
    /// Invalidation threshold - invalidate if reputation changes by more than this
    invalidation_threshold: f64,
    /// Event log for debugging/auditing (bounded)
    event_log: std::collections::VecDeque<ReputationChangeEvent>,
    /// Maximum event log size
    max_event_log_size: usize,
}

impl Default for TrustScoreCache {
    fn default() -> Self {
        Self {
            cache: HashMap::new(),
            ttl: 300, // 5 minutes
            invalidation_threshold: DEFAULT_INVALIDATION_THRESHOLD,
            event_log: std::collections::VecDeque::new(),
            max_event_log_size: 1000,
        }
    }
}

impl TrustScoreCache {
    /// Create a new cache with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a cache with custom TTL and invalidation threshold
    pub fn with_config(ttl_seconds: u64, invalidation_threshold: f64) -> Self {
        Self {
            cache: HashMap::new(),
            ttl: ttl_seconds,
            invalidation_threshold: invalidation_threshold.clamp(0.01, 0.5),
            event_log: std::collections::VecDeque::new(),
            max_event_log_size: 1000,
        }
    }

    /// Get a cached assessment if valid
    ///
    /// Returns None if:
    /// - No cached entry exists
    /// - Entry has expired (TTL)
    /// - Entry needs invalidation due to reputation change
    pub fn get(&self, agent_id: &str, current_reputation: f64) -> Option<&IdentityTrustAssessment> {
        let entry = self.cache.get(agent_id)?;
        let now = current_timestamp();

        // Check TTL
        if now - entry.cached_at >= self.ttl {
            return None;
        }

        // Check for significant reputation change (event-based invalidation)
        if (current_reputation - entry.last_reputation).abs() > self.invalidation_threshold {
            // Cache invalidated due to reputation change (FIND-009)
            return None;
        }

        Some(&entry.assessment)
    }

    /// Insert an assessment into the cache
    pub fn insert(
        &mut self,
        agent_id: impl Into<String>,
        assessment: IdentityTrustAssessment,
        reputation: f64,
    ) {
        let agent_id = agent_id.into();
        self.cache.insert(
            agent_id,
            CachedScore {
                assessment,
                last_reputation: reputation,
                cached_at: current_timestamp(),
            },
        );
    }

    /// Process a reputation change event and invalidate cache if needed
    ///
    /// Returns true if the cache was invalidated
    pub fn invalidate_on_reputation_change(&mut self, event: ReputationChangeEvent) -> bool {
        // Log the event
        self.event_log.push_back(event.clone());
        if self.event_log.len() > self.max_event_log_size {
            self.event_log.pop_front();
        }

        // Check if this is a significant change
        if event.is_significant(self.invalidation_threshold)
            && self.cache.remove(&event.agent_id).is_some()
        {
            // Cache invalidated due to significant reputation change (FIND-009)
            return true;
        }
        false
    }

    /// Invalidate cache entry for a specific agent
    pub fn invalidate(&mut self, agent_id: &str) {
        self.cache.remove(agent_id);
    }

    /// Clear all cached entries
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> TrustScoreCacheStats {
        TrustScoreCacheStats {
            entry_count: self.cache.len(),
            ttl_seconds: self.ttl,
            invalidation_threshold: self.invalidation_threshold,
            event_log_size: self.event_log.len(),
        }
    }

    /// Get recent reputation change events
    pub fn recent_events(&self, count: usize) -> Vec<&ReputationChangeEvent> {
        self.event_log.iter().rev().take(count).collect()
    }
}

/// Statistics about the trust score cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScoreCacheStats {
    /// Number of cached entries
    pub entry_count: usize,
    /// TTL in seconds
    pub ttl_seconds: u64,
    /// Invalidation threshold
    pub invalidation_threshold: f64,
    /// Number of events in the log
    pub event_log_size: usize,
}

// Add event-based invalidation to the coordinator
impl ByzantineIdentityCoordinator {
    /// Set the cache invalidation threshold
    ///
    /// If a reputation changes by more than this threshold, the cache
    /// entry is invalidated regardless of TTL.
    pub fn set_invalidation_threshold(&mut self, _threshold: f64) {
        // We track this in the coordinator and apply during evaluation
        // For now, we just invalidate aggressively on report_* methods
        // Future: integrate TrustScoreCache
    }

    /// Handle a reputation change event (FIND-009)
    ///
    /// This method should be called when a reputation score changes.
    /// It will invalidate the cache if the change is significant.
    pub fn on_reputation_change(
        &mut self,
        agent_id: &str,
        old_score: f64,
        new_score: f64,
        threshold: f64,
    ) {
        if (new_score - old_score).abs() > threshold {
            self.assessment_cache.remove(agent_id);
            // Cache invalidated due to significant reputation change (FIND-009)
        }
    }

    /// Create and process a reputation change event
    pub fn process_reputation_event(&mut self, event: ReputationChangeEvent, threshold: f64) {
        self.on_reputation_change(&event.agent_id, event.old_score, event.new_score, threshold);
    }
}

// =============================================================================
// Aggregated Reputation System
// =============================================================================

/// Source of reputation data
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReputationSource {
    /// Direct hApp interaction data
    HappInteraction(String),
    /// K-Vector trust dimensions
    KVectorDimension(String),
    /// MATL Byzantine detection
    MatlEvaluation,
    /// Credential verification status
    CredentialVerification,
    /// External attestation
    ExternalAttestation(String),
}

/// Single reputation data point from a source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationDataPoint {
    /// Source of this data
    pub source: ReputationSource,
    /// Score value (0.0-1.0)
    pub score: f64,
    /// Confidence in this score (0.0-1.0)
    pub confidence: f64,
    /// Number of interactions/samples underlying this score
    pub sample_count: u64,
    /// Weight to apply when aggregating
    pub weight: f64,
    /// When this data was recorded
    pub timestamp: u64,
}

impl ReputationDataPoint {
    /// Create a new data point
    pub fn new(source: ReputationSource, score: f64, confidence: f64) -> Self {
        Self {
            source,
            score: score.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            sample_count: 1,
            weight: 1.0,
            timestamp: current_timestamp(),
        }
    }

    /// Set the sample count
    pub fn with_sample_count(mut self, count: u64) -> Self {
        self.sample_count = count;
        self
    }

    /// Set the weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    /// Compute weighted contribution
    pub fn weighted_score(&self) -> f64 {
        self.score * self.weight * self.confidence
    }
}

/// Comprehensive aggregated reputation for an identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedReputation {
    /// The DID being evaluated
    pub did: String,
    /// Individual data points from all sources
    pub data_points: Vec<ReputationDataPoint>,
    /// K-Vector snapshot (if available)
    pub k_vector: Option<KVector>,
    /// Final aggregated score (0.0-1.0)
    pub aggregated_score: f64,
    /// Confidence in the aggregated score
    pub aggregated_confidence: f64,
    /// Total interaction count across all sources
    pub total_interactions: u64,
    /// Verification level from k_v dimension
    pub verification_level: f64,
    /// Byzantine risk score from MATL
    pub byzantine_risk: f64,
    /// Timestamp of aggregation
    pub aggregated_at: u64,
}

impl AggregatedReputation {
    /// Check if this is a well-established reputation
    pub fn is_established(&self) -> bool {
        self.total_interactions >= 10 && self.aggregated_confidence >= 0.5
    }

    /// Check if identity is verified
    pub fn is_verified(&self) -> bool {
        self.verification_level >= 0.5
    }

    /// Check if identity is strongly verified
    pub fn is_strongly_verified(&self) -> bool {
        self.verification_level >= 0.7
    }

    /// Check if identity has low Byzantine risk
    pub fn is_low_risk(&self) -> bool {
        self.byzantine_risk <= 0.2
    }

    /// Get trust classification
    pub fn trust_classification(&self) -> TrustLevel {
        if self.byzantine_risk > 0.5 {
            return TrustLevel::Untrusted;
        }
        if self.byzantine_risk > 0.3 || self.aggregated_score < 0.3 {
            return TrustLevel::Suspicious;
        }
        if !self.is_established() {
            return TrustLevel::Unknown;
        }
        if self.aggregated_score >= 0.8 && self.is_verified() {
            return TrustLevel::Trusted;
        }
        if self.aggregated_score >= 0.5 {
            return TrustLevel::Conditional;
        }
        TrustLevel::Suspicious
    }

    /// Convert to CrossHappReputation (for compatibility)
    pub fn to_cross_happ_reputation(&self) -> CrossHappReputation {
        let scores: Vec<HappReputationScore> = self
            .data_points
            .iter()
            .filter_map(|dp| {
                if let ReputationSource::HappInteraction(ref happ_id) = dp.source {
                    Some(HappReputationScore {
                        happ_id: happ_id.clone(),
                        happ_name: happ_id.clone(),
                        score: dp.score,
                        interactions: dp.sample_count,
                        last_updated: dp.timestamp,
                    })
                } else {
                    None
                }
            })
            .collect();

        CrossHappReputation::from_scores(&self.did, scores)
    }
}

/// Configuration for reputation aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Weight for hApp interaction scores
    pub happ_weight: f64,
    /// Weight for K-Vector dimensions
    pub kvector_weight: f64,
    /// Weight for MATL evaluation
    pub matl_weight: f64,
    /// Weight for credential verification
    pub credential_weight: f64,
    /// Weight for external attestations
    pub attestation_weight: f64,
    /// Minimum data points required
    pub min_data_points: usize,
    /// Time decay half-life in seconds
    pub decay_half_life: u64,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            happ_weight: 0.35,
            kvector_weight: 0.25,
            matl_weight: 0.20,
            credential_weight: 0.10,
            attestation_weight: 0.10,
            min_data_points: 1,
            decay_half_life: 86400 * 30, // 30 days
        }
    }
}

impl ByzantineIdentityCoordinator {
    /// Get aggregated reputation for a DID from all available sources
    ///
    /// This is the primary entry point for comprehensive reputation queries.
    /// It combines:
    /// - Cross-hApp reputation scores
    /// - K-Vector trust dimensions
    /// - MATL Byzantine detection results
    /// - Credential verification status
    ///
    /// # Arguments
    /// * `did` - The DID to query
    /// * `happ_scores` - Optional pre-fetched hApp scores
    /// * `k_vector` - Optional K-Vector for the identity
    /// * `config` - Aggregation configuration
    ///
    /// # Returns
    /// Comprehensive `AggregatedReputation` combining all factors
    pub fn get_aggregated_reputation(
        &mut self,
        did: &str,
        happ_scores: Option<Vec<HappReputationScore>>,
        k_vector: Option<KVector>,
        config: &AggregationConfig,
    ) -> AggregatedReputation {
        let now = current_timestamp();
        let mut data_points = Vec::new();
        let mut total_weight = 0.0;
        let mut weighted_score_sum = 0.0;
        let mut total_interactions: u64 = 0;

        // 1. Process hApp interaction scores
        if let Some(scores) = happ_scores {
            for score in scores {
                let weight = config.happ_weight;
                let confidence = Self::compute_confidence_from_interactions(score.interactions);
                let time_decay = Self::time_decay(now, score.last_updated, config.decay_half_life);

                let dp = ReputationDataPoint::new(
                    ReputationSource::HappInteraction(score.happ_id.clone()),
                    score.score,
                    confidence * time_decay,
                )
                .with_sample_count(score.interactions)
                .with_weight(weight);

                weighted_score_sum += dp.weighted_score();
                total_weight += weight * dp.confidence;
                total_interactions += score.interactions;
                data_points.push(dp);
            }
        }

        // 2. Process K-Vector dimensions
        let (verification_level, _kvector_score) = if let Some(ref kv) = k_vector {
            let trust_score = kv.trust_score() as f64;
            let verification = kv.k_v as f64;

            // Add K-Vector as a data point
            let dp = ReputationDataPoint::new(
                ReputationSource::KVectorDimension("trust_score".to_string()),
                trust_score,
                0.9, // High confidence in K-Vector
            )
            .with_weight(config.kvector_weight);

            weighted_score_sum += dp.weighted_score();
            total_weight += config.kvector_weight * 0.9;
            data_points.push(dp);

            // Add verification as a data point
            let vdp = ReputationDataPoint::new(
                ReputationSource::CredentialVerification,
                verification,
                if kv.is_strongly_verified() {
                    0.95
                } else if kv.is_verified() {
                    0.7
                } else {
                    0.3
                },
            )
            .with_weight(config.credential_weight);

            weighted_score_sum += vdp.weighted_score();
            total_weight += config.credential_weight * vdp.confidence;
            data_points.push(vdp);

            (verification, trust_score)
        } else {
            (0.0, 0.5) // Default
        };

        // 3. MATL evaluation (use cached if available)
        let byzantine_risk: f64 = if let Some(assessment) = self.assessment_cache.get(did) {
            let _matl_score = 1.0 - (assessment.matl_evaluation.composite_score.max(0.0));
            let risk: f64 = if assessment.matl_evaluation.is_anomalous {
                0.5
            } else {
                0.1
            } + if assessment.matl_evaluation.in_suspicious_cluster {
                0.4
            } else {
                0.0
            };

            // Add MATL as a data point (inverted - higher score = lower risk)
            let dp = ReputationDataPoint::new(
                ReputationSource::MatlEvaluation,
                1.0 - risk.min(1.0),
                0.85, // Good confidence in MATL
            )
            .with_weight(config.matl_weight);

            weighted_score_sum += dp.weighted_score();
            total_weight += config.matl_weight * 0.85;
            data_points.push(dp);

            risk.min(1.0)
        } else {
            // No MATL data - assume low risk
            0.1
        };

        // 4. Compute aggregated score
        let aggregated_score = if total_weight > 0.0 {
            weighted_score_sum / total_weight
        } else {
            0.5 // Default for no data
        };

        // 5. Compute aggregated confidence
        let aggregated_confidence = if data_points.is_empty() {
            0.0
        } else {
            let avg_confidence: f64 =
                data_points.iter().map(|dp| dp.confidence).sum::<f64>() / data_points.len() as f64;
            // Boost confidence with more data points
            let data_point_boost = (data_points.len() as f64 / 5.0).min(1.0);
            (avg_confidence * 0.7 + data_point_boost * 0.3).min(1.0)
        };

        AggregatedReputation {
            did: did.to_string(),
            data_points,
            k_vector,
            aggregated_score,
            aggregated_confidence,
            total_interactions,
            verification_level,
            byzantine_risk,
            aggregated_at: now,
        }
    }

    /// Compute confidence from interaction count (diminishing returns)
    fn compute_confidence_from_interactions(interactions: u64) -> f64 {
        // Confidence grows with sqrt of interactions, capped at 0.95
        (1.0 - 1.0 / (1.0 + (interactions as f64).sqrt() * 0.3)).min(0.95)
    }

    /// Compute time decay factor
    fn time_decay(now: u64, timestamp: u64, half_life: u64) -> f64 {
        let age = now.saturating_sub(timestamp);
        if half_life == 0 {
            return 1.0;
        }
        0.5_f64.powf(age as f64 / half_life as f64)
    }

    /// Quick aggregation with defaults
    pub fn get_aggregated_reputation_quick(
        &mut self,
        did: &str,
        reputation: &CrossHappReputation,
        k_vector: Option<KVector>,
    ) -> AggregatedReputation {
        self.get_aggregated_reputation(
            did,
            Some(reputation.scores.clone()),
            k_vector,
            &AggregationConfig::default(),
        )
    }

    /// Get aggregation config used by this coordinator
    pub fn aggregation_config(&self) -> AggregationConfig {
        AggregationConfig {
            happ_weight: 0.35,
            kvector_weight: 0.25,
            matl_weight: self.config.matl_weight,
            credential_weight: 0.10,
            attestation_weight: 0.10,
            min_data_points: 1,
            decay_half_life: 86400 * 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_reputation(score: f64, interactions: u64) -> CrossHappReputation {
        CrossHappReputation::from_scores(
            "test_agent",
            vec![HappReputationScore {
                happ_id: "test_happ".to_string(),
                happ_name: "Test hApp".to_string(),
                score,
                interactions,
                last_updated: 0,
            }],
        )
    }

    #[test]
    fn test_trust_level_classification() {
        let mut coordinator = ByzantineIdentityCoordinator::new();

        // High reputation -> Trusted
        let high_rep = mock_reputation(0.9, 100);
        let assessment = coordinator.evaluate_identity("did:mycelix:trusted", &high_rep);
        assert_eq!(assessment.trust_level, TrustLevel::Trusted);

        // Medium reputation -> Conditional
        let medium_rep = mock_reputation(0.6, 50);
        let assessment = coordinator.evaluate_identity("did:mycelix:conditional", &medium_rep);
        assert_eq!(assessment.trust_level, TrustLevel::Conditional);

        // Low reputation -> Suspicious
        let low_rep = mock_reputation(0.3, 50);
        let assessment = coordinator.evaluate_identity("did:mycelix:suspicious", &low_rep);
        assert_eq!(assessment.trust_level, TrustLevel::Suspicious);

        // Insufficient interactions -> Unknown
        let few_interactions = mock_reputation(0.9, 2);
        let assessment = coordinator.evaluate_identity("did:mycelix:unknown", &few_interactions);
        assert_eq!(assessment.trust_level, TrustLevel::Unknown);
    }

    #[test]
    fn test_trust_score_computation() {
        let mut coordinator = ByzantineIdentityCoordinator::new();

        let reputation = mock_reputation(0.8, 100);
        let assessment = coordinator.evaluate_identity("did:mycelix:test", &reputation);

        // Score should be reasonable
        assert!(assessment.trust_score > 0.5);
        assert!(assessment.trust_score <= 1.0);
    }

    #[test]
    fn test_caching() {
        let mut coordinator = ByzantineIdentityCoordinator::new();
        let reputation = mock_reputation(0.9, 100);

        // First evaluation
        let assessment1 = coordinator.evaluate_identity("did:mycelix:cached", &reputation);

        // Second evaluation should use cache
        let assessment2 = coordinator.evaluate_identity("did:mycelix:cached", &reputation);

        assert_eq!(assessment1.assessed_at, assessment2.assessed_at);
    }

    #[test]
    fn test_batch_evaluation() {
        let mut coordinator = ByzantineIdentityCoordinator::new();

        let identities = vec![
            ("did:mycelix:a".to_string(), mock_reputation(0.9, 100)),
            ("did:mycelix:b".to_string(), mock_reputation(0.5, 50)),
            ("did:mycelix:c".to_string(), mock_reputation(0.2, 30)),
        ];

        let assessments = coordinator.evaluate_batch(&identities);

        assert_eq!(assessments.len(), 3);
        assert_eq!(assessments[0].trust_level, TrustLevel::Trusted);
        assert_eq!(assessments[1].trust_level, TrustLevel::Conditional);
        // Low reputation can be Suspicious or Untrusted depending on MATL cluster detection
        assert!(
            assessments[2].trust_level == TrustLevel::Suspicious
                || assessments[2].trust_level == TrustLevel::Untrusted,
            "Expected Suspicious or Untrusted for low reputation, got {:?}",
            assessments[2].trust_level
        );
    }

    #[test]
    fn test_simple_trust_functions() {
        let high_rep = mock_reputation(0.9, 100);
        assert!(is_reputation_trustworthy(&high_rep, 0.8));

        let low_rep = mock_reputation(0.3, 100);
        assert!(!is_reputation_trustworthy(&low_rep, 0.8));

        let few_interactions = mock_reputation(0.9, 2);
        assert!(!is_reputation_trustworthy(&few_interactions, 0.8));
    }

    #[test]
    fn test_trust_level_permissions() {
        assert!(TrustLevel::Trusted.allows_high_stakes());
        assert!(TrustLevel::Trusted.allows_standard_ops());
        assert!(!TrustLevel::Trusted.requires_verification());

        assert!(!TrustLevel::Conditional.allows_high_stakes());
        assert!(TrustLevel::Conditional.allows_standard_ops());

        assert!(!TrustLevel::Suspicious.allows_standard_ops());
        assert!(TrustLevel::Suspicious.requires_verification());

        assert!(!TrustLevel::Untrusted.allows_standard_ops());
    }

    // =========================================================================
    // Aggregated Reputation Tests
    // =========================================================================

    #[test]
    fn test_aggregated_reputation_basic() {
        let mut coordinator = ByzantineIdentityCoordinator::new();
        let happ_scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: current_timestamp(),
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.8,
                interactions: 50,
                last_updated: current_timestamp(),
            },
        ];

        let agg = coordinator.get_aggregated_reputation(
            "did:mycelix:test",
            Some(happ_scores),
            None,
            &AggregationConfig::default(),
        );

        assert!(agg.aggregated_score > 0.5);
        assert!(agg.total_interactions == 150);
        assert!(agg.data_points.len() >= 2);
    }

    #[test]
    fn test_aggregated_reputation_with_kvector() {
        let mut coordinator = ByzantineIdentityCoordinator::new();
        let happ_scores = vec![HappReputationScore {
            happ_id: "test".to_string(),
            happ_name: "Test".to_string(),
            score: 0.7,
            interactions: 50,
            last_updated: current_timestamp(),
        }];

        // Create K-Vector with high verification
        let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.85, 0.7);

        let agg = coordinator.get_aggregated_reputation(
            "did:mycelix:verified",
            Some(happ_scores),
            Some(kv),
            &AggregationConfig::default(),
        );

        assert!(agg.is_verified());
        assert!(agg.is_strongly_verified());
        assert!(agg.verification_level >= 0.8);
        assert!(agg.k_vector.is_some());
    }

    #[test]
    fn test_aggregated_reputation_trust_classification() {
        let mut coordinator = ByzantineIdentityCoordinator::new();

        // High score with verification -> Trusted
        let kv_high = KVector::new(0.9, 0.8, 1.0, 0.9, 0.5, 0.7, 0.8, 0.6, 0.9, 0.85);
        let agg_high = coordinator.get_aggregated_reputation(
            "did:trusted",
            Some(vec![HappReputationScore {
                happ_id: "test".to_string(),
                happ_name: "Test".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: current_timestamp(),
            }]),
            Some(kv_high),
            &AggregationConfig::default(),
        );
        assert_eq!(agg_high.trust_classification(), TrustLevel::Trusted);

        // Low score -> Suspicious
        let kv_low = KVector::new(0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.0, 0.1);
        let agg_low = coordinator.get_aggregated_reputation(
            "did:suspicious",
            Some(vec![HappReputationScore {
                happ_id: "test".to_string(),
                happ_name: "Test".to_string(),
                score: 0.2,
                interactions: 100,
                last_updated: current_timestamp(),
            }]),
            Some(kv_low),
            &AggregationConfig::default(),
        );
        assert_eq!(agg_low.trust_classification(), TrustLevel::Suspicious);
    }

    #[test]
    fn test_aggregated_reputation_to_cross_happ() {
        let mut coordinator = ByzantineIdentityCoordinator::new();
        let happ_scores = vec![HappReputationScore {
            happ_id: "mail".to_string(),
            happ_name: "Mail".to_string(),
            score: 0.9,
            interactions: 100,
            last_updated: 0,
        }];

        let agg = coordinator.get_aggregated_reputation(
            "did:mycelix:test",
            Some(happ_scores),
            None,
            &AggregationConfig::default(),
        );

        let cross_happ = agg.to_cross_happ_reputation();
        assert_eq!(cross_happ.scores.len(), 1);
        assert!((cross_happ.aggregate - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_reputation_data_point() {
        let dp = ReputationDataPoint::new(
            ReputationSource::HappInteraction("test".to_string()),
            0.8,
            0.9,
        )
        .with_sample_count(100)
        .with_weight(0.5);

        assert_eq!(dp.sample_count, 100);
        assert!((dp.weight - 0.5).abs() < 0.001);
        assert!((dp.weighted_score() - 0.36).abs() < 0.01); // 0.8 * 0.5 * 0.9 = 0.36
    }

    #[test]
    fn test_quick_aggregation() {
        let mut coordinator = ByzantineIdentityCoordinator::new();
        let reputation = mock_reputation(0.85, 75);
        let kv = KVector::new_verified_participant(0.7);

        let agg = coordinator.get_aggregated_reputation_quick("did:quick", &reputation, Some(kv));

        assert!(agg.aggregated_score > 0.5);
        assert!(agg.is_verified());
    }
}
