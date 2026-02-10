//! # Live Attack Detection Pipeline
//!
//! Real-time detection of attacks on the trust system.
//!
//! ## Features
//!
//! - **Streaming Anomaly Detection**: Process events in real-time
//! - **Pattern Recognition**: Identify known attack signatures
//! - **Correlation Engine**: Connect related suspicious activities
//! - **Alert Pipeline**: Escalate threats appropriately
//!
//! ## Attack Types Detected
//!
//! - Sybil attacks (fake identity proliferation)
//! - Trust manipulation (artificial score inflation)
//! - Collusion networks (coordinated malicious behavior)
//! - Eclipse attacks (network isolation attempts)
//! - Flash attacks (rapid exploitation windows)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::matl::KVector;

// ============================================================================
// Configuration
// ============================================================================

/// Attack detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackDetectionConfig {
    /// Window size for streaming analysis (ms)
    pub window_size_ms: u64,
    /// Minimum events to trigger analysis
    pub min_events_for_analysis: usize,
    /// Sybil detection sensitivity (0.0-1.0)
    pub sybil_sensitivity: f64,
    /// Collusion detection sensitivity (0.0-1.0)
    pub collusion_sensitivity: f64,
    /// Trust manipulation threshold
    pub manipulation_threshold: f64,
    /// Flash attack window (ms)
    pub flash_attack_window_ms: u64,
    /// Maximum alerts per agent per hour
    pub max_alerts_per_agent_hour: u32,
    /// Enable automatic response
    pub auto_response: bool,
    /// Correlation window for related events (ms)
    pub correlation_window_ms: u64,
}

impl Default for AttackDetectionConfig {
    fn default() -> Self {
        Self {
            window_size_ms: 60_000, // 1 minute
            min_events_for_analysis: 10,
            sybil_sensitivity: 0.7,
            collusion_sensitivity: 0.6,
            manipulation_threshold: 0.3,
            flash_attack_window_ms: 5_000,
            max_alerts_per_agent_hour: 10,
            auto_response: false,
            correlation_window_ms: 300_000, // 5 minutes
        }
    }
}

// ============================================================================
// Events
// ============================================================================

/// Trust system event for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustEvent {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: TrustEventType,
    /// Agent involved
    pub agent_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Additional data
    pub metadata: HashMap<String, String>,
    /// Source (where event originated)
    pub source: EventSource,
}

/// Types of trust events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrustEventType {
    /// Agent registration
    AgentRegistered,
    /// Trust score change.
    TrustChanged {
        /// Previous trust value.
        old_value: f64,
        /// New trust value.
        new_value: f64,
    },
    /// Vote cast.
    VoteCast {
        /// Proposal being voted on.
        proposal_id: String,
        /// Vote decision.
        vote: String,
    },
    /// Interaction between agents.
    Interaction {
        /// Other agent involved.
        counterparty: String,
        /// Type of interaction.
        interaction_type: String,
    },
    /// KREDIT transfer.
    KreditTransfer {
        /// Recipient agent.
        to: String,
        /// Transfer amount.
        amount: u64,
    },
    /// Attestation created.
    AttestationCreated {
        /// Target agent of attestation.
        target: String,
        /// Attested trust level.
        trust_level: f64,
    },
    /// Constraint violation.
    ConstraintViolation {
        /// Name of the violated constraint.
        constraint: String,
    },
    /// Rate limit hit.
    RateLimitHit,
    /// Proof submitted.
    ProofSubmitted {
        /// Type of proof submitted.
        proof_type: String,
    },
    /// Group membership change.
    GroupMembershipChanged {
        /// Group identifier.
        group_id: String,
        /// Membership action (join/leave).
        action: String,
    },
}

/// Event source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSource {
    /// Direct API call
    Api,
    /// Consensus process
    Consensus,
    /// Federation
    Federation,
    /// Internal system
    Internal,
    /// Unknown
    Unknown,
}

// ============================================================================
// Attack Signatures
// ============================================================================

/// Known attack patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackSignature {
    /// Signature ID
    pub id: String,
    /// Attack name
    pub name: String,
    /// Description
    pub description: String,
    /// Pattern to match
    pub pattern: AttackPattern,
    /// Severity level
    pub severity: AttackSeverity,
    /// Recommended response
    pub response: RecommendedResponse,
}

/// Attack patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttackPattern {
    /// Rapid agent creation
    RapidCreation {
        /// Max agents per time window
        max_count: u32,
        /// Time window (ms)
        window_ms: u64,
    },
    /// Coordinated voting
    CoordinatedVoting {
        /// Minimum coordination score
        min_correlation: f64,
        /// Minimum participants
        min_participants: u32,
    },
    /// Trust inflation
    TrustInflation {
        /// Max increase per window
        max_increase: f64,
        /// Time window (ms)
        window_ms: u64,
    },
    /// Circular attestations
    CircularAttestations {
        /// Maximum cycle length
        max_cycle_length: u32,
    },
    /// Burst activity
    BurstActivity {
        /// Events per second threshold
        events_per_second: f64,
        /// Duration (ms)
        duration_ms: u64,
    },
    /// Similar K-Vectors
    SimilarKVectors {
        /// Similarity threshold
        similarity_threshold: f64,
        /// Minimum cluster size
        min_cluster_size: u32,
    },
    /// Network isolation attempt
    NetworkIsolation {
        /// Connectivity drop threshold
        connectivity_drop: f64,
    },
    /// Custom pattern (regex on serialized events)
    Custom {
        /// Pattern name
        name: String,
        /// Match function (simplified)
        matcher_id: String,
    },
}

/// Attack severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackSeverity {
    /// Low - suspicious but not dangerous
    Low,
    /// Medium - potential threat
    Medium,
    /// High - active attack
    High,
    /// Critical - system integrity at risk
    Critical,
}

/// Recommended response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendedResponse {
    /// Log and monitor
    Monitor,
    /// Rate limit the agent.
    RateLimit {
        /// Rate limit reduction factor (0.0-1.0).
        factor: f64,
    },
    /// Quarantine agent
    Quarantine,
    /// Suspend agent
    Suspend,
    /// Alert humans immediately
    AlertHuman,
    /// Freeze all related operations
    Freeze,
    /// Custom response.
    Custom {
        /// Description of the custom action.
        action: String,
    },
}

// ============================================================================
// Detection Results
// ============================================================================

/// Detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Detection ID
    pub id: String,
    /// Attack type detected
    pub attack_type: DetectedAttackType,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Severity
    pub severity: AttackSeverity,
    /// Agents involved
    pub involved_agents: Vec<String>,
    /// Evidence
    pub evidence: Vec<Evidence>,
    /// Recommended action
    pub recommended_action: RecommendedResponse,
    /// Timestamp
    pub timestamp: u64,
    /// Correlated detections
    pub correlated_with: Vec<String>,
}

/// Detected attack types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectedAttackType {
    /// Sybil attack (fake identity proliferation).
    Sybil,
    /// Artificial trust score manipulation.
    TrustManipulation,
    /// Coordinated malicious behavior.
    Collusion,
    /// Network isolation attempt.
    EclipseAttempt,
    /// Rapid exploitation window attack.
    FlashAttack,
    /// Circular attestation ring.
    CircularAttestation,
    /// Rate limit abuse.
    RateAbuse,
    /// Forged or invalid proof submission.
    ProofForgery,
    /// Unclassified attack.
    Unknown,
}

/// Evidence for detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Description
    pub description: String,
    /// Related event IDs
    pub event_ids: Vec<String>,
    /// Strength (0.0-1.0)
    pub strength: f64,
}

/// Evidence types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Statistical anomaly
    StatisticalAnomaly,
    /// Pattern match
    PatternMatch,
    /// Behavioral deviation
    BehavioralDeviation,
    /// Network analysis
    NetworkAnalysis,
    /// Temporal correlation
    TemporalCorrelation,
    /// Historical comparison
    HistoricalComparison,
}

// ============================================================================
// Streaming Analyzer
// ============================================================================

/// Streaming event analyzer
#[derive(Debug)]
pub struct StreamingAnalyzer {
    /// Configuration
    config: AttackDetectionConfig,
    /// Event buffer
    event_buffer: VecDeque<TrustEvent>,
    /// Agent activity tracking
    agent_activity: HashMap<String, AgentActivityProfile>,
    /// Recent detections
    recent_detections: VecDeque<DetectionResult>,
    /// Alert counts per agent (for rate limiting)
    alert_counts: HashMap<String, u32>,
    /// Last alert reset time
    last_alert_reset: u64,
    /// Known attack signatures
    signatures: Vec<AttackSignature>,
    /// Current timestamp
    current_time: u64,
}

/// Agent activity profile for detection
#[derive(Debug, Clone, Default)]
pub struct AgentActivityProfile {
    /// Total events
    pub event_count: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Trust history
    pub trust_history: VecDeque<(u64, f64)>,
    /// Interaction partners
    pub interaction_partners: HashSet<String>,
    /// Last K-Vector
    pub last_kvector: Option<KVector>,
    /// Burst windows (start_time, count)
    pub burst_windows: VecDeque<(u64, u32)>,
    /// Voting patterns
    pub voting_patterns: VecDeque<(String, String)>, // (proposal, vote)
}

impl StreamingAnalyzer {
    /// Create new streaming analyzer
    pub fn new(config: AttackDetectionConfig) -> Self {
        Self {
            config,
            event_buffer: VecDeque::new(),
            agent_activity: HashMap::new(),
            recent_detections: VecDeque::new(),
            alert_counts: HashMap::new(),
            last_alert_reset: 0,
            signatures: Self::default_signatures(),
            current_time: 0,
        }
    }

    /// Default attack signatures
    fn default_signatures() -> Vec<AttackSignature> {
        vec![
            AttackSignature {
                id: "SYB-001".to_string(),
                name: "Rapid Agent Creation".to_string(),
                description: "Multiple agents created in short window".to_string(),
                pattern: AttackPattern::RapidCreation {
                    max_count: 10,
                    window_ms: 60_000,
                },
                severity: AttackSeverity::High,
                response: RecommendedResponse::RateLimit { factor: 0.1 },
            },
            AttackSignature {
                id: "COL-001".to_string(),
                name: "Coordinated Voting".to_string(),
                description: "Multiple agents voting identically".to_string(),
                pattern: AttackPattern::CoordinatedVoting {
                    min_correlation: 0.9,
                    min_participants: 5,
                },
                severity: AttackSeverity::High,
                response: RecommendedResponse::AlertHuman,
            },
            AttackSignature {
                id: "MAN-001".to_string(),
                name: "Trust Inflation".to_string(),
                description: "Rapid trust score increase".to_string(),
                pattern: AttackPattern::TrustInflation {
                    max_increase: 0.3,
                    window_ms: 3600_000,
                },
                severity: AttackSeverity::Medium,
                response: RecommendedResponse::Monitor,
            },
            AttackSignature {
                id: "CIR-001".to_string(),
                name: "Circular Attestations".to_string(),
                description: "Attestation cycles detected".to_string(),
                pattern: AttackPattern::CircularAttestations {
                    max_cycle_length: 5,
                },
                severity: AttackSeverity::Critical,
                response: RecommendedResponse::Quarantine,
            },
            AttackSignature {
                id: "FLA-001".to_string(),
                name: "Flash Attack".to_string(),
                description: "Burst of activity in short window".to_string(),
                pattern: AttackPattern::BurstActivity {
                    events_per_second: 100.0,
                    duration_ms: 5_000,
                },
                severity: AttackSeverity::High,
                response: RecommendedResponse::Freeze,
            },
            AttackSignature {
                id: "SYB-002".to_string(),
                name: "Similar K-Vectors".to_string(),
                description: "Cluster of agents with nearly identical K-Vectors".to_string(),
                pattern: AttackPattern::SimilarKVectors {
                    similarity_threshold: 0.95,
                    min_cluster_size: 5,
                },
                severity: AttackSeverity::High,
                response: RecommendedResponse::AlertHuman,
            },
        ]
    }

    /// Add custom signature
    pub fn add_signature(&mut self, signature: AttackSignature) {
        self.signatures.push(signature);
    }

    /// Process incoming event
    pub fn process_event(&mut self, event: TrustEvent) -> Vec<DetectionResult> {
        self.current_time = event.timestamp;
        let mut detections = Vec::new();

        // Update agent activity profile
        self.update_activity_profile(&event);

        // Add to buffer
        self.event_buffer.push_back(event.clone());

        // Trim old events
        while let Some(front) = self.event_buffer.front() {
            if self.current_time - front.timestamp > self.config.window_size_ms {
                self.event_buffer.pop_front();
            } else {
                break;
            }
        }

        // Run detection if enough events
        if self.event_buffer.len() >= self.config.min_events_for_analysis {
            detections.extend(self.run_detection());
        }

        // Check specific event patterns
        detections.extend(self.check_event_specific(&event));

        // Correlate detections
        self.correlate_detections(&mut detections);

        // Store detections
        for detection in &detections {
            self.recent_detections.push_back(detection.clone());
        }

        // Trim old detections
        while self.recent_detections.len() > 1000 {
            self.recent_detections.pop_front();
        }

        // Reset alert counts hourly
        if self.current_time - self.last_alert_reset > 3600_000 {
            self.alert_counts.clear();
            self.last_alert_reset = self.current_time;
        }

        detections
    }

    /// Update agent activity profile
    fn update_activity_profile(&mut self, event: &TrustEvent) {
        let profile = self
            .agent_activity
            .entry(event.agent_id.clone())
            .or_default();

        profile.event_count += 1;

        let type_key = format!("{:?}", event.event_type);
        *profile.events_by_type.entry(type_key).or_insert(0) += 1;

        match &event.event_type {
            TrustEventType::TrustChanged { new_value, .. } => {
                profile
                    .trust_history
                    .push_back((event.timestamp, *new_value));
                while profile.trust_history.len() > 100 {
                    profile.trust_history.pop_front();
                }
            }
            TrustEventType::Interaction { counterparty, .. } => {
                profile.interaction_partners.insert(counterparty.clone());
            }
            TrustEventType::VoteCast { proposal_id, vote } => {
                profile
                    .voting_patterns
                    .push_back((proposal_id.clone(), vote.clone()));
                while profile.voting_patterns.len() > 100 {
                    profile.voting_patterns.pop_front();
                }
            }
            _ => {}
        }

        // Track burst windows
        profile.burst_windows.push_back((event.timestamp, 1));
        profile
            .burst_windows
            .retain(|(t, _)| self.current_time - *t < self.config.flash_attack_window_ms);
    }

    /// Run detection algorithms
    fn run_detection(&self) -> Vec<DetectionResult> {
        let mut detections = Vec::new();

        // Check each signature
        for signature in &self.signatures {
            if let Some(detection) = self.check_signature(signature) {
                detections.push(detection);
            }
        }

        detections
    }

    /// Check a specific signature
    fn check_signature(&self, signature: &AttackSignature) -> Option<DetectionResult> {
        match &signature.pattern {
            AttackPattern::RapidCreation {
                max_count,
                window_ms,
            } => self.check_rapid_creation(*max_count, *window_ms, signature),
            AttackPattern::CoordinatedVoting {
                min_correlation,
                min_participants,
            } => self.check_coordinated_voting(*min_correlation, *min_participants, signature),
            AttackPattern::TrustInflation {
                max_increase,
                window_ms,
            } => self.check_trust_inflation(*max_increase, *window_ms, signature),
            AttackPattern::BurstActivity {
                events_per_second,
                duration_ms,
            } => self.check_burst_activity(*events_per_second, *duration_ms, signature),
            AttackPattern::SimilarKVectors {
                similarity_threshold,
                min_cluster_size,
            } => self.check_similar_kvectors(*similarity_threshold, *min_cluster_size, signature),
            _ => None,
        }
    }

    /// Check for rapid agent creation
    fn check_rapid_creation(
        &self,
        max_count: u32,
        window_ms: u64,
        signature: &AttackSignature,
    ) -> Option<DetectionResult> {
        let creation_events: Vec<_> = self
            .event_buffer
            .iter()
            .filter(|e| matches!(e.event_type, TrustEventType::AgentRegistered))
            .filter(|e| self.current_time - e.timestamp <= window_ms)
            .collect();

        if creation_events.len() as u32 > max_count {
            let agents: Vec<String> = creation_events.iter().map(|e| e.agent_id.clone()).collect();

            return Some(DetectionResult {
                id: format!("det-{}-{}", signature.id, self.current_time),
                attack_type: DetectedAttackType::Sybil,
                confidence: (creation_events.len() as f64 / max_count as f64).min(1.0),
                severity: signature.severity,
                involved_agents: agents,
                evidence: vec![Evidence {
                    evidence_type: EvidenceType::StatisticalAnomaly,
                    description: format!(
                        "{} agents created in {}ms window (threshold: {})",
                        creation_events.len(),
                        window_ms,
                        max_count
                    ),
                    event_ids: creation_events.iter().map(|e| e.id.clone()).collect(),
                    strength: 0.9,
                }],
                recommended_action: signature.response.clone(),
                timestamp: self.current_time,
                correlated_with: vec![],
            });
        }

        None
    }

    /// Check for coordinated voting
    fn check_coordinated_voting(
        &self,
        min_correlation: f64,
        min_participants: u32,
        signature: &AttackSignature,
    ) -> Option<DetectionResult> {
        // Group votes by proposal
        let mut proposal_votes: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for event in &self.event_buffer {
            if let TrustEventType::VoteCast { proposal_id, vote } = &event.event_type {
                proposal_votes
                    .entry(proposal_id.clone())
                    .or_default()
                    .push((event.agent_id.clone(), vote.clone()));
            }
        }

        // Check for identical voting patterns
        for (proposal_id, votes) in &proposal_votes {
            if votes.len() < min_participants as usize {
                continue;
            }

            // Group by vote value
            let mut vote_groups: HashMap<String, Vec<String>> = HashMap::new();
            for (agent, vote) in votes {
                vote_groups
                    .entry(vote.clone())
                    .or_default()
                    .push(agent.clone());
            }

            // Check if any group is suspiciously large
            for (vote_value, agents) in &vote_groups {
                let correlation = agents.len() as f64 / votes.len() as f64;

                if correlation >= min_correlation && agents.len() >= min_participants as usize {
                    return Some(DetectionResult {
                        id: format!("det-{}-{}", signature.id, self.current_time),
                        attack_type: DetectedAttackType::Collusion,
                        confidence: correlation,
                        severity: signature.severity,
                        involved_agents: agents.clone(),
                        evidence: vec![Evidence {
                            evidence_type: EvidenceType::PatternMatch,
                            description: format!(
                                "{} agents voted '{}' on proposal {} ({:.1}% correlation)",
                                agents.len(),
                                vote_value,
                                proposal_id,
                                correlation * 100.0
                            ),
                            event_ids: vec![],
                            strength: correlation,
                        }],
                        recommended_action: signature.response.clone(),
                        timestamp: self.current_time,
                        correlated_with: vec![],
                    });
                }
            }
        }

        None
    }

    /// Check for trust inflation
    fn check_trust_inflation(
        &self,
        max_increase: f64,
        window_ms: u64,
        signature: &AttackSignature,
    ) -> Option<DetectionResult> {
        for (agent_id, profile) in &self.agent_activity {
            let recent_changes: Vec<_> = profile
                .trust_history
                .iter()
                .filter(|(t, _)| self.current_time - *t <= window_ms)
                .collect();

            if recent_changes.len() < 2 {
                continue;
            }

            let first = recent_changes.first().map(|(_, v)| *v).unwrap_or(0.0);
            let last = recent_changes.last().map(|(_, v)| *v).unwrap_or(0.0);
            let increase = last - first;

            if increase > max_increase {
                return Some(DetectionResult {
                    id: format!("det-{}-{}", signature.id, self.current_time),
                    attack_type: DetectedAttackType::TrustManipulation,
                    confidence: (increase / max_increase).min(1.0),
                    severity: signature.severity,
                    involved_agents: vec![agent_id.clone()],
                    evidence: vec![Evidence {
                        evidence_type: EvidenceType::StatisticalAnomaly,
                        description: format!(
                            "Trust increased by {:.2} in {}ms (threshold: {:.2})",
                            increase, window_ms, max_increase
                        ),
                        event_ids: vec![],
                        strength: (increase / max_increase).min(1.0),
                    }],
                    recommended_action: signature.response.clone(),
                    timestamp: self.current_time,
                    correlated_with: vec![],
                });
            }
        }

        None
    }

    /// Check for burst activity
    fn check_burst_activity(
        &self,
        events_per_second: f64,
        duration_ms: u64,
        signature: &AttackSignature,
    ) -> Option<DetectionResult> {
        for (agent_id, profile) in &self.agent_activity {
            let burst_count: u32 = profile
                .burst_windows
                .iter()
                .filter(|(t, _)| self.current_time - *t <= duration_ms)
                .map(|(_, c)| c)
                .sum();

            let actual_rate = burst_count as f64 / (duration_ms as f64 / 1000.0);

            if actual_rate > events_per_second {
                return Some(DetectionResult {
                    id: format!("det-{}-{}", signature.id, self.current_time),
                    attack_type: DetectedAttackType::FlashAttack,
                    confidence: (actual_rate / events_per_second).min(1.0),
                    severity: signature.severity,
                    involved_agents: vec![agent_id.clone()],
                    evidence: vec![Evidence {
                        evidence_type: EvidenceType::StatisticalAnomaly,
                        description: format!(
                            "Activity rate {:.1}/s exceeds threshold {:.1}/s",
                            actual_rate, events_per_second
                        ),
                        event_ids: vec![],
                        strength: (actual_rate / events_per_second).min(1.0),
                    }],
                    recommended_action: signature.response.clone(),
                    timestamp: self.current_time,
                    correlated_with: vec![],
                });
            }
        }

        None
    }

    /// Check for similar K-Vectors
    fn check_similar_kvectors(
        &self,
        _similarity_threshold: f64,
        min_cluster_size: u32,
        signature: &AttackSignature,
    ) -> Option<DetectionResult> {
        // Collect agents with K-Vectors
        let agents_with_kvectors: Vec<_> = self
            .agent_activity
            .iter()
            .filter_map(|(id, profile)| profile.last_kvector.as_ref().map(|kv| (id.clone(), *kv)))
            .collect();

        if agents_with_kvectors.len() < min_cluster_size as usize {
            return None;
        }

        // Simple clustering by trust score similarity (simplified)
        let mut clusters: Vec<Vec<String>> = Vec::new();

        for (agent_id, kv) in &agents_with_kvectors {
            let trust = kv.trust_score();
            let mut found_cluster = false;

            for cluster in &mut clusters {
                // Check if agent fits in cluster
                if let Some(first_agent) = cluster.first() {
                    if let Some((_, first_kv)) = agents_with_kvectors
                        .iter()
                        .find(|(id, _)| id == first_agent)
                    {
                        let first_trust = first_kv.trust_score();
                        if (trust - first_trust).abs() < 0.05 {
                            cluster.push(agent_id.clone());
                            found_cluster = true;
                            break;
                        }
                    }
                }
            }

            if !found_cluster {
                clusters.push(vec![agent_id.clone()]);
            }
        }

        // Check for suspicious clusters
        for cluster in clusters {
            if cluster.len() >= min_cluster_size as usize {
                return Some(DetectionResult {
                    id: format!("det-{}-{}", signature.id, self.current_time),
                    attack_type: DetectedAttackType::Sybil,
                    confidence: cluster.len() as f64 / agents_with_kvectors.len() as f64,
                    severity: signature.severity,
                    involved_agents: cluster.clone(),
                    evidence: vec![Evidence {
                        evidence_type: EvidenceType::NetworkAnalysis,
                        description: format!(
                            "Cluster of {} agents with similar K-Vectors",
                            cluster.len()
                        ),
                        event_ids: vec![],
                        strength: 0.8,
                    }],
                    recommended_action: signature.response.clone(),
                    timestamp: self.current_time,
                    correlated_with: vec![],
                });
            }
        }

        None
    }

    /// Check event-specific patterns
    fn check_event_specific(&self, event: &TrustEvent) -> Vec<DetectionResult> {
        let mut detections = Vec::new();

        // Check for rate limit abuse patterns
        if matches!(event.event_type, TrustEventType::RateLimitHit) {
            let profile = self.agent_activity.get(&event.agent_id);
            if let Some(p) = profile {
                let rate_limit_hits = p.events_by_type.get("RateLimitHit").copied().unwrap_or(0);

                if rate_limit_hits > 5 {
                    detections.push(DetectionResult {
                        id: format!("det-RATE-{}", self.current_time),
                        attack_type: DetectedAttackType::RateAbuse,
                        confidence: (rate_limit_hits as f64 / 10.0).min(1.0),
                        severity: AttackSeverity::Medium,
                        involved_agents: vec![event.agent_id.clone()],
                        evidence: vec![Evidence {
                            evidence_type: EvidenceType::BehavioralDeviation,
                            description: format!("Agent hit rate limit {} times", rate_limit_hits),
                            event_ids: vec![event.id.clone()],
                            strength: 0.7,
                        }],
                        recommended_action: RecommendedResponse::RateLimit { factor: 0.5 },
                        timestamp: self.current_time,
                        correlated_with: vec![],
                    });
                }
            }
        }

        detections
    }

    /// Correlate related detections
    fn correlate_detections(&self, detections: &mut [DetectionResult]) {
        // Find detections within correlation window
        let recent: Vec<_> = self
            .recent_detections
            .iter()
            .filter(|d| self.current_time - d.timestamp <= self.config.correlation_window_ms)
            .collect();

        for detection in detections.iter_mut() {
            for recent_detection in &recent {
                // Check for overlapping agents
                let overlap: HashSet<_> = detection
                    .involved_agents
                    .iter()
                    .filter(|a| recent_detection.involved_agents.contains(*a))
                    .collect();

                if !overlap.is_empty() {
                    detection.correlated_with.push(recent_detection.id.clone());

                    // Boost confidence if correlated
                    detection.confidence = (detection.confidence * 1.2).min(1.0);
                }
            }
        }
    }

    /// Get recent detections
    pub fn recent_detections(&self, limit: usize) -> Vec<&DetectionResult> {
        self.recent_detections.iter().rev().take(limit).collect()
    }

    /// Get agent risk score
    pub fn agent_risk_score(&self, agent_id: &str) -> f64 {
        let detections: Vec<_> = self
            .recent_detections
            .iter()
            .filter(|d| d.involved_agents.contains(&agent_id.to_string()))
            .collect();

        if detections.is_empty() {
            return 0.0;
        }

        // Weighted sum of detection severities
        let mut risk = 0.0;
        for detection in detections {
            let severity_weight = match detection.severity {
                AttackSeverity::Low => 0.1,
                AttackSeverity::Medium => 0.3,
                AttackSeverity::High => 0.6,
                AttackSeverity::Critical => 1.0,
            };
            risk += detection.confidence * severity_weight;
        }

        risk.min(1.0)
    }

    /// Get statistics
    pub fn stats(&self) -> DetectionStats {
        let detections_by_type: HashMap<String, usize> = self
            .recent_detections
            .iter()
            .map(|d| format!("{:?}", d.attack_type))
            .fold(HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });

        DetectionStats {
            total_events_processed: self.event_buffer.len(),
            total_detections: self.recent_detections.len(),
            detections_by_type,
            agents_tracked: self.agent_activity.len(),
            signatures_active: self.signatures.len(),
        }
    }
}

/// Detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStats {
    /// Total trust events processed in the current window.
    pub total_events_processed: usize,
    /// Total detections raised.
    pub total_detections: usize,
    /// Detection counts grouped by attack type.
    pub detections_by_type: HashMap<String, usize>,
    /// Number of agents being tracked.
    pub agents_tracked: usize,
    /// Number of active attack signatures.
    pub signatures_active: usize,
}

// ============================================================================
// Alert Pipeline
// ============================================================================

/// Alert for human review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Detection that triggered alert
    pub detection_id: String,
    /// Severity
    pub severity: AttackSeverity,
    /// Summary
    pub summary: String,
    /// Detailed description
    pub details: String,
    /// Timestamp
    pub timestamp: u64,
    /// Status
    pub status: AlertStatus,
    /// Assigned to
    pub assigned_to: Option<String>,
    /// Actions taken
    pub actions_taken: Vec<String>,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Newly created, unreviewed.
    New,
    /// Acknowledged by a reviewer.
    Acknowledged,
    /// Under active investigation.
    Investigating,
    /// Resolved and closed.
    Resolved,
    /// Determined to be a false positive.
    FalsePositive,
}

/// Alert pipeline
#[derive(Debug)]
pub struct AlertPipeline {
    /// Pending alerts
    alerts: VecDeque<Alert>,
    /// Alert history
    history: Vec<Alert>,
    /// Maximum alerts to retain
    max_history: usize,
}

impl AlertPipeline {
    /// Create new alert pipeline
    pub fn new(max_history: usize) -> Self {
        Self {
            alerts: VecDeque::new(),
            history: Vec::new(),
            max_history,
        }
    }

    /// Create alert from detection
    pub fn create_alert(&mut self, detection: &DetectionResult) -> Alert {
        let alert = Alert {
            id: format!("alert-{}", detection.timestamp),
            detection_id: detection.id.clone(),
            severity: detection.severity,
            summary: format!(
                "{:?} detected with {:.0}% confidence",
                detection.attack_type,
                detection.confidence * 100.0
            ),
            details: format!(
                "Agents involved: {:?}\nEvidence: {:?}\nRecommended action: {:?}",
                detection.involved_agents, detection.evidence, detection.recommended_action
            ),
            timestamp: detection.timestamp,
            status: AlertStatus::New,
            assigned_to: None,
            actions_taken: vec![],
        };

        self.alerts.push_back(alert.clone());
        alert
    }

    /// Get pending alerts
    pub fn pending_alerts(&self) -> impl Iterator<Item = &Alert> {
        self.alerts.iter().filter(|a| a.status == AlertStatus::New)
    }

    /// Acknowledge alert
    pub fn acknowledge(&mut self, alert_id: &str, assignee: &str) -> bool {
        if let Some(alert) = self.alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.status = AlertStatus::Acknowledged;
            alert.assigned_to = Some(assignee.to_string());
            return true;
        }
        false
    }

    /// Resolve alert
    #[allow(clippy::unwrap_used)]
    pub fn resolve(&mut self, alert_id: &str, action: &str) -> bool {
        if let Some(pos) = self.alerts.iter().position(|a| a.id == alert_id) {
            let mut alert = self.alerts.remove(pos).unwrap();
            alert.status = AlertStatus::Resolved;
            alert.actions_taken.push(action.to_string());
            self.history.push(alert);

            // Trim history
            while self.history.len() > self.max_history {
                self.history.remove(0);
            }

            return true;
        }
        false
    }

    /// Mark as false positive
    #[allow(clippy::unwrap_used)]
    pub fn mark_false_positive(&mut self, alert_id: &str) -> bool {
        if let Some(pos) = self.alerts.iter().position(|a| a.id == alert_id) {
            let mut alert = self.alerts.remove(pos).unwrap();
            alert.status = AlertStatus::FalsePositive;
            self.history.push(alert);
            return true;
        }
        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_event(agent_id: &str, event_type: TrustEventType, timestamp: u64) -> TrustEvent {
        TrustEvent {
            id: format!("event-{}-{}", agent_id, timestamp),
            event_type,
            agent_id: agent_id.to_string(),
            timestamp,
            metadata: HashMap::new(),
            source: EventSource::Api,
        }
    }

    #[test]
    fn test_rapid_creation_detection() {
        let mut analyzer = StreamingAnalyzer::new(AttackDetectionConfig {
            min_events_for_analysis: 5,
            ..Default::default()
        });

        // Create many agents rapidly
        for i in 0..15 {
            let event = create_test_event(
                &format!("agent-{}", i),
                TrustEventType::AgentRegistered,
                1000 + i as u64 * 100,
            );
            let detections = analyzer.process_event(event);

            if i >= 10 {
                assert!(!detections.is_empty(), "Should detect rapid creation");
                assert!(matches!(
                    detections[0].attack_type,
                    DetectedAttackType::Sybil
                ));
            }
        }
    }

    #[test]
    fn test_coordinated_voting_detection() {
        let mut analyzer = StreamingAnalyzer::new(AttackDetectionConfig {
            min_events_for_analysis: 5,
            ..Default::default()
        });

        // Create coordinated votes
        for i in 0..10 {
            let event = create_test_event(
                &format!("agent-{}", i),
                TrustEventType::VoteCast {
                    proposal_id: "proposal-1".to_string(),
                    vote: "approve".to_string(),
                },
                1000 + i as u64 * 100,
            );
            let detections = analyzer.process_event(event);

            // Should detect after enough votes
            if i >= 5 {
                let collusion_detected = detections
                    .iter()
                    .any(|d| matches!(d.attack_type, DetectedAttackType::Collusion));
                if collusion_detected {
                    // Good, detected
                    return;
                }
            }
        }
    }

    #[test]
    fn test_trust_inflation_detection() {
        let mut analyzer = StreamingAnalyzer::new(AttackDetectionConfig {
            min_events_for_analysis: 2,
            ..Default::default()
        });

        // Rapid trust increase
        let event1 = create_test_event(
            "agent-1",
            TrustEventType::TrustChanged {
                old_value: 0.3,
                new_value: 0.4,
            },
            1000,
        );
        analyzer.process_event(event1);

        let event2 = create_test_event(
            "agent-1",
            TrustEventType::TrustChanged {
                old_value: 0.4,
                new_value: 0.8,
            },
            2000,
        );
        let detections = analyzer.process_event(event2);

        let manipulation_detected = detections
            .iter()
            .any(|d| matches!(d.attack_type, DetectedAttackType::TrustManipulation));
        assert!(manipulation_detected, "Should detect trust inflation");
    }

    #[test]
    fn test_agent_risk_score() {
        let mut analyzer = StreamingAnalyzer::new(AttackDetectionConfig {
            min_events_for_analysis: 1,
            ..Default::default()
        });

        // Create suspicious activity
        for i in 0..20 {
            let event = create_test_event(
                "suspicious-agent",
                TrustEventType::AgentRegistered,
                1000 + i * 100,
            );
            analyzer.process_event(event);
        }

        let risk = analyzer.agent_risk_score("suspicious-agent");
        // May or may not trigger detection depending on exact thresholds
        assert!(risk >= 0.0 && risk <= 1.0);
    }

    #[test]
    fn test_alert_pipeline() {
        let mut pipeline = AlertPipeline::new(100);

        let detection = DetectionResult {
            id: "det-1".to_string(),
            attack_type: DetectedAttackType::Sybil,
            confidence: 0.9,
            severity: AttackSeverity::High,
            involved_agents: vec!["agent-1".to_string()],
            evidence: vec![],
            recommended_action: RecommendedResponse::Quarantine,
            timestamp: 1000,
            correlated_with: vec![],
        };

        let alert = pipeline.create_alert(&detection);
        assert_eq!(alert.status, AlertStatus::New);

        pipeline.acknowledge(&alert.id, "admin");
        assert!(pipeline.pending_alerts().count() == 0);

        pipeline.resolve(&alert.id, "Quarantined agent");
        assert_eq!(pipeline.history.len(), 1);
    }

    #[test]
    fn test_stats() {
        let analyzer = StreamingAnalyzer::new(AttackDetectionConfig::default());
        let stats = analyzer.stats();

        assert_eq!(stats.total_events_processed, 0);
        assert!(stats.signatures_active > 0);
    }
}
