// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sentinel Manager — Distributed Immune Pattern Recognition
//!
//! Detects governance anomalies, network threats, and behavioral patterns
//! that indicate attacks on the Mycelix-Symthaea ecosystem.
//!
//! Implements [`CognitiveSubsystem`] at interval 67 (co-prime with all other managers).
//!
//! ## Threat Signals
//!
//! 1. **Proposal flooding**: Excessive proposals from single agent (rate limiting)
//! 2. **Vote clustering**: Correlated voting patterns suggesting Sybil attack
//! 3. **Peer behavior anomaly**: Mahalanobis distance on consciousness vectors
//! 4. **Gradient fingerprinting**: Same source behind multiple federated identities
//! 5. **Timing anomaly**: Suspiciously consistent latency (bot behavior)
//! 6. **Dispatch loop**: Cross-cluster call amplification attack
//! 7. **Consciousness manipulation**: Rapid tier changes suggesting gaming
//!
//! ## Design
//!
//! Like the other managers, SentinelManager maintains internal state but proposes
//! changes via `SubsystemOutput`. It does NOT mutate CognitiveLoopService directly.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::super::thresholds;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// SENTINEL EVENTS — injected from governance/network layers
// ═══════════════════════════════════════════════════════════════════════════════

/// External events that feed the sentinel's threat detection.
#[derive(Debug, Clone)]
pub enum SentinelEvent {
    /// A governance proposal was created.
    ProposalCreated {
        agent_id: String,
        proposal_id: String,
        timestamp_us: u64,
    },
    /// A vote was cast.
    VoteCast {
        agent_id: String,
        proposal_id: String,
        vote_direction: bool, // true = approve, false = reject
        timestamp_us: u64,
    },
    /// A cross-cluster dispatch was executed.
    CrossClusterDispatch {
        source_cluster: String,
        target_cluster: String,
        timestamp_us: u64,
    },
    /// A peer's consciousness tier changed.
    TierChange {
        agent_id: String,
        old_tier: u8,
        new_tier: u8,
        timestamp_us: u64,
    },
    /// A federated learning gradient was received.
    GradientReceived {
        peer_id: String,
        magnitude: f64,
        timestamp_us: u64,
    },
    /// An external threat report from another sentinel node.
    ExternalThreatReport {
        reporter_id: String,
        threat_type: ThreatSignalKind,
        severity: f32,
        evidence_hash: u64,
    },
    /// A coordinated cartel detected by the CartelDetector.
    ///
    /// When confidence exceeds `CARTEL_SLASH_MIN_CONFIDENCE` (0.7),
    /// the sentinel records this pattern in ThreatMemory and may
    /// escalate safety level for the affected governance domain.
    CartelDetected {
        members: Vec<String>,
        confidence: f64,
        avg_similarity: f64,
    },
    /// Oppression pattern detected by governance reflection.
    ///
    /// Emitted when `reflect_on_proposal()` identifies high echo chamber
    /// risk combined with power concentration, absent harmonies, or
    /// fragmentation warnings. Feeds into ThreatMemory for HDV pattern
    /// recognition of systematic oppression across proposals.
    OppressionDetected {
        proposal_id: String,
        echo_chamber_risk: String,
        concentration_warning: bool,
        fragmentation_warning: bool,
        absent_harmonies: Vec<String>,
        timestamp_us: u64,
    },
}

/// Types of threat signals the sentinel can detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatSignalKind {
    /// Excessive proposals from single agent.
    ProposalFlood,
    /// Correlated voting patterns (Sybil indicator).
    VoteClustering,
    /// Peer consciousness vector anomaly.
    PeerBehaviorAnomaly,
    /// Same gradient source behind multiple identities.
    GradientFingerprint,
    /// Suspiciously consistent latency (bot behavior).
    TimingAnomaly,
    /// Cross-cluster call amplification.
    DispatchLoop,
    /// Rapid consciousness tier changes.
    ConsciousnessManipulation,
    /// Coordinated cartel attack (from CartelDetector).
    CartelAttack,
    /// Systematic oppression pattern in governance.
    GovernanceOppression,
    /// Epistemic threat: uncalibrated confidence, dogmatic certainty,
    /// or systematic resistance to evidence update.
    EpistemicThreat,
}

impl ThreatSignalKind {
    /// Base severity of this threat type (used for prioritization).
    pub fn base_severity(&self) -> f32 {
        match self {
            Self::ProposalFlood => 0.4,
            Self::VoteClustering => 0.6,
            Self::PeerBehaviorAnomaly => 0.5,
            Self::GradientFingerprint => 0.7,
            Self::TimingAnomaly => 0.3,
            Self::DispatchLoop => 0.5,
            Self::ConsciousnessManipulation => 0.6,
            Self::CartelAttack => 0.7,
            Self::GovernanceOppression => 0.6,
            Self::EpistemicThreat => 0.5,
        }
    }
}

/// A detected threat signal with evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatSignal {
    /// What kind of threat.
    pub kind: ThreatSignalKind,
    /// Confidence in this detection (0.0-1.0).
    pub confidence: f32,
    /// Severity (0.0-1.0), combining base severity and evidence strength.
    pub severity: f32,
    /// Agent or peer implicated (if any).
    pub subject: Option<String>,
    /// Cycle at which this was detected.
    pub detected_at: u64,
    /// Human-readable evidence summary.
    pub evidence: String,
}

/// Telemetry snapshot for CycleMetadata integration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SentinelTelemetry {
    /// Number of active threat signals.
    pub active_threats: usize,
    /// Highest threat severity this cycle.
    pub max_severity: f32,
    /// Number of agents being monitored.
    pub monitored_agents: usize,
    /// Number of quarantined peers.
    pub quarantined_peers: usize,
    /// Aggregate threat level (0.0 = calm, 1.0 = critical).
    pub threat_level: f32,
    /// Number of proposals rate-limited this interval.
    pub proposals_rate_limited: u32,
    /// Number of external threat reports received.
    pub external_reports_received: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SENTINEL MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-agent activity tracking for anomaly detection.
#[derive(Debug, Clone, Default)]
struct AgentActivity {
    /// Recent proposal timestamps (sliding window, last 60s).
    proposal_times: VecDeque<u64>,
    /// Recent vote directions for correlation analysis.
    vote_history: VecDeque<(String, bool)>, // (proposal_id, direction)
    /// Recent tier changes.
    tier_changes: VecDeque<u64>,
    /// Anomaly score (EMA, 0.0-1.0).
    anomaly_score: f64,
}

/// Per-peer gradient tracking for fingerprinting.
#[derive(Debug, Clone, Default)]
struct GradientProfile {
    /// Recent gradient magnitudes.
    magnitudes: VecDeque<f64>,
    /// Recent timestamps.
    timestamps: VecDeque<u64>,
}

/// Sentinel Manager — immune pattern recognition for governance and network defense.
///
/// Implements `CognitiveSubsystem` at interval 67.
pub(crate) struct SentinelManager {
    // ── Event queue ─────────────────────────────────────────────────────
    pending_events: Vec<SentinelEvent>,

    // ── Per-agent tracking ──────────────────────────────────────────────
    agent_activity: HashMap<String, AgentActivity>,

    // ── Per-peer gradient tracking ──────────────────────────────────────
    gradient_profiles: HashMap<String, GradientProfile>,

    // ── Cross-cluster dispatch tracking ─────────────────────────────────
    dispatch_history: VecDeque<(String, String, u64)>, // (src, dst, timestamp)

    // ── Active threats ──────────────────────────────────────────────────
    active_threats: Vec<ThreatSignal>,

    // ── Quarantine list ─────────────────────────────────────────────────
    quarantined_peers: HashMap<String, u64>, // peer_id -> expiry cycle

    // ── External reports ────────────────────────────────────────────────
    external_reports: Vec<(ThreatSignalKind, f32)>, // (kind, severity)

    // ── Telemetry ───────────────────────────────────────────────────────
    last_telemetry: SentinelTelemetry,

    // ── Configuration ───────────────────────────────────────────────────
    /// Maximum proposals per agent per 60-second window.
    proposal_rate_limit: u32,
    /// Anomaly EMA alpha.
    anomaly_ema_alpha: f64,
    /// Maximum tracked agents (prevents unbounded growth).
    max_tracked_agents: usize,
    /// Maximum tracked gradient profiles.
    max_tracked_gradients: usize,
}

impl Default for SentinelManager {
    fn default() -> Self {
        Self {
            pending_events: Vec::with_capacity(64),
            agent_activity: HashMap::new(),
            gradient_profiles: HashMap::new(),
            dispatch_history: VecDeque::with_capacity(256),
            active_threats: Vec::new(),
            quarantined_peers: HashMap::new(),
            external_reports: Vec::new(),
            last_telemetry: SentinelTelemetry::default(),
            proposal_rate_limit: 10, // per 60s
            anomaly_ema_alpha: 0.15,
            max_tracked_agents: 1024,
            max_tracked_gradients: 256,
        }
    }
}

impl SentinelManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 67;

    /// Maximum dispatch history entries.
    const MAX_DISPATCH_HISTORY: usize = 256;

    /// Sliding window for proposal rate limiting (microseconds = 60s).
    const PROPOSAL_WINDOW_US: u64 = 60_000_000;

    /// Tier change window (microseconds = 5 minutes).
    const TIER_CHANGE_WINDOW_US: u64 = 300_000_000;

    /// Maximum tier changes in window before flagging manipulation.
    const MAX_TIER_CHANGES: usize = 3;

    /// Vote correlation threshold for Sybil detection.
    /// If two agents agree on >90% of proposals, flag as suspicious.
    const VOTE_CORRELATION_THRESHOLD: f64 = 0.9;

    /// Gradient magnitude similarity threshold for fingerprinting.
    const GRADIENT_SIMILARITY_THRESHOLD: f64 = 0.95;

    /// Threat decay rate per cycle (multiplicative).
    const THREAT_DECAY: f32 = 0.995;

    /// Minimum confidence to emit a threat signal.
    const MIN_THREAT_CONFIDENCE: f32 = 0.3;

    // ── Public API ──────────────────────────────────────────────────────

    /// Inject a sentinel event for processing on the next cycle.
    pub fn inject_event(&mut self, event: SentinelEvent) {
        self.pending_events.push(event);
    }

    /// Get the current telemetry snapshot.
    pub fn telemetry(&self) -> &SentinelTelemetry {
        &self.last_telemetry
    }

    /// Get active threat signals.
    pub fn active_threats(&self) -> &[ThreatSignal] {
        &self.active_threats
    }

    /// Check if a peer is quarantined.
    pub fn is_quarantined(&self, peer_id: &str) -> bool {
        self.quarantined_peers.contains_key(peer_id)
    }

    /// Quarantine a peer for a given number of cycles.
    pub fn quarantine_peer(&mut self, peer_id: String, duration_cycles: u64, current_cycle: u64) {
        self.quarantined_peers
            .insert(peer_id, current_cycle + duration_cycles);
    }

    /// Release expired quarantines.
    pub fn release_expired_quarantines(&mut self, current_cycle: u64) {
        self.quarantined_peers
            .retain(|_, expiry| *expiry > current_cycle);
    }

    /// Aggregate threat level (0.0-1.0).
    pub fn threat_level(&self) -> f32 {
        if self.active_threats.is_empty() {
            return 0.0;
        }
        // Weighted sum: higher severity threats contribute more
        let weighted_sum: f32 = self
            .active_threats
            .iter()
            .map(|t| t.severity * t.confidence)
            .sum();
        (weighted_sum / self.active_threats.len().max(1) as f32).clamp(0.0, 1.0)
    }

    // ── Internal detection logic ────────────────────────────────────────

    fn process_events(&mut self, cycle: u64) {
        let events = std::mem::take(&mut self.pending_events);
        let mut proposals_rate_limited = 0u32;
        let mut external_count = 0u32;

        for event in events {
            match event {
                SentinelEvent::ProposalCreated {
                    agent_id,
                    timestamp_us,
                    ..
                } => {
                    let rate_limit = self.proposal_rate_limit;
                    let ema_alpha = self.anomaly_ema_alpha;
                    let activity = self.get_or_create_agent(&agent_id);
                    // Evict old proposals outside window
                    while activity.proposal_times.front().map_or(false, |t| {
                        timestamp_us.saturating_sub(*t) > Self::PROPOSAL_WINDOW_US
                    }) {
                        activity.proposal_times.pop_front();
                    }
                    activity.proposal_times.push_back(timestamp_us);

                    // Check rate limit
                    let count = activity.proposal_times.len();
                    if count > rate_limit as usize {
                        let confidence = (count as f32 / rate_limit as f32 - 1.0).min(1.0);
                        // Bump anomaly score before dropping the borrow
                        activity.anomaly_score =
                            activity.anomaly_score * (1.0 - ema_alpha) + ema_alpha;
                        self.emit_threat(ThreatSignal {
                            kind: ThreatSignalKind::ProposalFlood,
                            confidence,
                            severity: ThreatSignalKind::ProposalFlood.base_severity() * confidence,
                            subject: Some(agent_id.clone()),
                            detected_at: cycle,
                            evidence: format!(
                                "{} proposals in 60s from agent {} (limit: {})",
                                count, agent_id, rate_limit
                            ),
                        });
                        proposals_rate_limited += 1;
                    }
                }

                SentinelEvent::VoteCast {
                    agent_id,
                    proposal_id,
                    vote_direction,
                    ..
                } => {
                    let activity = self.get_or_create_agent(&agent_id);
                    activity
                        .vote_history
                        .push_back((proposal_id, vote_direction));
                    if activity.vote_history.len() > 100 {
                        activity.vote_history.pop_front();
                    }
                }

                SentinelEvent::TierChange {
                    agent_id,
                    timestamp_us,
                    old_tier,
                    new_tier,
                    ..
                } => {
                    let ema_alpha = self.anomaly_ema_alpha;
                    let activity = self.get_or_create_agent(&agent_id);
                    // Evict old changes outside window
                    while activity.tier_changes.front().map_or(false, |t| {
                        timestamp_us.saturating_sub(*t) > Self::TIER_CHANGE_WINDOW_US
                    }) {
                        activity.tier_changes.pop_front();
                    }
                    activity.tier_changes.push_back(timestamp_us);

                    let count = activity.tier_changes.len();
                    if count > Self::MAX_TIER_CHANGES {
                        let confidence =
                            (count as f32 / Self::MAX_TIER_CHANGES as f32 - 1.0).min(1.0);
                        activity.anomaly_score =
                            activity.anomaly_score * (1.0 - ema_alpha) + ema_alpha * 0.8;
                        self.emit_threat(ThreatSignal {
                            kind: ThreatSignalKind::ConsciousnessManipulation,
                            confidence,
                            severity: ThreatSignalKind::ConsciousnessManipulation.base_severity()
                                * confidence,
                            subject: Some(agent_id.clone()),
                            detected_at: cycle,
                            evidence: format!(
                                "{} tier changes in 5min for agent {} ({}->{})",
                                count, agent_id, old_tier, new_tier
                            ),
                        });
                    }
                }

                SentinelEvent::CrossClusterDispatch {
                    source_cluster,
                    target_cluster,
                    timestamp_us,
                } => {
                    self.dispatch_history.push_back((
                        source_cluster.clone(),
                        target_cluster.clone(),
                        timestamp_us,
                    ));
                    while self.dispatch_history.len() > Self::MAX_DISPATCH_HISTORY {
                        self.dispatch_history.pop_front();
                    }

                    // Detect ping-pong: same (src, dst) pair > 20 times in last 256 entries
                    let pair_count = self
                        .dispatch_history
                        .iter()
                        .filter(|(s, d, _)| s == &source_cluster && d == &target_cluster)
                        .count();
                    if pair_count > 20 {
                        let confidence = ((pair_count as f32 - 20.0) / 20.0).min(1.0);
                        self.emit_threat(ThreatSignal {
                            kind: ThreatSignalKind::DispatchLoop,
                            confidence,
                            severity: ThreatSignalKind::DispatchLoop.base_severity() * confidence,
                            subject: Some(format!("{}->{}", source_cluster, target_cluster)),
                            detected_at: cycle,
                            evidence: format!(
                                "Dispatch loop: {}->{} occurred {} times in recent history",
                                source_cluster, target_cluster, pair_count
                            ),
                        });
                    }
                }

                SentinelEvent::GradientReceived {
                    peer_id,
                    magnitude,
                    timestamp_us,
                } => {
                    if self.gradient_profiles.len() < self.max_tracked_gradients
                        || self.gradient_profiles.contains_key(&peer_id)
                    {
                        let profile = self.gradient_profiles.entry(peer_id).or_default();
                        profile.magnitudes.push_back(magnitude);
                        profile.timestamps.push_back(timestamp_us);
                        if profile.magnitudes.len() > 50 {
                            profile.magnitudes.pop_front();
                            profile.timestamps.pop_front();
                        }
                    }
                }

                SentinelEvent::ExternalThreatReport {
                    threat_type,
                    severity,
                    ..
                } => {
                    self.external_reports.push((threat_type, severity));
                    external_count += 1;
                }

                SentinelEvent::CartelDetected {
                    ref members,
                    confidence,
                    ..
                } => {
                    if confidence > 0.7 && members.len() >= 3 {
                        self.emit_threat(ThreatSignal {
                            kind: ThreatSignalKind::CartelAttack,
                            confidence: confidence as f32,
                            severity: (confidence * 0.8) as f32,
                            subject: Some(members.join(",")),
                            detected_at: cycle,
                            evidence: format!(
                                "Cartel of {} members detected with confidence {:.2}",
                                members.len(),
                                confidence
                            ),
                        });
                    }
                }

                SentinelEvent::OppressionDetected {
                    ref proposal_id,
                    ref echo_chamber_risk,
                    concentration_warning,
                    fragmentation_warning,
                    ref absent_harmonies,
                    ..
                } => {
                    // Score oppression severity from combined signals
                    let severity = {
                        let mut s = 0.0f32;
                        if echo_chamber_risk == "High" {
                            s += 0.4;
                        }
                        if concentration_warning {
                            s += 0.3;
                        }
                        if fragmentation_warning {
                            s += 0.2;
                        }
                        if absent_harmonies.len() > 2 {
                            s += 0.1;
                        }
                        s.min(1.0)
                    };
                    if severity > 0.3 {
                        self.emit_threat(ThreatSignal {
                            kind: ThreatSignalKind::GovernanceOppression,
                            confidence: severity,
                            severity,
                            subject: Some(proposal_id.clone()),
                            detected_at: cycle,
                            evidence: format!(
                                "Oppression pattern: echo={}, concentration={}, fragmentation={}, absent_harmonies={}",
                                echo_chamber_risk, concentration_warning, fragmentation_warning, absent_harmonies.len()
                            ),
                        });
                    }
                }
            }
        }

        // Run cross-agent vote correlation analysis (expensive, only when we have data)
        self.detect_vote_clustering(cycle);

        // Run gradient fingerprinting
        self.detect_gradient_fingerprints(cycle);

        // Decay existing threats
        self.active_threats.retain_mut(|t| {
            t.severity *= Self::THREAT_DECAY;
            t.severity > 0.01 // Remove negligible threats
        });

        // Process external reports
        for (kind, severity) in std::mem::take(&mut self.external_reports) {
            // Corroborate: boost existing matching threats, or add new
            if let Some(existing) = self.active_threats.iter_mut().find(|t| t.kind == kind) {
                existing.confidence = (existing.confidence + 0.1).min(1.0);
                existing.severity = (existing.severity + severity * 0.3).min(1.0);
            } else if severity > Self::MIN_THREAT_CONFIDENCE {
                self.emit_threat(ThreatSignal {
                    kind,
                    confidence: 0.5, // external reports get 50% base confidence
                    severity: severity * 0.7,
                    subject: None,
                    detected_at: cycle,
                    evidence: "External threat report from peer sentinel".to_string(),
                });
            }
        }

        // Update telemetry
        self.last_telemetry = SentinelTelemetry {
            active_threats: self.active_threats.len(),
            max_severity: self
                .active_threats
                .iter()
                .map(|t| t.severity)
                .fold(0.0f32, f32::max),
            monitored_agents: self.agent_activity.len(),
            quarantined_peers: self.quarantined_peers.len(),
            threat_level: self.threat_level(),
            proposals_rate_limited,
            external_reports_received: external_count,
        };
    }

    fn detect_vote_clustering(&mut self, cycle: u64) {
        // Compare vote histories between agents for correlation
        let agents: Vec<String> = self.agent_activity.keys().cloned().collect();
        if agents.len() < 2 {
            return;
        }

        // Only check pairs with sufficient shared proposals (>= 5)
        for i in 0..agents.len().min(50) {
            // cap at 50 agents to avoid O(n^2) explosion
            for j in (i + 1)..agents.len().min(50) {
                let a = &self.agent_activity[&agents[i]];
                let b = &self.agent_activity[&agents[j]];

                // Find shared proposals
                let shared: Vec<bool> = a
                    .vote_history
                    .iter()
                    .filter_map(|(pid, dir_a)| {
                        b.vote_history
                            .iter()
                            .find(|(bpid, _)| bpid == pid)
                            .map(|(_, dir_b)| dir_a == dir_b)
                    })
                    .collect();

                if shared.len() >= 5 {
                    let agreement_rate =
                        shared.iter().filter(|&&x| x).count() as f64 / shared.len() as f64;
                    if agreement_rate > Self::VOTE_CORRELATION_THRESHOLD {
                        let confidence = ((agreement_rate - Self::VOTE_CORRELATION_THRESHOLD)
                            * 10.0)
                            .min(1.0) as f32;
                        self.emit_threat(ThreatSignal {
                            kind: ThreatSignalKind::VoteClustering,
                            confidence,
                            severity: ThreatSignalKind::VoteClustering.base_severity() * confidence,
                            subject: Some(format!("{} + {}", agents[i], agents[j])),
                            detected_at: cycle,
                            evidence: format!(
                                "Vote correlation {:.1}% between {} and {} over {} shared proposals",
                                agreement_rate * 100.0,
                                agents[i],
                                agents[j],
                                shared.len()
                            ),
                        });
                    }
                }
            }
        }
    }

    fn detect_gradient_fingerprints(&mut self, cycle: u64) {
        let peers: Vec<String> = self.gradient_profiles.keys().cloned().collect();
        if peers.len() < 2 {
            return;
        }

        for i in 0..peers.len().min(30) {
            for j in (i + 1)..peers.len().min(30) {
                let a = &self.gradient_profiles[&peers[i]];
                let b = &self.gradient_profiles[&peers[j]];

                if a.magnitudes.len() >= 5 && b.magnitudes.len() >= 5 {
                    // Compare recent magnitude sequences using cosine similarity
                    let len = a.magnitudes.len().min(b.magnitudes.len()).min(20);
                    let a_slice: Vec<f64> = a.magnitudes.iter().rev().take(len).copied().collect();
                    let b_slice: Vec<f64> = b.magnitudes.iter().rev().take(len).copied().collect();

                    let dot: f64 = a_slice.iter().zip(&b_slice).map(|(x, y)| x * y).sum();
                    let norm_a: f64 = a_slice.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let norm_b: f64 = b_slice.iter().map(|x| x * x).sum::<f64>().sqrt();

                    if norm_a > 1e-10 && norm_b > 1e-10 {
                        let similarity = dot / (norm_a * norm_b);
                        if similarity > Self::GRADIENT_SIMILARITY_THRESHOLD {
                            let confidence = ((similarity - Self::GRADIENT_SIMILARITY_THRESHOLD)
                                * 20.0)
                                .min(1.0) as f32;
                            self.emit_threat(ThreatSignal {
                                kind: ThreatSignalKind::GradientFingerprint,
                                confidence,
                                severity: ThreatSignalKind::GradientFingerprint.base_severity()
                                    * confidence,
                                subject: Some(format!("{} ~ {}", peers[i], peers[j])),
                                detected_at: cycle,
                                evidence: format!(
                                    "Gradient fingerprint: similarity {:.3} between {} and {}",
                                    similarity, peers[i], peers[j]
                                ),
                            });
                        }
                    }
                }
            }
        }
    }

    fn emit_threat(&mut self, threat: ThreatSignal) {
        if threat.confidence < Self::MIN_THREAT_CONFIDENCE {
            return;
        }
        // Deduplicate: merge with existing threat of same kind + subject
        if let Some(existing) = self
            .active_threats
            .iter_mut()
            .find(|t| t.kind == threat.kind && t.subject == threat.subject)
        {
            existing.confidence = (existing.confidence + threat.confidence * 0.5).min(1.0);
            existing.severity = existing.severity.max(threat.severity);
            existing.detected_at = threat.detected_at;
            existing.evidence = threat.evidence;
        } else {
            self.active_threats.push(threat);
        }
    }

    fn get_or_create_agent(&mut self, agent_id: &str) -> &mut AgentActivity {
        if !self.agent_activity.contains_key(agent_id) {
            // Evict oldest agent if at capacity
            if self.agent_activity.len() >= self.max_tracked_agents {
                // Remove agent with lowest anomaly score
                if let Some(min_key) = self
                    .agent_activity
                    .iter()
                    .min_by(|a, b| {
                        a.1.anomaly_score
                            .partial_cmp(&b.1.anomaly_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| k.clone())
                {
                    self.agent_activity.remove(&min_key);
                }
            }
            self.agent_activity
                .insert(agent_id.to_string(), AgentActivity::default());
        }
        self.agent_activity
            .get_mut(agent_id)
            .expect("agent_activity entry was just inserted")
    }
}

impl CognitiveSubsystem for SentinelManager {
    fn name(&self) -> &'static str {
        "SentinelManager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let cycle = snapshot.cycle_number;

        // Release expired quarantines
        self.release_expired_quarantines(cycle);

        // Process all pending events
        self.process_events(cycle);

        // Compute output proposals based on threat level
        let mut output = SubsystemOutput::default();
        let threat_level = self.threat_level();

        if threat_level > 0.0 {
            // Arousal increase proportional to threat (vigilance)
            // Basis: Aston-Jones & Cohen (2005) — LC-NE governs arousal/vigilance
            output.arousal_delta = threat_level * thresholds::SENTINEL_AROUSAL_SCALE_NORMAL;
            output.flags |= output_flags::ANOMALY_DETECTED;

            if threat_level > thresholds::SENTINEL_THREAT_MODERATE {
                // Dampen exploration during active threat
                output.exploration_delta =
                    -(threat_level as f64) * thresholds::SENTINEL_EXPLORATION_DAMPEN_SCALE;
                // Additional arousal spike
                output.arousal_delta = threat_level * thresholds::SENTINEL_AROUSAL_SCALE_HEIGHTENED;
            }

            if threat_level > thresholds::SENTINEL_THREAT_CRITICAL {
                // Significant threat: reduce confidence (epistemic caution)
                output.confidence_delta =
                    -(threat_level as f64) * thresholds::SENTINEL_CONFIDENCE_DAMPEN_SCALE;
                // Signal urgency escalation
                output.flags |= output_flags::ESCALATE_URGENCY;
            }
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Serialize aggregate threat-tracking state. Per-agent/per-peer HashMaps
        // and event queues are transient and not checkpointed — they rebuild
        // from live events after restart.
        //
        // Layout: [threat_level: f32 = 4][active_threat_count: u32 = 4]
        //         [quarantined_count: u32 = 4][proposal_rate_limit: u32 = 4]
        //         [anomaly_ema_alpha: f64 = 8]
        // Total: 24 bytes
        let mut data = Vec::with_capacity(24);
        data.extend_from_slice(&self.threat_level().to_le_bytes());
        data.extend_from_slice(&(self.active_threats.len() as u32).to_le_bytes());
        data.extend_from_slice(&(self.quarantined_peers.len() as u32).to_le_bytes());
        data.extend_from_slice(&self.proposal_rate_limit.to_le_bytes());
        data.extend_from_slice(&self.anomaly_ema_alpha.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        // Only restore configuration fields that affect future behavior.
        // Threat state and quarantines are transient — they rebuild from
        // live events after restart.
        const MIN_SIZE: usize = 4 + 4 + 4 + 4 + 8; // 24
        if data.len() < MIN_SIZE {
            return Err(format!(
                "SentinelManager checkpoint too short: {} < {}",
                data.len(),
                MIN_SIZE
            ));
        }
        // bytes [0..4]: threat_level (informational, not restored — recomputed)
        // bytes [4..8]: active_threat_count (informational, not restored)
        // bytes [8..12]: quarantined_count (informational, not restored)
        self.proposal_rate_limit = u32::from_le_bytes(
            data[12..16]
                .try_into()
                .map_err(|_| "SentinelManager: corrupt proposal_rate_limit bytes")?,
        );
        self.anomaly_ema_alpha = f64::from_le_bytes(
            data[16..24]
                .try_into()
                .map_err(|_| "SentinelManager: corrupt anomaly_ema_alpha bytes")?,
        );
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOVERNANCE → SENTINEL EVENT BRIDGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert a governance event into a sentinel event for threat detection.
///
/// Returns `None` for governance events that don't carry threat-relevant data
/// (e.g., reputation changes, reciprocity pledges).
#[cfg(feature = "mycelix")]
pub fn bridge_governance_event(
    gov_event: &super::governance_manager::GovernanceEvent,
    agent_id: &str,
) -> Option<SentinelEvent> {
    use super::governance_manager::GovernanceEventKind;

    match &gov_event.kind {
        GovernanceEventKind::ProposalCreated => Some(SentinelEvent::ProposalCreated {
            agent_id: agent_id.to_string(),
            proposal_id: gov_event.proposal_id.clone().unwrap_or_else(|| {
                tracing::warn!("SentinelManager: missing proposal_id in ProposalCreated event");
                "[unknown]".to_string()
            }),
            timestamp_us: gov_event.timestamp_secs * 1_000_000,
        }),
        GovernanceEventKind::VoteCast { vote_value, .. } => Some(SentinelEvent::VoteCast {
            agent_id: agent_id.to_string(),
            proposal_id: gov_event.proposal_id.clone().unwrap_or_else(|| {
                tracing::warn!("SentinelManager: missing proposal_id in VoteCast event");
                "[unknown]".to_string()
            }),
            vote_direction: *vote_value > 0.0,
            timestamp_us: gov_event.timestamp_secs * 1_000_000,
        }),
        GovernanceEventKind::EmergencyDeclared => Some(SentinelEvent::ExternalThreatReport {
            reporter_id: "governance".to_string(),
            threat_type: ThreatSignalKind::ConsciousnessManipulation,
            severity: 0.8,
            evidence_hash: gov_event.timestamp_secs,
        }),
        _ => None, // Reputation, reciprocity, justice — not direct threat signals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(cycle: u64) -> CycleSnapshot {
        CycleSnapshot {
            cycle_number: cycle,
            ..CycleSnapshot::default()
        }
    }

    #[test]
    fn test_default_no_threats() {
        let mgr = SentinelManager::default();
        assert_eq!(mgr.threat_level(), 0.0);
        assert!(mgr.active_threats().is_empty());
    }

    #[test]
    fn test_proposal_flood_detection() {
        let mut mgr = SentinelManager::default();
        let now = 1_000_000_000u64;

        // Inject 15 proposals (limit is 10)
        for i in 0..15 {
            mgr.inject_event(SentinelEvent::ProposalCreated {
                agent_id: "attacker".to_string(),
                proposal_id: format!("prop_{}", i),
                timestamp_us: now + i as u64 * 1000, // 1ms apart
            });
        }

        let snap = make_snapshot(100);
        mgr.process(&snap);

        assert!(
            mgr.active_threats()
                .iter()
                .any(|t| t.kind == ThreatSignalKind::ProposalFlood)
        );
        assert!(mgr.threat_level() > 0.0);
    }

    #[test]
    fn test_tier_manipulation_detection() {
        let mut mgr = SentinelManager::default();
        let now = 1_000_000_000u64;

        // 5 tier changes in 5 minutes (limit is 3)
        for i in 0..5 {
            mgr.inject_event(SentinelEvent::TierChange {
                agent_id: "gamer".to_string(),
                old_tier: i % 5,
                new_tier: (i + 1) % 5,
                timestamp_us: now + i as u64 * 30_000_000, // 30s apart
            });
        }

        let snap = make_snapshot(200);
        mgr.process(&snap);

        assert!(
            mgr.active_threats()
                .iter()
                .any(|t| t.kind == ThreatSignalKind::ConsciousnessManipulation)
        );
    }

    #[test]
    fn test_dispatch_loop_detection() {
        let mut mgr = SentinelManager::default();
        let now = 1_000_000_000u64;

        // 30 dispatches between same pair (threshold is 20, need confidence > 0.3)
        // confidence = (30 - 20) / 20 = 0.5 > MIN_THREAT_CONFIDENCE (0.3)
        for i in 0..30 {
            mgr.inject_event(SentinelEvent::CrossClusterDispatch {
                source_cluster: "commons".to_string(),
                target_cluster: "civic".to_string(),
                timestamp_us: now + i as u64 * 1000,
            });
        }

        let snap = make_snapshot(300);
        mgr.process(&snap);

        assert!(
            mgr.active_threats()
                .iter()
                .any(|t| t.kind == ThreatSignalKind::DispatchLoop)
        );
    }

    #[test]
    fn test_quarantine_lifecycle() {
        let mut mgr = SentinelManager::default();

        mgr.quarantine_peer("bad_peer".to_string(), 100, 500);
        assert!(mgr.is_quarantined("bad_peer"));

        // Not expired yet
        mgr.release_expired_quarantines(550);
        assert!(mgr.is_quarantined("bad_peer"));

        // Expired
        mgr.release_expired_quarantines(601);
        assert!(!mgr.is_quarantined("bad_peer"));
    }

    #[test]
    fn test_threat_decay() {
        let mut mgr = SentinelManager::default();

        // Inject flood to create a threat
        let now = 1_000_000_000u64;
        for i in 0..15 {
            mgr.inject_event(SentinelEvent::ProposalCreated {
                agent_id: "flood".to_string(),
                proposal_id: format!("p{}", i),
                timestamp_us: now + i as u64 * 1000,
            });
        }
        mgr.process(&make_snapshot(100));
        let initial_severity = mgr.active_threats()[0].severity;

        // Process many cycles with no new events — threat should decay
        for c in 101..200 {
            mgr.process(&make_snapshot(c));
        }

        if !mgr.active_threats().is_empty() {
            assert!(mgr.active_threats()[0].severity < initial_severity);
        }
    }

    #[test]
    fn test_external_report_corroboration() {
        let mut mgr = SentinelManager::default();

        // Create initial threat
        let now = 1_000_000_000u64;
        for i in 0..15 {
            mgr.inject_event(SentinelEvent::ProposalCreated {
                agent_id: "flood".to_string(),
                proposal_id: format!("p{}", i),
                timestamp_us: now + i as u64 * 1000,
            });
        }
        mgr.process(&make_snapshot(100));
        let initial_confidence = mgr
            .active_threats()
            .iter()
            .find(|t| t.kind == ThreatSignalKind::ProposalFlood)
            .unwrap()
            .confidence;

        // External report corroborates
        mgr.inject_event(SentinelEvent::ExternalThreatReport {
            reporter_id: "peer_1".to_string(),
            threat_type: ThreatSignalKind::ProposalFlood,
            severity: 0.5,
            evidence_hash: 12345,
        });
        mgr.process(&make_snapshot(101));

        let updated_confidence = mgr
            .active_threats()
            .iter()
            .find(|t| t.kind == ThreatSignalKind::ProposalFlood)
            .unwrap()
            .confidence;
        assert!(updated_confidence >= initial_confidence);
    }

    #[test]
    fn test_vote_clustering_detection() {
        let mut mgr = SentinelManager::default();
        let now = 1_000_000_000u64;

        // Two agents that always agree
        for i in 0..10 {
            let pid = format!("proposal_{}", i);
            mgr.inject_event(SentinelEvent::VoteCast {
                agent_id: "sybil_a".to_string(),
                proposal_id: pid.clone(),
                vote_direction: true,
                timestamp_us: now + i as u64 * 1000,
            });
            mgr.inject_event(SentinelEvent::VoteCast {
                agent_id: "sybil_b".to_string(),
                proposal_id: pid,
                vote_direction: true,
                timestamp_us: now + i as u64 * 1000 + 100,
            });
        }

        mgr.process(&make_snapshot(400));

        assert!(
            mgr.active_threats()
                .iter()
                .any(|t| t.kind == ThreatSignalKind::VoteClustering)
        );
    }

    #[test]
    fn test_gradient_fingerprint_detection() {
        let mut mgr = SentinelManager::default();
        let now = 1_000_000_000u64;

        // Two peers with nearly identical gradient magnitudes
        for i in 0..10 {
            let mag = 0.5 + (i as f64) * 0.01;
            mgr.inject_event(SentinelEvent::GradientReceived {
                peer_id: "node_a".to_string(),
                magnitude: mag,
                timestamp_us: now + i as u64 * 1000,
            });
            mgr.inject_event(SentinelEvent::GradientReceived {
                peer_id: "node_b".to_string(),
                magnitude: mag + 0.0001, // nearly identical
                timestamp_us: now + i as u64 * 1000 + 50,
            });
        }

        mgr.process(&make_snapshot(500));

        assert!(
            mgr.active_threats()
                .iter()
                .any(|t| t.kind == ThreatSignalKind::GradientFingerprint)
        );
    }

    #[test]
    fn test_max_tracked_agents_eviction() {
        let mut mgr = SentinelManager::default();
        mgr.max_tracked_agents = 5;
        let now = 1_000_000_000u64;

        for i in 0..7 {
            mgr.inject_event(SentinelEvent::ProposalCreated {
                agent_id: format!("agent_{}", i),
                proposal_id: format!("prop_{}", i),
                timestamp_us: now,
            });
        }
        mgr.process(&make_snapshot(0));

        assert!(mgr.agent_activity.len() <= 5);
    }

    #[test]
    fn test_threat_level_bounded() {
        let mut mgr = SentinelManager::default();
        let now = 1_000_000_000u64;

        // Flood many different threat types
        for i in 0..50 {
            mgr.inject_event(SentinelEvent::ProposalCreated {
                agent_id: format!("flood_{}", i % 5),
                proposal_id: format!("p{}", i),
                timestamp_us: now + (i % 60) as u64 * 500_000, // within 60s window
            });
        }
        mgr.process(&make_snapshot(100));

        let level = mgr.threat_level();
        assert!(
            level >= 0.0 && level <= 1.0,
            "Threat level {} out of bounds",
            level
        );
    }

    #[test]
    fn test_subsystem_trait() {
        let mgr = SentinelManager::default();
        assert_eq!(mgr.name(), "SentinelManager");
        assert_eq!(mgr.interval(), 67);
    }

    #[test]
    fn test_arousal_boost_proportional_to_threat() {
        let mut mgr = SentinelManager::default();
        let now = 1_000_000_000u64;

        // Create a threat
        for i in 0..15 {
            mgr.inject_event(SentinelEvent::ProposalCreated {
                agent_id: "attacker".to_string(),
                proposal_id: format!("p{}", i),
                timestamp_us: now + i as u64 * 1000,
            });
        }
        let output = mgr.process(&make_snapshot(100));

        // Should have arousal boost and anomaly flag
        assert!(output.arousal_delta > 0.0);
        assert!(output.has_flag(output_flags::ANOMALY_DETECTED));
    }

    #[test]
    fn test_no_output_when_no_threats() {
        let mut mgr = SentinelManager::default();
        let output = mgr.process(&make_snapshot(67));

        assert!(output.is_neutral());
    }
}
