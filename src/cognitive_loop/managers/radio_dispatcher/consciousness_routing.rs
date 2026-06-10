// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-aware mesh routing, store-and-forward.

use super::tier::{PayloadClass, PayloadClassifier, RadioTier, RoutingDecision};
use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS-AWARE MESH ROUTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness-aware routing layer — routes mesh traffic based on cognitive
/// urgency, node consciousness level, moral topology, and governance tier.
///
/// This extends the physical-layer `PayloadClassifier` with consciousness semantics:
/// - Moral emergencies (ahimsa violations) bypass bandwidth budgets
/// - Messages route preferentially through high-Phi nodes (most trustworthy relay)
/// - Guardian-tier nodes gate relay of moral alerts
/// - Adaptive sharing cadence based on collective Phi divergence
///
/// ## Scientific Basis
///
/// - Tononi (2004): Phi as measure of integrated information → routing trust
/// - Friston (2010): Active inference over mesh topology
/// - Clark & Chalmers (1998): Extended mind → network is cognitive substrate
pub struct ConsciousnessAwareRouter {
    /// This node's current Phi (integrated information).
    local_phi: f32,
    /// This node's consciousness tier (0=Observer, 1=Participant, 2=Citizen, 3=Steward, 4=Guardian).
    local_tier: u8,
    /// Known peer Phi values (keyed by node_id prefix).
    pub(super) peer_phi: HashMap<[u8; 8], PeerConsciousnessState>,
    /// Moral urgency override: when true, bypasses bandwidth budgets for current cycle.
    moral_emergency_active: bool,
    /// Collective Phi (mean of all peers including self).
    collective_phi: f32,
    /// Collective Phi divergence (variance — high means disagreement, share more).
    collective_phi_divergence: f32,
    /// Adaptive sharing cadence: cycles between consciousness broadcasts.
    /// Lower when collective divergence is high.
    sharing_cadence: u32,
    /// Cycles since last consciousness broadcast.
    cycles_since_share: u32,
    /// Cumulative threat signatures received this session.
    threat_observations: Vec<ThreatObservation>,
}

/// Consciousness state of a known mesh peer.
#[derive(Debug, Clone)]
pub struct PeerConsciousnessState {
    /// Peer's reported Phi.
    pub phi: f32,
    /// Peer's consciousness level [0.0, 1.0].
    pub consciousness_level: f32,
    /// Peer's governance tier (0-4).
    pub governance_tier: u8,
    /// Cycle when last updated.
    pub last_update_cycle: u64,
    /// Trust score [0.0, 1.0] — influenced by consistency of reports.
    pub trust: f32,
}

/// Compact threat observation for collective immune memory.
#[derive(Debug, Clone)]
pub struct ThreatObservation {
    /// Threat type (maps to SentinelManager's 7 types).
    pub threat_type: u8,
    /// Severity [0.0, 1.0].
    pub severity: f32,
    /// Offending agent hash (8-byte prefix).
    pub agent_hash: [u8; 8],
    /// Compact signature (32 bytes, HDV-based).
    pub signature: [u8; 32],
    /// Cycle when observed.
    pub observed_cycle: u64,
    /// Number of independent corroborations.
    pub corroboration_count: u16,
}

/// Consciousness-aware routing decision.
#[derive(Debug, Clone)]
pub enum ConsciousRoutingDecision {
    /// Route normally via PayloadClassifier.
    Normal(RoutingDecision),
    /// Moral emergency — bypass bandwidth budgets, route through highest-Phi path.
    MoralEmergency {
        /// Best tier available (preferring fastest).
        tier: RadioTier,
        /// Highest-Phi relay node (if multi-hop).
        preferred_relay: Option<[u8; 8]>,
    },
    /// Consciousness sharing — adaptive cadence says share now.
    ConsciousnessShare { tier: RadioTier },
    /// Threat broadcast — share immune signature with all peers.
    ThreatBroadcast {
        tier: RadioTier,
        threat: ThreatObservation,
    },
    /// Suppress — cadence says wait, or peer already has current state.
    Suppressed {
        reason: &'static str,
        cycles_until_next: u32,
    },
}

/// Default sharing cadence (cycles between consciousness broadcasts).
const DEFAULT_SHARING_CADENCE: u32 = 50;
/// Minimum sharing cadence when collective divergence is high.
const MIN_SHARING_CADENCE: u32 = 5;
/// Maximum sharing cadence when collective is stable.
const MAX_SHARING_CADENCE: u32 = 200;
/// Phi divergence threshold for increasing sharing frequency.
const PHI_DIVERGENCE_THRESHOLD: f32 = 0.15;
/// Maximum threat observations to retain.
const MAX_THREAT_OBSERVATIONS: usize = 64;
/// Threat observation expiry (cycles).
const THREAT_EXPIRY_CYCLES: u64 = 10_000;

impl Default for ConsciousnessAwareRouter {
    fn default() -> Self {
        Self {
            local_phi: 0.0,
            local_tier: 0,
            peer_phi: HashMap::new(),
            moral_emergency_active: false,
            collective_phi: 0.0,
            collective_phi_divergence: 0.0,
            sharing_cadence: DEFAULT_SHARING_CADENCE,
            cycles_since_share: 0,
            threat_observations: Vec::new(),
        }
    }
}

impl ConsciousnessAwareRouter {
    /// Update local consciousness state (called each cycle from CLS).
    pub fn update_local(&mut self, phi: f32, consciousness_level: f32, governance_tier: u8) {
        self.local_phi = phi;
        self.local_tier = governance_tier;
        self.cycles_since_share += 1;

        // Recompute collective state
        self.recompute_collective();
        self.adapt_cadence();
    }

    /// Update a peer's consciousness state (received via mesh).
    pub fn update_peer(
        &mut self,
        peer_id: [u8; 8],
        phi: f32,
        consciousness_level: f32,
        governance_tier: u8,
        current_cycle: u64,
    ) {
        let state = self
            .peer_phi
            .entry(peer_id)
            .or_insert(PeerConsciousnessState {
                phi: 0.0,
                consciousness_level: 0.0,
                governance_tier: 0,
                last_update_cycle: 0,
                trust: 0.5, // Start at neutral trust
            });

        // Update trust based on consistency: large jumps in Phi are suspicious.
        let phi_delta = (phi - state.phi).abs();
        if phi_delta > 0.5 {
            state.trust = (state.trust - 0.1).max(0.0);
        } else {
            state.trust = (state.trust + 0.01).min(1.0);
        }

        state.phi = phi;
        state.consciousness_level = consciousness_level;
        state.governance_tier = governance_tier;
        state.last_update_cycle = current_cycle;

        self.recompute_collective();
        self.adapt_cadence();
    }

    /// Signal a moral emergency (e.g., ahimsa violation detected).
    /// Next routing decision will bypass bandwidth budgets.
    pub fn signal_moral_emergency(&mut self) {
        self.moral_emergency_active = true;
    }

    /// Record a threat observation for collective immune response.
    pub fn record_threat(&mut self, threat: ThreatObservation) {
        // Check for existing observation on same agent — corroborate instead of duplicate
        if let Some(existing) = self
            .threat_observations
            .iter_mut()
            .find(|t| t.agent_hash == threat.agent_hash && t.threat_type == threat.threat_type)
        {
            existing.corroboration_count = existing.corroboration_count.saturating_add(1);
            existing.severity = existing.severity.max(threat.severity);
            return;
        }

        self.threat_observations.push(threat);

        // Cap observations
        if self.threat_observations.len() > MAX_THREAT_OBSERVATIONS {
            self.threat_observations.remove(0);
        }
    }

    /// Prune stale peers and threat observations.
    pub fn prune(&mut self, current_cycle: u64, max_peer_age: u64) {
        self.peer_phi.retain(|_, state| {
            current_cycle.saturating_sub(state.last_update_cycle) < max_peer_age
        });
        self.threat_observations
            .retain(|t| current_cycle.saturating_sub(t.observed_cycle) < THREAT_EXPIRY_CYCLES);
    }

    /// Route a consciousness-aware payload.
    ///
    /// Extends the physical PayloadClassifier with consciousness semantics.
    pub fn route(
        &mut self,
        class: PayloadClass,
        payload_size: usize,
        urgency: u8,
        classifier: &PayloadClassifier,
    ) -> ConsciousRoutingDecision {
        // Moral emergency overrides everything
        if self.moral_emergency_active {
            self.moral_emergency_active = false; // One-shot

            // Find highest-Phi relay for maximum trust
            let preferred_relay = self.highest_phi_peer();

            // Use fastest available tier
            let tier = if classifier.is_available(RadioTier::Local) {
                RadioTier::Local
            } else if classifier.is_available(RadioTier::Metro) {
                RadioTier::Metro
            } else {
                RadioTier::Regional
            };

            return ConsciousRoutingDecision::MoralEmergency {
                tier,
                preferred_relay,
            };
        }

        // Check for pending threat broadcasts
        if let Some(threat) = self.next_unbroadcast_threat() {
            let tier = if classifier.is_available(RadioTier::Metro) {
                RadioTier::Metro
            } else if classifier.is_available(RadioTier::Local) {
                RadioTier::Local
            } else {
                RadioTier::Regional
            };

            return ConsciousRoutingDecision::ThreatBroadcast { tier, threat };
        }

        // Check if it's time to share consciousness state
        if class == PayloadClass::ConsciousnessDelta {
            if self.cycles_since_share >= self.sharing_cadence {
                self.cycles_since_share = 0;

                let tier = if classifier.is_available(RadioTier::Metro) {
                    RadioTier::Metro
                } else if classifier.is_available(RadioTier::Local) {
                    RadioTier::Local
                } else {
                    RadioTier::Regional
                };

                return ConsciousRoutingDecision::ConsciousnessShare { tier };
            } else {
                return ConsciousRoutingDecision::Suppressed {
                    reason: "sharing cadence not reached",
                    cycles_until_next: self.sharing_cadence - self.cycles_since_share,
                };
            }
        }

        // Default: normal routing
        match classifier.route(class, payload_size, urgency) {
            Some(decision) => ConsciousRoutingDecision::Normal(decision),
            None => ConsciousRoutingDecision::Suppressed {
                reason: "no routing decision",
                cycles_until_next: 1,
            },
        }
    }

    /// Get the highest-Phi peer (most trustworthy relay).
    pub fn highest_phi_peer(&self) -> Option<[u8; 8]> {
        self.peer_phi
            .iter()
            .filter(|(_, state)| state.trust > 0.3) // Minimum trust threshold
            .max_by(|(_, a), (_, b)| {
                // Weight: Phi * trust * consciousness_level
                let score_a = a.phi * a.trust * a.consciousness_level;
                let score_b = b.phi * b.trust * b.consciousness_level;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| *id)
    }

    /// Get peers sorted by trustworthiness (Phi * trust * consciousness).
    pub fn peers_by_trust(&self) -> Vec<([u8; 8], f32)> {
        let mut peers: Vec<_> = self
            .peer_phi
            .iter()
            .map(|(id, state)| {
                let score = state.phi * state.trust * state.consciousness_level;
                (*id, score)
            })
            .collect();
        peers.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        peers
    }

    /// Current collective Phi (mean across all nodes including self).
    pub fn collective_phi(&self) -> f32 {
        self.collective_phi
    }

    /// Current collective Phi divergence (variance — measures disagreement).
    pub fn collective_divergence(&self) -> f32 {
        self.collective_phi_divergence
    }

    /// Current adaptive sharing cadence.
    pub fn sharing_cadence(&self) -> u32 {
        self.sharing_cadence
    }

    /// Number of known consciousness peers.
    pub fn peer_count(&self) -> usize {
        self.peer_phi.len()
    }

    /// Active threat observations.
    pub fn threat_count(&self) -> usize {
        self.threat_observations.len()
    }

    /// Get all threat observations for broadcasting.
    pub fn threats(&self) -> &[ThreatObservation] {
        &self.threat_observations
    }

    // ── Private helpers ────────────────────────────────────────────────

    fn recompute_collective(&mut self) {
        let mut sum = self.local_phi;
        let mut count = 1u32;

        for state in self.peer_phi.values() {
            sum += state.phi * state.trust; // Trust-weighted
            count += 1;
        }

        self.collective_phi = sum / count as f32;

        // Variance
        let mut var_sum = (self.local_phi - self.collective_phi).powi(2);
        for state in self.peer_phi.values() {
            var_sum += (state.phi * state.trust - self.collective_phi).powi(2);
        }
        self.collective_phi_divergence = var_sum / count as f32;
    }

    fn adapt_cadence(&mut self) {
        // High divergence → share more frequently (converge faster)
        // Low divergence → share less frequently (save bandwidth)
        if self.collective_phi_divergence > PHI_DIVERGENCE_THRESHOLD {
            self.sharing_cadence = (self.sharing_cadence / 2).max(MIN_SHARING_CADENCE);
        } else if self.collective_phi_divergence < PHI_DIVERGENCE_THRESHOLD * 0.5 {
            self.sharing_cadence = (self.sharing_cadence * 2).min(MAX_SHARING_CADENCE);
        }
    }

    fn next_unbroadcast_threat(&self) -> Option<ThreatObservation> {
        // Return the highest-severity unbroadcast threat
        self.threat_observations
            .iter()
            .filter(|t| t.corroboration_count == 0) // Not yet corroborated = likely new
            .max_by(|a, b| {
                a.severity
                    .partial_cmp(&b.severity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STORE-AND-FORWARD — Dream-Consolidated Reconnection
// ═══════════════════════════════════════════════════════════════════════════════

/// Manages store-and-forward behavior for mesh nodes that experience
/// intermittent connectivity (solar nodes, mobile devices, disaster scenarios).
///
/// When a node reconnects after isolation, instead of dumping all accumulated
/// raw data, it runs dream consolidation to compress experiences into meaningful
/// wisdom patterns. This is biologically inspired: you don't relay every sensory
/// input — you relay what you learned from sleeping on it.
///
/// ## Integration
///
/// The Spore/Symthaea DreamEngine already performs counterfactual consolidation.
/// This module decides *when* to trigger consolidation and *what* to transmit
/// on reconnection.
pub struct StoreAndForward {
    /// Accumulated experiences during offline period.
    offline_buffer: VecDeque<OfflineExperience>,
    /// Maximum buffer size (experiences, not bytes).
    buffer_capacity: usize,
    /// Whether currently offline (no mesh connectivity).
    is_offline: bool,
    /// Cycle when connectivity was last lost.
    offline_since: Option<u64>,
    /// Number of reconnection events this session.
    reconnection_count: u32,
    /// Whether dream consolidation has been triggered for current buffer.
    consolidation_pending: bool,
}

/// A single offline experience — sensor event, governance action, or
/// consciousness state change recorded while disconnected.
#[derive(Debug, Clone)]
pub struct OfflineExperience {
    /// Cycle when recorded.
    pub cycle: u64,
    /// Type of experience.
    pub kind: OfflineExperienceKind,
    /// Salience score [0.0, 1.0] — high-salience experiences are prioritized.
    pub salience: f32,
}

/// Categories of offline experiences.
#[derive(Debug, Clone)]
pub enum OfflineExperienceKind {
    /// Sensor reading that exceeded a threshold.
    SensorAnomaly { sensor_id: String, value: f32 },
    /// Consciousness level crossed a boundary.
    ConsciousnessShift { from: f32, to: f32 },
    /// Moral evaluation with significant salience.
    MoralEvent { salience: f32 },
    /// Threat detected while offline.
    ThreatDetected { threat_type: u8, severity: f32 },
    /// Space situational awareness event (conjunction, debris alert, comm window).
    #[cfg(feature = "space-alerts")]
    SpaceEvent {
        /// Alert type name (e.g. "ConjunctionWarning", "DebrisProximity").
        alert_type: String,
        /// Severity [0.0, 1.0].
        severity: f32,
        /// Seconds until event (negative = past).
        time_to_event_seconds: f64,
    },
}

/// Result of dream consolidation on the offline buffer.
#[derive(Debug, Clone)]
pub struct ConsolidatedWisdom {
    /// Number of raw experiences consolidated.
    pub experiences_consolidated: usize,
    /// Duration of offline period (cycles).
    pub offline_duration: u64,
    /// Mean salience of consolidated experiences.
    pub mean_salience: f32,
    /// Summary patterns extracted (to transmit as WisdomPacket).
    pub patterns: Vec<ConsolidatedPattern>,
}

/// A single pattern extracted from dream consolidation.
#[derive(Debug, Clone)]
pub struct ConsolidatedPattern {
    /// Pattern type.
    pub kind: String,
    /// Confidence in this pattern [0.0, 1.0].
    pub confidence: f32,
    /// Compact representation (fits in a single mesh packet).
    pub data: Vec<u8>,
}

/// Default buffer capacity (experiences).
const STORE_FORWARD_BUFFER_CAPACITY: usize = 1000;
/// Salience threshold for keeping an experience in the buffer.
const SALIENCE_THRESHOLD: f32 = 0.3;

impl Default for StoreAndForward {
    fn default() -> Self {
        Self {
            offline_buffer: VecDeque::with_capacity(STORE_FORWARD_BUFFER_CAPACITY),
            buffer_capacity: STORE_FORWARD_BUFFER_CAPACITY,
            is_offline: false,
            offline_since: None,
            reconnection_count: 0,
            consolidation_pending: false,
        }
    }
}

impl StoreAndForward {
    /// Notify that connectivity has been lost.
    pub fn go_offline(&mut self, current_cycle: u64) {
        if !self.is_offline {
            self.is_offline = true;
            self.offline_since = Some(current_cycle);
            self.consolidation_pending = false;
        }
    }

    /// Notify that connectivity has been restored.
    /// Returns true if dream consolidation should be triggered.
    pub fn go_online(&mut self, current_cycle: u64) -> bool {
        if self.is_offline {
            self.is_offline = false;
            self.reconnection_count += 1;

            // Trigger consolidation if we accumulated significant experiences
            if self.offline_buffer.len() >= 10 {
                self.consolidation_pending = true;
                return true;
            }
        }
        false
    }

    /// Record an experience during offline period.
    /// Low-salience experiences are dropped to save memory.
    pub fn record(&mut self, experience: OfflineExperience) {
        if experience.salience < SALIENCE_THRESHOLD {
            return;
        }

        if self.offline_buffer.len() >= self.buffer_capacity {
            // Evict lowest-salience experience
            if let Some(min_idx) = self
                .offline_buffer
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.salience
                        .partial_cmp(&b.salience)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if self.offline_buffer[min_idx].salience < experience.salience {
                    self.offline_buffer.remove(min_idx);
                } else {
                    return; // New experience is lower salience than everything in buffer
                }
            }
        }

        self.offline_buffer.push_back(experience);
    }

    /// Perform dream consolidation on the offline buffer.
    ///
    /// Extracts meaningful patterns from raw experiences:
    /// - Sensor anomalies → trend summaries (mean, variance, duration)
    /// - Consciousness shifts → boundary crossing events
    /// - Moral events → aggregate moral landscape change
    /// - Threats → consolidated threat signatures
    ///
    /// Returns consolidated wisdom ready for mesh transmission.
    ///
    /// `max_items` caps how many buffered experiences are processed in this
    /// call.  Pass `usize::MAX` for unbounded (legacy behaviour).  When the
    /// buffer still has items after the batch, `consolidation_pending` stays
    /// `true` so the caller can continue draining on subsequent cycles.
    pub fn consolidate(&mut self, current_cycle: u64) -> ConsolidatedWisdom {
        self.consolidate_batch(current_cycle, usize::MAX)
    }

    /// Batch-limited variant of [`Self::consolidate`].
    pub fn consolidate_batch(
        &mut self,
        current_cycle: u64,
        max_items: usize,
    ) -> ConsolidatedWisdom {
        let offline_duration = self
            .offline_since
            .map(|since| current_cycle.saturating_sub(since))
            .unwrap_or(0);

        // Drain at most `max_items` from the front of the buffer.
        let drain_count = self.offline_buffer.len().min(max_items);
        let batch: Vec<OfflineExperience> = self.offline_buffer.drain(..drain_count).collect();

        let mean_salience = if batch.is_empty() {
            0.0
        } else {
            batch.iter().map(|e| e.salience).sum::<f32>() / batch.len() as f32
        };

        let mut patterns = Vec::new();

        // Consolidate sensor anomalies into trend summaries
        let sensor_events: Vec<_> = batch
            .iter()
            .filter(|e| matches!(e.kind, OfflineExperienceKind::SensorAnomaly { .. }))
            .collect();
        if !sensor_events.is_empty() {
            patterns.push(ConsolidatedPattern {
                kind: "sensor_trend".into(),
                confidence: (sensor_events.len() as f32 / 10.0).min(1.0),
                data: format!("anomalies:{}", sensor_events.len()).into_bytes(),
            });
        }

        // Consolidate consciousness shifts
        let consciousness_events: Vec<_> = batch
            .iter()
            .filter(|e| matches!(e.kind, OfflineExperienceKind::ConsciousnessShift { .. }))
            .collect();
        if !consciousness_events.is_empty() {
            patterns.push(ConsolidatedPattern {
                kind: "consciousness_trajectory".into(),
                confidence: 0.8,
                data: format!("shifts:{}", consciousness_events.len()).into_bytes(),
            });
        }

        // Consolidate threats into aggregate signature
        let threat_events: Vec<_> = batch
            .iter()
            .filter(|e| matches!(e.kind, OfflineExperienceKind::ThreatDetected { .. }))
            .collect();
        if !threat_events.is_empty() {
            let max_severity = threat_events
                .iter()
                .filter_map(|e| match &e.kind {
                    OfflineExperienceKind::ThreatDetected { severity, .. } => Some(*severity),
                    _ => None,
                })
                .fold(0.0f32, f32::max);
            patterns.push(ConsolidatedPattern {
                kind: "threat_summary".into(),
                confidence: max_severity,
                data: format!(
                    "threats:{},max_severity:{:.2}",
                    threat_events.len(),
                    max_severity
                )
                .into_bytes(),
            });
        }

        // Consolidate space events into alert summary
        #[cfg(feature = "space-alerts")]
        {
            let space_events: Vec<_> = batch
                .iter()
                .filter(|e| matches!(e.kind, OfflineExperienceKind::SpaceEvent { .. }))
                .collect();
            if !space_events.is_empty() {
                let max_severity = space_events
                    .iter()
                    .filter_map(|e| match &e.kind {
                        OfflineExperienceKind::SpaceEvent { severity, .. } => Some(*severity),
                        _ => None,
                    })
                    .fold(0.0f32, f32::max);
                patterns.push(ConsolidatedPattern {
                    kind: "space_alert_summary".into(),
                    confidence: max_severity,
                    data: format!(
                        "space_alerts:{},max_severity:{:.2}",
                        space_events.len(),
                        max_severity
                    )
                    .into_bytes(),
                });
            }
        }

        let experiences_consolidated = batch.len();
        // Only mark consolidation complete when the buffer is fully drained.
        if self.offline_buffer.is_empty() {
            self.consolidation_pending = false;
        }

        ConsolidatedWisdom {
            experiences_consolidated,
            offline_duration,
            mean_salience,
            patterns,
        }
    }

    /// Whether consolidation is pending (buffer has data, just reconnected).
    pub fn needs_consolidation(&self) -> bool {
        self.consolidation_pending
    }

    /// Number of buffered experiences.
    pub fn buffer_len(&self) -> usize {
        self.offline_buffer.len()
    }

    /// Whether currently offline.
    pub fn is_offline(&self) -> bool {
        self.is_offline
    }

    /// Total reconnection events this session.
    pub fn reconnection_count(&self) -> u32 {
        self.reconnection_count
    }
}
