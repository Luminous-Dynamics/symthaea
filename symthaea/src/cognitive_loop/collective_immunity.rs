// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Collective Immune Intelligence — Swarm defense coordination
//!
//! Extends the collective consciousness framework to coordinate
//! distributed immune responses across the Symthaea-Mycelix network.
//!
//! ## Key Mechanisms
//!
//! 1. **Threat severity from collective Phi**: Low collective coherence during
//!    threat → higher severity assessment (Woolley et al. 2010)
//! 2. **Epistemic blind spot detection**: Nodes identify gaps in their threat
//!    detection coverage and request help from peers
//! 3. **Federated threat learning**: Threat pattern HDVs shared during
//!    federated rounds for collective pattern recognition
//!
//! ## Cross-coupling Matrix
//!
//! ```text
//! SentinelManager ←→ SwarmManager (threat sharing, peer anomaly)
//! SentinelManager ←→ GovernanceManager (proposal rate limiting, tier tracking)
//! SwarmManager ←→ GovernanceManager (collective Phi, reputation)
//! SentinelManager → SafetyAgent (threat level → safety escalation)
//! SafetyAgent → GuardianState (safety → physical posture)
//! ```

use serde::{Deserialize, Serialize};

/// Collective immune state aggregated across the swarm.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectiveImmuneState {
    /// Aggregate threat level across all connected sentinels (0.0–1.0).
    pub swarm_threat_level: f32,
    /// Number of peers reporting active threats.
    pub peers_under_threat: usize,
    /// Collective Phi during threat (lower = more severe).
    pub collective_phi_during_threat: f64,
    /// Threat severity adjusted by collective coherence.
    pub coherence_adjusted_severity: f32,
    /// Number of unique threat kinds detected across the swarm.
    pub distinct_threat_kinds: usize,
    /// Whether the swarm is in collective immune response mode.
    pub immune_response_active: bool,
    /// Blind spots: threat kinds not covered by local sentinel.
    pub blind_spots: Vec<String>,
    /// Number of federated threat patterns shared this round.
    pub patterns_shared: usize,
    /// Domains where epistemic coverage is lacking.
    #[cfg(feature = "epistemic")]
    #[serde(default)]
    pub epistemic_blind_spots: Vec<String>,
    /// Fraction of network with calibrated epistemic confidence (0.0-1.0).
    #[cfg(feature = "epistemic")]
    #[serde(default)]
    pub epistemic_herd_immunity: f32,
    /// Network-level epistemic health: collective_phi * (1 - echo_chamber_risk).
    #[cfg(feature = "epistemic")]
    #[serde(default)]
    pub network_epistemic_health: f64,
    /// Echo chamber risk: high individual Phi + low collective Phi = fragmentation.
    #[cfg(feature = "epistemic")]
    #[serde(default)]
    pub echo_chamber_risk: f32,
}

impl CollectiveImmuneState {
    /// Compute coherence-adjusted threat severity.
    ///
    /// When collective Phi is low during a threat, severity is amplified:
    /// `adjusted = raw_threat × (2.0 - collective_phi)`
    ///
    /// Rationale: low collective coherence means the swarm is confused/
    /// fragmented — the threat is harder to respond to, so it's effectively
    /// more severe.
    ///
    /// Basis: Woolley et al. (2010) — collective intelligence correlates
    /// with group coherence, not individual ability.
    pub fn compute_coherence_adjusted_severity(raw_threat_level: f32, collective_phi: f64) -> f32 {
        let phi_factor = (2.0 - collective_phi.clamp(0.0, 1.0)) as f32;
        (raw_threat_level * phi_factor).clamp(0.0, 1.0)
    }

    /// Detect epistemic blind spots by comparing local threat detection
    /// capabilities against known swarm-wide threat kinds.
    ///
    /// Returns threat kinds that the local sentinel has never detected
    /// but other peers have reported.
    pub fn detect_blind_spots(
        local_threat_kinds: &[String],
        swarm_threat_kinds: &[String],
    ) -> Vec<String> {
        swarm_threat_kinds
            .iter()
            .filter(|k| !local_threat_kinds.contains(k))
            .cloned()
            .collect()
    }

    /// Determine whether the swarm should enter collective immune response.
    ///
    /// Triggers when:
    /// - 30%+ of peers report active threats, OR
    /// - Any single threat severity > 0.7, OR
    /// - Coherence-adjusted severity > 0.6
    pub fn should_activate_immune_response(
        peers_under_threat: usize,
        total_peers: usize,
        max_threat_severity: f32,
        coherence_adjusted_severity: f32,
    ) -> bool {
        if total_peers == 0 {
            return false;
        }
        let threat_ratio = peers_under_threat as f32 / total_peers as f32;
        threat_ratio > 0.3 || max_threat_severity > 0.7 || coherence_adjusted_severity > 0.6
    }

    /// Update the collective immune state from swarm data.
    pub fn update(
        &mut self,
        local_threat_level: f32,
        peer_threat_reports: &[(String, f32)], // (peer_id, threat_level)
        collective_phi: f64,
        total_peers: usize,
        local_threat_kinds: &[String],
        swarm_threat_kinds: &[String],
    ) {
        self.peers_under_threat = peer_threat_reports
            .iter()
            .filter(|(_, level)| *level > 0.1)
            .count();

        let max_peer_threat = peer_threat_reports
            .iter()
            .map(|(_, level)| *level)
            .fold(0.0f32, f32::max);

        self.swarm_threat_level = local_threat_level.max(max_peer_threat);
        self.collective_phi_during_threat = collective_phi;

        self.coherence_adjusted_severity =
            Self::compute_coherence_adjusted_severity(self.swarm_threat_level, collective_phi);

        self.distinct_threat_kinds = swarm_threat_kinds.len();

        self.immune_response_active = Self::should_activate_immune_response(
            self.peers_under_threat,
            total_peers,
            self.swarm_threat_level,
            self.coherence_adjusted_severity,
        );

        self.blind_spots = Self::detect_blind_spots(local_threat_kinds, swarm_threat_kinds);

        #[cfg(feature = "epistemic")]
        {
            self.epistemic_blind_spots = Self::detect_epistemic_blind_spots(swarm_threat_kinds);
            self.epistemic_herd_immunity =
                Self::compute_epistemic_herd_immunity(peer_threat_reports, total_peers);
        }
    }

    /// Detect epistemic coverage gaps: domains present in swarm threats
    /// that relate to epistemic integrity (EpistemicThreat, ConsciousnessManipulation).
    #[cfg(feature = "epistemic")]
    pub fn detect_epistemic_blind_spots(swarm_threat_kinds: &[String]) -> Vec<String> {
        const EPISTEMIC_DOMAINS: &[&str] = &[
            "EpistemicThreat",
            "ConsciousnessManipulation",
            "GradientFingerprint",
        ];
        swarm_threat_kinds
            .iter()
            .filter(|k| EPISTEMIC_DOMAINS.contains(&k.as_str()))
            .cloned()
            .collect()
    }

    /// Compute fraction of peers with calibrated epistemic confidence.
    ///
    /// A peer is considered "calibrated" if its threat level is moderate
    /// (between 0.05 and 0.6) — neither oblivious (0.0) nor panic-level.
    /// This is a proxy for epistemic health: the node is detecting real
    /// threats without over-reporting.
    #[cfg(feature = "epistemic")]
    pub fn compute_epistemic_herd_immunity(
        peer_threat_reports: &[(String, f32)],
        total_peers: usize,
    ) -> f32 {
        if total_peers == 0 {
            return 0.0;
        }
        let calibrated = peer_threat_reports
            .iter()
            .filter(|(_, level)| *level > 0.05 && *level < 0.6)
            .count();
        (calibrated as f32 / total_peers as f32).clamp(0.0, 1.0)
    }

    /// Compute echo chamber risk and network epistemic health.
    ///
    /// Echo chamber: individual nodes have high Phi (confident in their own beliefs)
    /// but collective Phi is low (the network isn't integrating across boundaries).
    ///
    /// Network epistemic health = collective_phi * (1.0 - echo_chamber_risk)
    ///
    /// Reference: arXiv:2512.15011 — epistemic diversity across LMs mitigates knowledge collapse
    #[cfg(feature = "epistemic")]
    pub fn compute_echo_chamber_risk(&mut self, individual_phi_mean: f64, collective_phi: f64) {
        // High individual Phi + low collective Phi = echo chamber
        let individual = individual_phi_mean.clamp(0.0, 1.0);
        let collective = collective_phi.clamp(0.0, 1.0);

        self.echo_chamber_risk = if individual > 0.01 {
            ((individual - collective) / individual).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };

        self.network_epistemic_health = collective * (1.0 - self.echo_chamber_risk as f64);
    }

    /// Neuromodulator adjustments for collective immune response.
    ///
    /// Returns (ne_delta, cortisol_delta, oxytocin_delta):
    /// - NE: vigilance proportional to swarm threat
    /// - Cortisol: stress proportional to coherence-adjusted severity
    /// - Oxytocin: social bonding boost during collective response (in-group solidarity)
    ///
    /// Basis: De Dreu et al. (2010) — oxytocin promotes in-group cooperation under threat.
    pub fn neuromod_deltas(&self) -> (f64, f64, f64) {
        if !self.immune_response_active {
            return (0.0, 0.0, 0.0);
        }
        let ne = self.swarm_threat_level as f64 * 0.04;
        let cortisol = self.coherence_adjusted_severity as f64 * 0.03;
        // Oxytocin boost during collective response — in-group solidarity
        let oxytocin = 0.02;
        (ne, cortisol, oxytocin)
    }
}

/// Cross-coupling: compute safety level adjustment from sentinel + collective state.
///
/// If sentinel threat_level is high AND collective immune response is active,
/// recommend escalating the safety level. Only escalates, never de-escalates
/// (safety agent handles recovery).
pub fn recommend_safety_escalation(
    current_safety: crate::safety::SafetyLevel,
    sentinel_threat_level: f32,
    immune_state: &CollectiveImmuneState,
) -> Option<crate::safety::SafetyLevel> {
    use crate::safety::SafetyLevel;

    // Only escalate, never de-escalate (safety agent handles recovery)
    if sentinel_threat_level > 0.7 && immune_state.immune_response_active {
        if current_safety < SafetyLevel::Orange {
            return Some(SafetyLevel::Orange);
        }
    }
    if sentinel_threat_level > 0.5 || immune_state.coherence_adjusted_severity > 0.5 {
        if current_safety < SafetyLevel::Yellow {
            return Some(SafetyLevel::Yellow);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_adjusted_severity_high_phi() {
        // High collective Phi → severity unchanged
        let adj = CollectiveImmuneState::compute_coherence_adjusted_severity(0.5, 1.0);
        assert!((adj - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_adjusted_severity_low_phi() {
        // Low collective Phi → severity amplified (up to 2×)
        let adj = CollectiveImmuneState::compute_coherence_adjusted_severity(0.5, 0.0);
        assert!((adj - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_adjusted_severity_clamped() {
        let adj = CollectiveImmuneState::compute_coherence_adjusted_severity(0.8, 0.0);
        assert!(adj <= 1.0);
    }

    #[test]
    fn test_blind_spot_detection() {
        let local = vec!["ProposalFlood".to_string(), "TimingAnomaly".to_string()];
        let swarm = vec![
            "ProposalFlood".to_string(),
            "VoteClustering".to_string(),
            "GradientFingerprint".to_string(),
        ];
        let spots = CollectiveImmuneState::detect_blind_spots(&local, &swarm);
        assert_eq!(spots.len(), 2);
        assert!(spots.contains(&"VoteClustering".to_string()));
        assert!(spots.contains(&"GradientFingerprint".to_string()));
    }

    #[test]
    fn test_immune_response_activation_by_ratio() {
        // 40% of peers under threat → activate
        assert!(CollectiveImmuneState::should_activate_immune_response(
            4, 10, 0.3, 0.3
        ));
        // 10% → don't activate
        assert!(!CollectiveImmuneState::should_activate_immune_response(
            1, 10, 0.3, 0.3
        ));
    }

    #[test]
    fn test_immune_response_activation_by_severity() {
        // High severity → activate regardless of peer count
        assert!(CollectiveImmuneState::should_activate_immune_response(
            0, 10, 0.8, 0.3
        ));
    }

    #[test]
    fn test_immune_response_activation_by_coherence() {
        // High coherence-adjusted severity → activate
        assert!(CollectiveImmuneState::should_activate_immune_response(
            0, 10, 0.3, 0.7
        ));
    }

    #[test]
    fn test_immune_response_zero_peers() {
        assert!(!CollectiveImmuneState::should_activate_immune_response(
            0, 0, 0.8, 0.8
        ));
    }

    #[test]
    fn test_neuromod_deltas_inactive() {
        let state = CollectiveImmuneState::default();
        let (ne, cortisol, oxy) = state.neuromod_deltas();
        assert_eq!(ne, 0.0);
        assert_eq!(cortisol, 0.0);
        assert_eq!(oxy, 0.0);
    }

    #[test]
    fn test_neuromod_deltas_active() {
        let state = CollectiveImmuneState {
            swarm_threat_level: 0.8,
            coherence_adjusted_severity: 0.7,
            immune_response_active: true,
            ..Default::default()
        };
        let (ne, cortisol, oxy) = state.neuromod_deltas();
        assert!(ne > 0.0);
        assert!(cortisol > 0.0);
        assert!(oxy > 0.0);
    }

    #[test]
    fn test_safety_escalation_high_threat() {
        let state = CollectiveImmuneState {
            immune_response_active: true,
            coherence_adjusted_severity: 0.8,
            ..Default::default()
        };
        let rec = recommend_safety_escalation(crate::safety::SafetyLevel::Green, 0.8, &state);
        assert_eq!(rec, Some(crate::safety::SafetyLevel::Orange));
    }

    #[test]
    fn test_safety_escalation_moderate_threat() {
        let state = CollectiveImmuneState {
            coherence_adjusted_severity: 0.6,
            ..Default::default()
        };
        let rec = recommend_safety_escalation(crate::safety::SafetyLevel::Green, 0.6, &state);
        assert_eq!(rec, Some(crate::safety::SafetyLevel::Yellow));
    }

    #[test]
    fn test_safety_no_deescalation() {
        let state = CollectiveImmuneState::default();
        let rec = recommend_safety_escalation(crate::safety::SafetyLevel::Orange, 0.0, &state);
        assert_eq!(rec, None); // Never de-escalate
    }

    #[test]
    fn test_update_state() {
        let mut state = CollectiveImmuneState::default();
        let reports = vec![
            ("peer1".to_string(), 0.6f32),
            ("peer2".to_string(), 0.8f32),
            ("peer3".to_string(), 0.0f32),
        ];
        state.update(
            0.5,
            &reports,
            0.7,
            10,
            &["ProposalFlood".to_string()],
            &["ProposalFlood".to_string(), "VoteClustering".to_string()],
        );

        assert_eq!(state.peers_under_threat, 2);
        assert!((state.swarm_threat_level - 0.8).abs() < 1e-6);
        assert_eq!(state.distinct_threat_kinds, 2);
        assert_eq!(state.blind_spots.len(), 1);
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_herd_immunity_computation() {
        // 3 calibrated peers (threat 0.1-0.5), 2 non-calibrated (0.0 and 0.9)
        let reports = vec![
            ("p1".to_string(), 0.1f32),
            ("p2".to_string(), 0.3),
            ("p3".to_string(), 0.5),
            ("p4".to_string(), 0.0),
            ("p5".to_string(), 0.9),
        ];
        let herd = CollectiveImmuneState::compute_epistemic_herd_immunity(&reports, 5);
        assert!((herd - 0.6).abs() < 1e-6); // 3/5 = 0.6
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_herd_immunity_zero_peers() {
        let herd = CollectiveImmuneState::compute_epistemic_herd_immunity(&[], 0);
        assert_eq!(herd, 0.0);
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_blind_spot_detection() {
        let swarm_kinds = vec![
            "ProposalFlood".to_string(),
            "EpistemicThreat".to_string(),
            "GradientFingerprint".to_string(),
            "TimingAnomaly".to_string(),
        ];
        let spots = CollectiveImmuneState::detect_epistemic_blind_spots(&swarm_kinds);
        assert_eq!(spots.len(), 2);
        assert!(spots.contains(&"EpistemicThreat".to_string()));
        assert!(spots.contains(&"GradientFingerprint".to_string()));
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_update_populates_epistemic_fields() {
        let mut state = CollectiveImmuneState::default();
        let reports = vec![
            ("p1".to_string(), 0.2f32),
            ("p2".to_string(), 0.4),
            ("p3".to_string(), 0.0),
        ];
        state.update(
            0.3,
            &reports,
            0.8,
            3,
            &["ProposalFlood".to_string()],
            &["ProposalFlood".to_string(), "EpistemicThreat".to_string()],
        );
        assert_eq!(state.epistemic_blind_spots, vec!["EpistemicThreat"]);
        // 2 calibrated out of 3 total peers
        assert!((state.epistemic_herd_immunity - 2.0 / 3.0).abs() < 1e-5);
    }
}
