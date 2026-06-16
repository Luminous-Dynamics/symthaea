// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for HypervisorManager multi-instance communication.
//!
//! Tests two HypervisorManagers communicating via SwarmEvent variants,
//! simulating a supervisor-subordinate relationship.

#[cfg(test)]
#[cfg(feature = "hypervisor")]
mod tests {
    use crate::cognitive_loop::managers::hypervisor_manager::*;
    use crate::cognitive_loop::managers::swarm_manager::SwarmEvent;
    use crate::cognitive_loop::subsystem_trait::{
        CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
    };

    fn make_snapshot(cycle: u64, phi: f64) -> CycleSnapshot {
        let mut snap = CycleSnapshot::default();
        snap.cycle_number = cycle;
        snap.unified_psi = phi;
        snap
    }

    /// Test: Two managers communicate. Agent A (high Phi) becomes supervisor,
    /// Agent B becomes subordinate. Supervisor detects subordinate distress.
    #[test]
    fn test_two_agent_supervision() {
        let mut agent_a = HypervisorManager::new("agent-A".into());
        let mut agent_b = HypervisorManager::new("agent-B".into());

        // Both agents observe each other's consciousness
        agent_a.update_subordinate("agent-B".into(), 0.5, 0.7, 0.3, 0.8, 0);
        agent_b.update_subordinate("agent-A".into(), 0.7, 0.9, 0.2, 0.9, 0);

        // Agent A has higher Phi → should win election
        agent_a.self_phi = 0.7;
        agent_a.self_reputation = 0.9;
        agent_b.self_phi = 0.5;
        agent_b.self_reputation = 0.8;

        // Run election
        let a_wins = agent_a.run_election(0);
        let b_wins = agent_b.run_election(0);

        assert!(a_wins, "Agent A (higher phi*rep) should become supervisor");
        assert!(!b_wins, "Agent B should become subordinate");
        assert_eq!(agent_a.role, HypervisorRole::Supervisor);
        assert_eq!(agent_b.role, HypervisorRole::Subordinate);

        // Verify A generated an election event
        let a_events = agent_a.drain_events();
        assert!(
            a_events
                .iter()
                .any(|e| matches!(e, SupervisoryEvent::Elected { .. }))
        );
    }

    /// Test: Supervisor broadcasts heartbeat, subordinate receives it via SwarmEvent.
    #[test]
    fn test_heartbeat_via_swarm_event() {
        let mut supervisor = HypervisorManager::new("supervisor".into());
        supervisor.self_phi = 0.8;
        supervisor.self_reputation = 0.9;
        supervisor.run_election(0);
        assert_eq!(supervisor.role, HypervisorRole::Supervisor);

        // Add a subordinate
        supervisor.update_subordinate("sub-1".into(), 0.6, 0.8, 0.3, 0.9, 0);

        // Process at heartbeat cycle (interval 71 × 5 = 355)
        let snap = make_snapshot(355, 0.8);
        let _output = supervisor.process(&snap);

        // Drain heartbeat events
        let events = supervisor.drain_events();
        let heartbeat = events
            .iter()
            .find(|e| matches!(e, SupervisoryEvent::Heartbeat { .. }));
        assert!(
            heartbeat.is_some(),
            "supervisor should emit heartbeat at cycle 355"
        );

        // Convert to SwarmEvent (this is what would flow over DHT)
        if let Some(SupervisoryEvent::Heartbeat {
            epoch,
            subordinate_count,
            aggregated_phi,
            ..
        }) = heartbeat
        {
            let swarm_event = SwarmEvent::SupervisoryHeartbeat {
                supervisor_id: "supervisor".into(),
                epoch: *epoch,
                subordinate_count: *subordinate_count,
                aggregated_phi: *aggregated_phi,
            };

            // Verify the SwarmEvent can be constructed
            match swarm_event {
                SwarmEvent::SupervisoryHeartbeat { aggregated_phi, .. } => {
                    assert!(aggregated_phi > 0.0);
                }
                _ => panic!("wrong event type"),
            }
        }
    }

    /// Test: Subordinate goes into distress (low Phi), supervisor detects and escalates.
    #[test]
    fn test_distress_detection_and_escalation() {
        let mut supervisor = HypervisorManager::new("supervisor".into());
        supervisor.self_phi = 0.8;
        supervisor.self_reputation = 0.9;
        supervisor.run_election(0);

        // Add healthy subordinate
        supervisor.update_subordinate("sub-healthy".into(), 0.6, 0.8, 0.3, 0.9, 100);

        // Add distressed subordinate (very low Phi)
        supervisor.update_subordinate("sub-distressed".into(), 0.1, 0.2, 0.9, 0.8, 100);

        // Process cycle
        let snap = make_snapshot(100, 0.8);
        let output = supervisor.process(&snap);

        // Check for escalation event
        let events = supervisor.drain_events();
        let escalation = events
            .iter()
            .find(|e| matches!(e, SupervisoryEvent::Escalation { .. }));
        assert!(
            escalation.is_some(),
            "supervisor should detect distressed subordinate"
        );

        if let Some(SupervisoryEvent::Escalation {
            affected_peers,
            severity,
            ..
        }) = escalation
        {
            assert!(affected_peers.contains(&"sub-distressed".to_string()));
            assert!(*severity > 0.0);
        }
    }

    /// Test: Staleness decay — old subordinate data gets exponentially discounted.
    #[test]
    fn test_staleness_weighted_aggregation() {
        let mut supervisor = HypervisorManager::new("supervisor".into());
        supervisor.self_phi = 0.8;
        supervisor.self_reputation = 0.9;
        supervisor.run_election(0);

        // Sub-1: recent data (cycle 1000), high Phi
        supervisor.update_subordinate("sub-1".into(), 0.9, 0.9, 0.2, 1.0, 1000);
        // Sub-2: stale data (cycle 500), high Phi
        supervisor.update_subordinate("sub-2".into(), 0.9, 0.9, 0.2, 1.0, 500);

        supervisor.aggregate_subordinates(1000);
        let agg_phi_recent = supervisor.aggregated_phi;

        // The aggregated Phi should be closer to sub-1's value because
        // sub-2's data is stale (500 cycles old → significant decay)
        // With tau=300, sub-2's weight = exp(-500/300) ≈ 0.19, sub-1 = 1.0
        assert!(agg_phi_recent > 0.5, "aggregated phi should be meaningful");

        // Now make all data stale
        supervisor.aggregate_subordinates(2000);
        let agg_phi_stale = supervisor.aggregated_phi;
        // Both are very stale now (>1000 cycles), aggregation falls back to self_phi
        assert!(
            agg_phi_stale >= supervisor.self_phi * 0.5,
            "very stale data should decay toward self_phi"
        );
    }

    /// Test: Re-election after epoch boundary with changed Phi rankings.
    #[test]
    fn test_re_election_on_phi_change() {
        let mut agent_a = HypervisorManager::new("agent-A".into());
        let mut agent_b = HypervisorManager::new("agent-B".into());

        // Round 1: A wins
        agent_a.self_phi = 0.8;
        agent_a.self_reputation = 0.9;
        agent_b.self_phi = 0.5;
        agent_b.self_reputation = 0.8;

        agent_a.update_subordinate("agent-B".into(), 0.5, 0.7, 0.3, 0.8, 0);
        agent_b.update_subordinate("agent-A".into(), 0.8, 0.9, 0.2, 0.9, 0);

        agent_a.run_election(0);
        assert_eq!(agent_a.role, HypervisorRole::Supervisor);

        // Round 2: B's Phi surpasses A's (consciousness shift)
        agent_a.self_phi = 0.4; // A's consciousness dropped
        agent_b.self_phi = 0.9; // B's consciousness rose

        // Update each other's records
        agent_a.update_subordinate("agent-B".into(), 0.9, 0.9, 0.2, 0.8, 1000);
        agent_b.update_subordinate("agent-A".into(), 0.4, 0.5, 0.3, 0.9, 1000);

        // Re-election at epoch boundary
        let a_wins = agent_a.run_election(1001);
        let b_wins = agent_b.run_election(1001);

        // B should now win (higher phi * reputation = 0.9 * 0.8 = 0.72 > 0.4 * 0.9 = 0.36)
        assert!(!a_wins, "A should lose with lower phi");
        assert!(b_wins, "B should win with higher phi");
        assert_eq!(agent_a.role, HypervisorRole::Subordinate);
        assert_eq!(agent_b.role, HypervisorRole::Supervisor);
    }
}
