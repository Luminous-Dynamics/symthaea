//! Fail-closed readiness evaluation for Luminous networking roles.
//!
//! Metrics are cumulative for the current process. Operators should evaluate a
//! fresh process or compare snapshots when they need windowed SLOs; this module
//! intentionally does not hide historical queue exhaustion or invalid traffic.

use crate::{
    direct::{DirectHealthSnapshot, DirectPeerPolicy},
    networking::{SocketMetricsSnapshot, SocketState},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransportReadinessProfile {
    pub require_gossip: bool,
    pub minimum_gossip_neighbors: usize,
    pub require_direct: bool,
    pub minimum_direct_peers: usize,
    pub require_datagrams_for_all_direct_peers: bool,
    pub require_pinned_direct_peers: bool,
    pub maximum_gossip_send_errors: u64,
    pub maximum_durable_queue_exhaustion: u64,
    pub maximum_direct_event_queue_exhaustion: u64,
    pub maximum_invalid_direct_packets: u64,
    pub maximum_idempotent_conflicts: u64,
}

impl TransportReadinessProfile {
    /// Governance/proof control plane. Direct real-time connectivity is optional.
    pub const CONTROL_PLANE: Self = Self {
        require_gossip: true,
        minimum_gossip_neighbors: 1,
        require_direct: false,
        minimum_direct_peers: 0,
        require_datagrams_for_all_direct_peers: false,
        require_pinned_direct_peers: false,
        maximum_gossip_send_errors: 0,
        maximum_durable_queue_exhaustion: 0,
        maximum_direct_event_queue_exhaustion: 0,
        maximum_invalid_direct_packets: 0,
        maximum_idempotent_conflicts: 0,
    };

    /// Multiplayer/robotics role with authenticated pinned peers and datagrams.
    pub const REALTIME_DATA_PLANE: Self = Self {
        require_gossip: false,
        minimum_gossip_neighbors: 0,
        require_direct: true,
        minimum_direct_peers: 1,
        require_datagrams_for_all_direct_peers: true,
        require_pinned_direct_peers: true,
        maximum_gossip_send_errors: 0,
        maximum_durable_queue_exhaustion: 0,
        maximum_direct_event_queue_exhaustion: 0,
        maximum_invalid_direct_packets: 0,
        maximum_idempotent_conflicts: 0,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessFailure {
    GossipSnapshotMissing,
    GossipNotActive { state: SocketState },
    GossipNeighborsBelowMinimum { actual: usize, minimum: usize },
    GossipSendErrors { actual: u64, maximum: u64 },
    DurableQueueExhausted { actual: u64, maximum: u64 },
    DirectSnapshotMissing,
    DirectPeersBelowMinimum { actual: usize, minimum: usize },
    DirectPeerWithoutDatagrams,
    DirectPeerPolicyNotPinned { actual: DirectPeerPolicy },
    DirectEventQueueExhausted { actual: u64, maximum: u64 },
    InvalidDirectPackets { actual: u64, maximum: u64 },
    IdempotentOperationConflicts { actual: u64, maximum: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessWarning {
    BestEffortGossipDropped { count: u64 },
    GossipLagged { count: u64 },
    DirectDatagramsDropped { count: u64 },
    DirectPacketsRateLimited { count: u64 },
    IdempotentOperationStillInProgress { count: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportReadinessReport {
    pub ready: bool,
    pub failures: Vec<ReadinessFailure>,
    pub warnings: Vec<ReadinessWarning>,
}

impl TransportReadinessReport {
    pub fn evaluate(
        profile: TransportReadinessProfile,
        gossip: Option<(&SocketState, SocketMetricsSnapshot)>,
        direct: Option<&DirectHealthSnapshot>,
    ) -> Self {
        let mut failures = Vec::new();
        let mut warnings = Vec::new();

        if profile.require_gossip || gossip.is_some() {
            match gossip {
                None => failures.push(ReadinessFailure::GossipSnapshotMissing),
                Some((state, metrics)) => {
                    match state {
                        SocketState::Active { neighbors }
                            if *neighbors >= profile.minimum_gossip_neighbors => {}
                        SocketState::Active { neighbors } => {
                            failures.push(ReadinessFailure::GossipNeighborsBelowMinimum {
                                actual: *neighbors,
                                minimum: profile.minimum_gossip_neighbors,
                            })
                        }
                        _ if profile.require_gossip => {
                            failures.push(ReadinessFailure::GossipNotActive {
                                state: state.clone(),
                            })
                        }
                        _ => {}
                    }
                    if metrics.send_errors > profile.maximum_gossip_send_errors {
                        failures.push(ReadinessFailure::GossipSendErrors {
                            actual: metrics.send_errors,
                            maximum: profile.maximum_gossip_send_errors,
                        });
                    }
                    if metrics.durable_queue_full > profile.maximum_durable_queue_exhaustion {
                        failures.push(ReadinessFailure::DurableQueueExhausted {
                            actual: metrics.durable_queue_full,
                            maximum: profile.maximum_durable_queue_exhaustion,
                        });
                    }
                    if metrics.best_effort_dropped > 0 {
                        warnings.push(ReadinessWarning::BestEffortGossipDropped {
                            count: metrics.best_effort_dropped,
                        });
                    }
                    if metrics.gossip_lagged > 0 {
                        warnings.push(ReadinessWarning::GossipLagged {
                            count: metrics.gossip_lagged,
                        });
                    }
                }
            }
        }

        if profile.require_direct || direct.is_some() {
            match direct {
                None => failures.push(ReadinessFailure::DirectSnapshotMissing),
                Some(snapshot) => {
                    if snapshot.peers.len() < profile.minimum_direct_peers {
                        failures.push(ReadinessFailure::DirectPeersBelowMinimum {
                            actual: snapshot.peers.len(),
                            minimum: profile.minimum_direct_peers,
                        });
                    }
                    if profile.require_datagrams_for_all_direct_peers
                        && snapshot.peers.iter().any(|peer| !peer.datagrams_supported)
                    {
                        failures.push(ReadinessFailure::DirectPeerWithoutDatagrams);
                    }
                    if profile.require_pinned_direct_peers
                        && snapshot.peer_policy != DirectPeerPolicy::PinnedOnly
                    {
                        failures.push(ReadinessFailure::DirectPeerPolicyNotPinned {
                            actual: snapshot.peer_policy,
                        });
                    }
                    if snapshot.metrics.event_queue_full
                        > profile.maximum_direct_event_queue_exhaustion
                    {
                        failures.push(ReadinessFailure::DirectEventQueueExhausted {
                            actual: snapshot.metrics.event_queue_full,
                            maximum: profile.maximum_direct_event_queue_exhaustion,
                        });
                    }
                    if snapshot.metrics.invalid_packets > profile.maximum_invalid_direct_packets {
                        failures.push(ReadinessFailure::InvalidDirectPackets {
                            actual: snapshot.metrics.invalid_packets,
                            maximum: profile.maximum_invalid_direct_packets,
                        });
                    }
                    if snapshot.metrics.idempotent_conflicts > profile.maximum_idempotent_conflicts
                    {
                        failures.push(ReadinessFailure::IdempotentOperationConflicts {
                            actual: snapshot.metrics.idempotent_conflicts,
                            maximum: profile.maximum_idempotent_conflicts,
                        });
                    }
                    if snapshot.metrics.datagrams_dropped > 0 {
                        warnings.push(ReadinessWarning::DirectDatagramsDropped {
                            count: snapshot.metrics.datagrams_dropped,
                        });
                    }
                    if snapshot.metrics.rate_limited > 0 {
                        warnings.push(ReadinessWarning::DirectPacketsRateLimited {
                            count: snapshot.metrics.rate_limited,
                        });
                    }
                    if snapshot.metrics.idempotent_in_progress > 0 {
                        warnings.push(ReadinessWarning::IdempotentOperationStillInProgress {
                            count: snapshot.metrics.idempotent_in_progress,
                        });
                    }
                }
            }
        }

        Self {
            ready: failures.is_empty(),
            failures,
            warnings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{
        DirectConnectionOrigin, DirectMetricsSnapshot, DirectTransportCapabilities,
        PeerConnectionSnapshot,
    };
    use iroh::SecretKey;
    use uuid::Uuid;

    fn direct_snapshot(datagrams: bool, policy: DirectPeerPolicy) -> DirectHealthSnapshot {
        let local = SecretKey::from_bytes(&[51u8; 32]).public();
        let peer = SecretKey::from_bytes(&[52u8; 32]).public();
        DirectHealthSnapshot {
            local_endpoint: local,
            session_id: Uuid::from_u128(1),
            peers: vec![PeerConnectionSnapshot {
                peer,
                origin: DirectConnectionOrigin::Outgoing,
                stable_id: 1,
                datagrams_supported: datagrams,
                maximum_datagram_size: datagrams.then_some(1_200),
                datagram_send_buffer_space: 64 * 1024,
            }],
            peer_policy: policy,
            metrics: DirectMetricsSnapshot::default(),
            capabilities: DirectTransportCapabilities::DATA_PLANE_V1,
        }
    }

    #[test]
    fn realtime_profile_requires_pinned_datagram_peers() {
        let open = direct_snapshot(false, DirectPeerPolicy::AnyAuthenticated);
        let report = TransportReadinessReport::evaluate(
            TransportReadinessProfile::REALTIME_DATA_PLANE,
            None,
            Some(&open),
        );
        assert!(!report.ready);
        assert!(
            report
                .failures
                .contains(&ReadinessFailure::DirectPeerWithoutDatagrams)
        );
        assert!(
            report.failures.iter().any(|failure| matches!(
                failure,
                ReadinessFailure::DirectPeerPolicyNotPinned { .. }
            ))
        );

        let pinned = direct_snapshot(true, DirectPeerPolicy::PinnedOnly);
        assert!(
            TransportReadinessReport::evaluate(
                TransportReadinessProfile::REALTIME_DATA_PLANE,
                None,
                Some(&pinned),
            )
            .ready
        );
    }

    #[test]
    fn control_plane_fails_closed_on_durable_queue_exhaustion() {
        let state = SocketState::Active { neighbors: 1 };
        let metrics = SocketMetricsSnapshot {
            durable_queue_full: 1,
            ..SocketMetricsSnapshot::default()
        };
        let report = TransportReadinessReport::evaluate(
            TransportReadinessProfile::CONTROL_PLANE,
            Some((&state, metrics)),
            None,
        );
        assert!(!report.ready);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| matches!(failure, ReadinessFailure::DurableQueueExhausted { .. }))
        );
    }
    #[test]
    fn idempotency_conflict_fails_realtime_readiness() {
        let mut snapshot = direct_snapshot(true, DirectPeerPolicy::PinnedOnly);
        snapshot.metrics.idempotent_conflicts = 1;
        let report = TransportReadinessReport::evaluate(
            TransportReadinessProfile::REALTIME_DATA_PLANE,
            None,
            Some(&snapshot),
        );
        assert!(!report.ready);
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            ReadinessFailure::IdempotentOperationConflicts {
                actual: 1,
                maximum: 0
            }
        )));
    }
}
