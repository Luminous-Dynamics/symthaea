use std::time::Duration;

use symthaea_boot_ecology_live::{
    DiagnosticFloor, LiveAdapterError, LiveEcologyReducer, SemanticBootAnchor, VisualAccent,
};
use symthaea_boot_protocol::{
    BootDomain, BootHealth, BootPhase, BootSnapshot, DomainSnapshot, DomainState,
};

const PHASES: [BootPhase; 10] = [
    BootPhase::Kernel,
    BootPhase::Initrd,
    BootPhase::Storage,
    BootPhase::Filesystems,
    BootPhase::Security,
    BootPhase::Network,
    BootPhase::Services,
    BootPhase::Graphics,
    BootPhase::Session,
    BootPhase::Ready,
];

const HEALTHS: [BootHealth; 5] = [
    BootHealth::Normal,
    BootHealth::Unknown,
    BootHealth::Delayed,
    BootHealth::Degraded,
    BootHealth::Failed,
];

const DOMAINS: [BootDomain; 9] = [
    BootDomain::Kernel,
    BootDomain::Initrd,
    BootDomain::Storage,
    BootDomain::Filesystems,
    BootDomain::Security,
    BootDomain::Network,
    BootDomain::Services,
    BootDomain::Graphics,
    BootDomain::Session,
];

const DOMAIN_STATES: [DomainState; 6] = [
    DomainState::Pending,
    DomainState::Starting,
    DomainState::Ready,
    DomainState::Delayed,
    DomainState::Degraded,
    DomainState::Failed,
];

fn snapshot(sequence: u64, phase: BootPhase, health: BootHealth) -> BootSnapshot {
    let mut snapshot = BootSnapshot::new(sequence, Duration::from_millis(sequence * 100), phase);
    snapshot.health = health;
    snapshot
}

#[test]
fn all_single_domain_state_combinations_reduce_to_valid_bounded_modulation() {
    for phase in PHASES {
        for health in HEALTHS {
            for domain in DOMAINS {
                for state in DOMAIN_STATES {
                    let mut current = snapshot(1, phase, health);
                    current.domains = vec![DomainSnapshot {
                        domain,
                        state,
                        elapsed_ms: Some(50),
                    }];

                    let mut reducer = LiveEcologyReducer::new();
                    let modulation = reducer.reduce(&current).unwrap_or_else(|error| {
                        panic!(
                            "phase={phase:?} health={health:?} domain={domain:?} state={state:?}: {error}"
                        )
                    });

                    assert!(modulation.validate());
                    assert_eq!(modulation.anchor, SemanticBootAnchor::from(phase));
                    assert_eq!(modulation.reveal_floor, modulation.anchor.reveal_floor());
                    assert_eq!(modulation.handoff_ready, phase == BootPhase::Ready);
                    assert_eq!(modulation.accent, VisualAccent::None);

                    match state {
                        DomainState::Delayed => {
                            assert!(modulation.delayed_domains.contains(domain));
                            assert!(modulation.diagnostic_floor >= DiagnosticFloor::Status);
                        }
                        DomainState::Degraded => {
                            assert!(modulation.degraded_domains.contains(domain));
                            assert_eq!(modulation.diagnostic_floor, DiagnosticFloor::Diagnostics);
                        }
                        DomainState::Failed => {
                            assert!(modulation.failed_domains.contains(domain));
                            assert_eq!(modulation.diagnostic_floor, DiagnosticFloor::Diagnostics);
                        }
                        DomainState::Pending | DomainState::Starting | DomainState::Ready => {
                            assert!(!modulation.delayed_domains.contains(domain));
                            assert!(!modulation.degraded_domains.contains(domain));
                            assert!(!modulation.failed_domains.contains(domain));
                        }
                    }

                    if matches!(health, BootHealth::Degraded | BootHealth::Failed) {
                        assert_eq!(modulation.diagnostic_floor, DiagnosticFloor::Diagnostics);
                    } else if health == BootHealth::Delayed {
                        assert!(modulation.diagnostic_floor >= DiagnosticFloor::Status);
                    }
                }
            }
        }
    }
}

#[test]
fn every_forward_or_equal_phase_transition_is_accepted_and_every_rewind_is_rejected() {
    for (from_index, from) in PHASES.iter().copied().enumerate() {
        for (to_index, to) in PHASES.iter().copied().enumerate() {
            let mut reducer = LiveEcologyReducer::new();
            reducer
                .reduce(&snapshot(1, from, BootHealth::Normal))
                .unwrap();

            let result = reducer.reduce(&snapshot(2, to, BootHealth::Normal));
            if to_index >= from_index {
                assert!(result.is_ok(), "expected {from:?} -> {to:?} to be accepted");
            } else {
                assert!(
                    matches!(result, Err(LiveAdapterError::AnchorRegressed { .. })),
                    "expected {from:?} -> {to:?} to be rejected, got {result:?}"
                );
            }
        }
    }
}
