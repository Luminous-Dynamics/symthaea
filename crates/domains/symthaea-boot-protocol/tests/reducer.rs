use symthaea_boot_protocol::{
    BootDomain, BootEvent, BootHealth, BootPhase, Criticality,
};
use symthaea_boot_protocol::state::BootStateReducer;

#[test]
fn reducer_tracks_phase_and_domains_without_inventing_recovery() {
    let mut reducer = BootStateReducer::default();

    assert!(reducer.apply(&BootEvent::PhaseEntered {
        sequence: 1,
        elapsed_ms: 100,
        phase: BootPhase::Network,
    }));
    assert!(reducer.apply(&BootEvent::DomainFailed {
        sequence: 2,
        elapsed_ms: 200,
        domain: BootDomain::Network,
        criticality: Criticality::NonCritical,
        detail: None,
    }));
    assert!(reducer.apply(&BootEvent::DomainRecovered {
        sequence: 3,
        elapsed_ms: 300,
        domain: BootDomain::Network,
    }));

    let snapshot = reducer.snapshot();
    assert_eq!(snapshot.phase, BootPhase::Network);
    assert_eq!(snapshot.health, BootHealth::Failed);
    assert_eq!(snapshot.sequence, 3);
}

#[test]
fn reducer_only_accepts_monotonically_newer_events() {
    let mut reducer = BootStateReducer::default();

    assert!(reducer.apply(&BootEvent::DomainReady {
        sequence: 10,
        elapsed_ms: 10,
        domain: BootDomain::Kernel,
    }));
    assert!(!reducer.apply(&BootEvent::DomainReady {
        sequence: 10,
        elapsed_ms: 11,
        domain: BootDomain::Initrd,
    }));
    assert!(!reducer.apply(&BootEvent::DomainReady {
        sequence: 9,
        elapsed_ms: 12,
        domain: BootDomain::Storage,
    }));
}
