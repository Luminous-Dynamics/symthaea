use symthaea_boot_protocol::state::BootStateReducer;
use symthaea_boot_protocol::{
    BootDomain, BootEvent, BootHealth, BootPhase, BootSnapshot, Criticality, ProtocolError,
};
use std::time::Duration;

#[test]
fn reducer_tracks_phase_and_domains_without_inventing_recovery() {
    let mut reducer = BootStateReducer::default();

    assert!(reducer
        .try_apply(&BootEvent::PhaseEntered {
            sequence: 1,
            elapsed_ms: 100,
            phase: BootPhase::Network,
        })
        .unwrap());
    assert!(reducer
        .try_apply(&BootEvent::DomainFailed {
            sequence: 2,
            elapsed_ms: 200,
            domain: BootDomain::Network,
            criticality: Criticality::NonCritical,
            detail: None,
        })
        .unwrap());
    assert!(reducer
        .try_apply(&BootEvent::DomainRecovered {
            sequence: 3,
            elapsed_ms: 300,
            domain: BootDomain::Network,
        })
        .unwrap());

    let snapshot = reducer.snapshot();
    assert_eq!(snapshot.phase, BootPhase::Network);
    assert_eq!(snapshot.health, BootHealth::Failed);
    assert_eq!(snapshot.sequence, 3);
    assert_eq!(snapshot.elapsed_ms, 300);
}

#[test]
fn reducer_only_accepts_monotonically_newer_events() {
    let mut reducer = BootStateReducer::default();

    assert!(reducer
        .try_apply(&BootEvent::DomainReady {
            sequence: 10,
            elapsed_ms: 10,
            domain: BootDomain::Kernel,
        })
        .unwrap());
    assert!(!reducer
        .try_apply(&BootEvent::DomainReady {
            sequence: 10,
            elapsed_ms: 11,
            domain: BootDomain::Initrd,
        })
        .unwrap());
    assert!(!reducer
        .try_apply(&BootEvent::DomainReady {
            sequence: 9,
            elapsed_ms: 12,
            domain: BootDomain::Storage,
        })
        .unwrap());
}

#[test]
fn newer_sequence_cannot_rewind_elapsed_time() {
    let mut reducer = BootStateReducer::default();
    reducer
        .try_apply(&BootEvent::DomainReady {
            sequence: 1,
            elapsed_ms: 100,
            domain: BootDomain::Kernel,
        })
        .unwrap();

    assert!(matches!(
        reducer.try_apply(&BootEvent::DomainReady {
            sequence: 2,
            elapsed_ms: 99,
            domain: BootDomain::Initrd,
        }),
        Err(ProtocolError::ElapsedRegressed {
            previous_ms: 100,
            observed_ms: 99,
        })
    ));

    let snapshot = reducer.snapshot();
    assert_eq!(snapshot.sequence, 1);
    assert_eq!(snapshot.elapsed_ms, 100);
}

#[test]
fn authoritative_snapshot_cannot_replace_state_with_older_lineage_point() {
    let mut reducer = BootStateReducer::default();
    reducer
        .try_apply(&BootEvent::DomainReady {
            sequence: 8,
            elapsed_ms: 800,
            domain: BootDomain::Storage,
        })
        .unwrap();

    let older = BootSnapshot::new(7, Duration::from_millis(900), BootPhase::Filesystems);
    assert!(!reducer.try_replace(older).unwrap());

    let same_sequence_but_earlier =
        BootSnapshot::new(8, Duration::from_millis(799), BootPhase::Filesystems);
    assert!(matches!(
        reducer.try_replace(same_sequence_but_earlier),
        Err(ProtocolError::ElapsedRegressed { .. })
    ));

    let authoritative = BootSnapshot::new(8, Duration::from_millis(800), BootPhase::Filesystems);
    assert!(reducer.try_replace(authoritative).unwrap());
    assert_eq!(reducer.snapshot().phase, BootPhase::Filesystems);
}
