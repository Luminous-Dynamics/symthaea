use symthaea_boot_protocol::{
    BootDomain, BootEvent, BootHealth, BootPhase, BootSnapshot, BoundedDetail, Criticality,
    DomainSnapshot, DomainState, MAX_DETAIL_BYTES,
};
use std::time::Duration;

#[test]
fn snapshot_accepts_all_declared_domains_once() {
    let mut snapshot = BootSnapshot::new(7, Duration::from_millis(125), BootPhase::Services);
    snapshot.domains = [
        BootDomain::Kernel,
        BootDomain::Initrd,
        BootDomain::Storage,
        BootDomain::Filesystems,
        BootDomain::Security,
        BootDomain::Network,
        BootDomain::Services,
        BootDomain::Graphics,
        BootDomain::Session,
    ]
    .into_iter()
    .map(|domain| DomainSnapshot {
        domain,
        state: DomainState::Ready,
        elapsed_ms: Some(1),
    })
    .collect();

    snapshot.validate().unwrap();
}

#[test]
fn bounded_detail_accepts_exact_byte_limit() {
    let detail = BoundedDetail::new("x".repeat(MAX_DETAIL_BYTES)).unwrap();
    assert_eq!(detail.as_str().len(), MAX_DETAIL_BYTES);
}

#[test]
fn recovery_event_does_not_imply_boot_health() {
    let event = BootEvent::DomainRecovered {
        sequence: 22,
        elapsed_ms: 3210,
        domain: BootDomain::Network,
    };

    // Presentation consumers can observe recovery, but authoritative health is
    // supplied independently by snapshots/BootReady rather than inferred here.
    assert_eq!(event.sequence(), 22);
    event.validate().unwrap();
}

#[test]
fn failed_event_can_carry_bounded_operator_hint() {
    let event = BootEvent::DomainFailed {
        sequence: 23,
        elapsed_ms: 3500,
        domain: BootDomain::Graphics,
        criticality: Criticality::Critical,
        detail: Some(BoundedDetail::new("display manager failed").unwrap()),
    };

    event.validate().unwrap();
}

#[test]
fn health_unknown_is_not_equivalent_to_normal() {
    assert_ne!(BootHealth::Unknown, BootHealth::Normal);
    assert!(BootHealth::Unknown.severity() > BootHealth::Normal.severity());
}
