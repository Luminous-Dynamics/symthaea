use std::time::Duration;

use symthaea_boot_ecology_live::{
    DiagnosticFloor, LiveEcologyModulation, LiveEcologyReducer, SemanticBootAnchor, VisualAccent,
};
use symthaea_boot_protocol::{
    BootDomain, BootHealth, BootPhase, BootSnapshot, DomainSnapshot, DomainState,
};

fn snapshot(sequence: u64, phase: BootPhase, health: BootHealth) -> BootSnapshot {
    let mut snapshot = BootSnapshot::new(sequence, Duration::from_millis(sequence * 100), phase);
    snapshot.health = health;
    snapshot
}

fn irregular_trace() -> Vec<BootSnapshot> {
    let kernel = snapshot(1, BootPhase::Kernel, BootHealth::Unknown);

    let storage = snapshot(2, BootPhase::Storage, BootHealth::Normal);

    let mut delayed_network = snapshot(3, BootPhase::Network, BootHealth::Delayed);
    delayed_network.domains = vec![DomainSnapshot {
        domain: BootDomain::Network,
        state: DomainState::Delayed,
        elapsed_ms: Some(300),
    }];

    // New telemetry sequence, but no presentation-relevant semantic change.
    let mut network_churn = snapshot(4, BootPhase::Network, BootHealth::Delayed);
    network_churn.domains = vec![DomainSnapshot {
        domain: BootDomain::Network,
        state: DomainState::Delayed,
        elapsed_ms: Some(400),
    }];

    let mut degraded_services = snapshot(5, BootPhase::Services, BootHealth::Degraded);
    degraded_services.domains = vec![
        DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Ready,
            elapsed_ms: Some(450),
        },
        DomainSnapshot {
            domain: BootDomain::Services,
            state: DomainState::Degraded,
            elapsed_ms: Some(500),
        },
    ];

    let mut recovered_services = snapshot(6, BootPhase::Services, BootHealth::Normal);
    recovered_services.domains = vec![
        DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Ready,
            elapsed_ms: Some(450),
        },
        DomainSnapshot {
            domain: BootDomain::Services,
            state: DomainState::Ready,
            elapsed_ms: Some(600),
        },
    ];

    let graphics = snapshot(7, BootPhase::Graphics, BootHealth::Normal);
    let session = snapshot(8, BootPhase::Session, BootHealth::Normal);
    let ready = snapshot(9, BootPhase::Ready, BootHealth::Normal);

    vec![
        kernel,
        storage,
        delayed_network,
        network_churn,
        degraded_services,
        recovered_services,
        graphics,
        session,
        ready,
    ]
}

fn replay(trace: &[BootSnapshot]) -> Vec<LiveEcologyModulation> {
    let mut reducer = LiveEcologyReducer::new();
    trace
        .iter()
        .map(|snapshot| reducer.reduce(snapshot).expect("trace must reduce"))
        .collect()
}

#[test]
fn irregular_trace_replays_identically() {
    let trace = irregular_trace();
    assert_eq!(replay(&trace), replay(&trace));
}

#[test]
fn irregular_trace_has_truthful_expected_semantics() {
    let output = replay(&irregular_trace());

    let expected_accents = [
        VisualAccent::None,
        VisualAccent::Progress,
        VisualAccent::Delay,
        VisualAccent::None,
        VisualAccent::Degraded,
        VisualAccent::Recovery,
        VisualAccent::Progress,
        VisualAccent::Progress,
        VisualAccent::Ready,
    ];
    let observed_accents: Vec<_> = output.iter().map(|item| item.accent).collect();
    assert_eq!(observed_accents, expected_accents);

    // Telemetry churn must not manufacture a second transient accent.
    assert_eq!(output[2].accent_token, 3);
    assert_eq!(output[3].accent_token, output[2].accent_token);

    assert_eq!(output[2].diagnostic_floor, DiagnosticFloor::Status);
    assert_eq!(output[4].diagnostic_floor, DiagnosticFloor::Diagnostics);
    assert_eq!(output[5].diagnostic_floor, DiagnosticFloor::Ambient);

    assert!(output[2].delayed_domains.contains(BootDomain::Network));
    assert!(output[4].degraded_domains.contains(BootDomain::Services));
    assert!(output[5].degraded_domains.is_empty());

    for pair in output.windows(2) {
        assert!(pair[1].anchor >= pair[0].anchor);
        assert!(pair[1].reveal_floor >= pair[0].reveal_floor);
    }

    let final_state = output.last().expect("non-empty trace");
    assert_eq!(final_state.anchor, SemanticBootAnchor::SessionReady);
    assert!(final_state.handoff_ready);
    assert_eq!(final_state.accent, VisualAccent::Ready);
}
