use symthaea_boot_ecology_live::{
    DiagnosticFloor, DomainMask, LiveEcologyModulation, SemanticBootAnchor, VisualAccent,
};
use symthaea_boot_protocol::BootHealth;
use symthaea_boot_visual_clock::{ElasticVisualClock, VisualClockPolicy, truth_band};

const ANCHORS: [SemanticBootAnchor; 10] = [
    SemanticBootAnchor::KernelPhase,
    SemanticBootAnchor::InitrdPhase,
    SemanticBootAnchor::StoragePhase,
    SemanticBootAnchor::FilesystemsPhase,
    SemanticBootAnchor::SecurityPhase,
    SemanticBootAnchor::NetworkPhase,
    SemanticBootAnchor::ServicesPhase,
    SemanticBootAnchor::GraphicsPhase,
    SemanticBootAnchor::SessionPhase,
    SemanticBootAnchor::SessionReady,
];

const HEALTHS: [BootHealth; 5] = [
    BootHealth::Normal,
    BootHealth::Unknown,
    BootHealth::Delayed,
    BootHealth::Degraded,
    BootHealth::Failed,
];

fn modulation(anchor: SemanticBootAnchor, health: BootHealth) -> LiveEcologyModulation {
    LiveEcologyModulation {
        observation_sequence: 1,
        anchor,
        health,
        reveal_floor: anchor.reveal_floor(),
        delayed_domains: DomainMask::empty(),
        degraded_domains: DomainMask::empty(),
        failed_domains: DomainMask::empty(),
        diagnostic_floor: match health {
            BootHealth::Normal | BootHealth::Unknown => DiagnosticFloor::Ambient,
            BootHealth::Delayed => DiagnosticFloor::Status,
            BootHealth::Degraded | BootHealth::Failed => DiagnosticFloor::Diagnostics,
        },
        accent_token: 0,
        accent: VisualAccent::None,
        handoff_ready: anchor == SemanticBootAnchor::SessionReady,
    }
}

#[test]
fn all_anchor_health_pairs_remain_monotonic_and_inside_their_truth_band() {
    let policy = VisualClockPolicy::default();

    for anchor in ANCHORS {
        for health in HEALTHS {
            let target = modulation(anchor, health);
            let band = truth_band(anchor);
            let mut clock = ElasticVisualClock::new(policy);
            let mut previous = 0;

            for step_ms in [0, 1, 8, 16, 33, 100, 250, 1_000, 10_000] {
                let step = clock.advance_ms(step_ms, &target);
                assert!(step.after >= previous);
                assert!(step.after <= band.ceiling);
                previous = step.after;
            }
        }
    }
}

#[test]
fn non_normal_health_never_drift_past_the_factual_floor() {
    let policy = VisualClockPolicy::default();

    for anchor in ANCHORS {
        for health in [
            BootHealth::Unknown,
            BootHealth::Delayed,
            BootHealth::Degraded,
            BootHealth::Failed,
        ] {
            let target = modulation(anchor, health);
            let mut clock = ElasticVisualClock::from_phase(target.reveal_floor, policy);
            for _ in 0..100 {
                clock.advance_ms(250, &target);
            }
            assert_eq!(clock.phase(), target.reveal_floor);
        }
    }
}
