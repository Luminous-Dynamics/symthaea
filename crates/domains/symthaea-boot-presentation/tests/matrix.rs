use symthaea_boot_ecology_live::{
    DiagnosticFloor, DomainMask, LiveEcologyModulation, SemanticBootAnchor, VisualAccent,
};
use symthaea_boot_presentation::{PresentationDriver, SemanticTraceHasher};
use symthaea_boot_protocol::BootHealth;
use symthaea_boot_visual_clock::{VisualClockPolicy, truth_band};

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

const SCHEDULE_MS: [u32; 9] = [0, 1, 8, 16, 33, 100, 250, 1_000, 10_000];

fn modulation(anchor: SemanticBootAnchor, health: BootHealth) -> LiveEcologyModulation {
    LiveEcologyModulation {
        observation_sequence: 11,
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

fn replay(
    anchor: SemanticBootAnchor,
    health: BootHealth,
) -> (
    Vec<symthaea_boot_presentation::EcologyFrameInput>,
    symthaea_boot_presentation::TraceDigest,
) {
    let target = modulation(anchor, health);
    let mut driver = PresentationDriver::new(VisualClockPolicy::default()).unwrap();
    let mut trace = SemanticTraceHasher::new();
    let mut frames = Vec::new();

    for step_ms in SCHEDULE_MS {
        let frame = driver.advance_ms(step_ms, &target).unwrap();
        frame.validate().unwrap();
        assert!(frame.visual_phase <= truth_band(anchor).ceiling);
        trace.push(&frame).unwrap();
        frames.push(frame);
    }

    (frames, trace.finalize())
}

#[test]
fn every_anchor_health_schedule_replays_byte_semantically_identically() {
    for anchor in ANCHORS {
        for health in HEALTHS {
            let (left_frames, left_trace) = replay(anchor, health);
            let (right_frames, right_trace) = replay(anchor, health);
            assert_eq!(left_frames, right_frames);
            assert_eq!(left_trace, right_trace);
        }
    }
}

#[test]
fn unknown_ready_reaches_terminal_fact_without_known_normal_celebration() {
    let (frames, _) = replay(SemanticBootAnchor::SessionReady, BootHealth::Unknown);
    assert!(frames.iter().all(|frame| frame.handoff_ready));
    assert!(frames
        .iter()
        .all(|frame| frame.accent != VisualAccent::Ready));
}

#[test]
fn non_normal_states_never_gain_ambient_drift() {
    for anchor in ANCHORS {
        for health in [
            BootHealth::Unknown,
            BootHealth::Delayed,
            BootHealth::Degraded,
            BootHealth::Failed,
        ] {
            let (frames, _) = replay(anchor, health);
            assert!(frames.iter().all(|frame| {
                frame.clock_mode != symthaea_boot_visual_clock::ClockMode::AmbientDrift
            }));
        }
    }
}
