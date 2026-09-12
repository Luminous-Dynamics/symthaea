//! Component-level GWT-3 theorem for the production global workspace.
//!
//! The key distinction is workspace entry versus global availability. The same
//! content is admitted to the workspace in both arms; only broadcasting changes.
//! Registered recipient handlers must receive the exact winning representation
//! when broadcasting is enabled and must remain untouched when broadcasting is
//! disabled. This is mechanistic evidence only and does not promote a Butlin tier.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{
    GlobalWorkspace, WorkspaceConfig, WorkspaceContent,
};

const RECIPIENTS: [&str; 5] = ["perception", "memory", "planning", "language", "action"];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DeliveryProbe {
    hits: usize,
    exact_content_matches: usize,
}

#[derive(Debug)]
struct ArmResult {
    workspace_entries: usize,
    broadcast_events: usize,
    probes: HashMap<String, DeliveryProbe>,
}

fn same_representation(actual: &[BinaryHV], expected: &[BinaryHV]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| a.similarity(b) > 0.999_999)
}

fn run_arm(enable_broadcasting: bool) -> ArmResult {
    let signal = vec![
        BinaryHV::random(0x6A73_0001),
        BinaryHV::random(0x6A73_0002),
    ];

    let mut workspace = GlobalWorkspace::new(WorkspaceConfig {
        max_capacity: 1,
        entry_threshold: 0.30,
        decay_rate: 0.0,
        enable_broadcasting,
        winner_takes_all: true,
        max_duration: 50,
    });

    let probes: Arc<Mutex<HashMap<String, DeliveryProbe>>> = Arc::new(Mutex::new(
        RECIPIENTS
            .iter()
            .chain(std::iter::once(&"sentinel"))
            .map(|name| ((*name).to_string(), DeliveryProbe::default()))
            .collect(),
    ));
    let expected = Arc::new(signal.clone());

    for recipient in RECIPIENTS.iter().chain(std::iter::once(&"sentinel")) {
        let recipient_name = (*recipient).to_string();
        let probes_for_handler = Arc::clone(&probes);
        let expected_for_handler = Arc::clone(&expected);
        workspace.register_handler(
            recipient,
            Box::new(move |content| {
                let mut map = probes_for_handler.lock().expect("probe mutex poisoned");
                let probe = map
                    .get_mut(&recipient_name)
                    .expect("registered recipient missing from probe map");
                probe.hits += 1;
                if same_representation(content, expected_for_handler.as_slice()) {
                    probe.exact_content_matches += 1;
                }
            }),
        );
    }

    workspace.submit(WorkspaceContent::new(
        signal,
        0.90,
        "perception-source".to_string(),
    ));
    let assessment = workspace.process();

    let probes = Arc::try_unwrap(probes)
        .expect("all broadcast handler references should be owned by workspace/probe only")
        .into_inner()
        .expect("probe mutex poisoned");

    ArmResult {
        workspace_entries: assessment.conscious_contents.len(),
        broadcast_events: assessment.broadcasts.len(),
        probes,
    }
}

#[test]
fn broadcast_delivers_exact_workspace_content_to_every_registered_global_recipient() {
    let run = run_arm(true);

    assert_eq!(run.workspace_entries, 1, "winning content did not enter workspace");
    assert_eq!(run.broadcast_events, 1, "expected one broadcast event");

    for recipient in RECIPIENTS {
        let probe = run.probes.get(recipient).unwrap();
        assert_eq!(probe.hits, 1, "{recipient} did not receive exactly one broadcast");
        assert_eq!(
            probe.exact_content_matches, 1,
            "{recipient} received a broadcast but not the exact winning representation",
        );
    }

    let sentinel = run.probes.get("sentinel").unwrap();
    assert_eq!(sentinel.hits, 0, "unlisted sentinel received a global broadcast");
}

#[test]
fn disabling_broadcast_preserves_workspace_entry_but_eliminates_recipient_uptake() {
    let enabled = run_arm(true);
    let disabled = run_arm(false);

    // Targeted intervention does not destroy competition/admission. This keeps
    // the causal contrast specific to broadcast availability.
    assert_eq!(enabled.workspace_entries, 1);
    assert_eq!(disabled.workspace_entries, 1);

    assert_eq!(enabled.broadcast_events, 1);
    assert_eq!(disabled.broadcast_events, 0);

    for recipient in RECIPIENTS {
        assert_eq!(enabled.probes.get(recipient).unwrap().hits, 1);
        assert_eq!(
            disabled.probes.get(recipient).unwrap().hits,
            0,
            "{recipient} received content despite broadcasting being disabled",
        );
    }
}

#[test]
fn reenabled_broadcast_rescues_global_delivery_under_identical_stimulus() {
    let disabled = run_arm(false);
    let rescued = run_arm(true);

    assert_eq!(disabled.workspace_entries, rescued.workspace_entries);
    assert_eq!(disabled.broadcast_events, 0);
    assert_eq!(rescued.broadcast_events, 1);

    for recipient in RECIPIENTS {
        assert_eq!(disabled.probes.get(recipient).unwrap().hits, 0);
        assert_eq!(rescued.probes.get(recipient).unwrap().hits, 1);
        assert_eq!(rescued.probes.get(recipient).unwrap().exact_content_matches, 1);
    }
}

#[test]
fn component_scope_refuses_gwt3_functional_promotion() {
    const CLAIM_SCOPE: &str =
        "component broadcast-delivery theorem only; no downstream module-state consequence; no GWT-3 tier promotion; no consciousness claim";
    assert!(CLAIM_SCOPE.contains("no GWT-3 tier promotion"));
    assert!(CLAIM_SCOPE.contains("no consciousness claim"));
}
