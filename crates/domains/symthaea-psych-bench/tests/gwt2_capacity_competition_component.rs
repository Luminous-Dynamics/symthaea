//! Component-level GWT-2 theorem for the production global workspace.
//!
//! GWT-2 requires a limited-capacity workspace with a real information
//! bottleneck and selective competition. This test therefore overfills the
//! production workspace with distinct candidates and verifies that admission is
//! capacity-limited and activation-sensitive rather than merely checking that a
//! coalition size is non-zero.

use std::collections::BTreeSet;

use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{
    GlobalWorkspace, WorkspaceAssessment, WorkspaceConfig, WorkspaceContent,
};

fn content(source: &str, activation: f64, seed: u64) -> WorkspaceContent {
    WorkspaceContent::new(
        vec![BinaryHV::random(seed)],
        activation,
        source.to_string(),
    )
}

fn run_workspace(
    max_capacity: usize,
    enable_broadcasting: bool,
    candidates: &[(&str, f64, u64)],
) -> WorkspaceAssessment {
    let mut workspace = GlobalWorkspace::new(WorkspaceConfig {
        max_capacity,
        entry_threshold: 0.10,
        decay_rate: 0.0,
        enable_broadcasting,
        winner_takes_all: false,
        max_duration: 50,
    });

    for &(source, activation, seed) in candidates {
        workspace.submit(content(source, activation, seed));
    }

    workspace.process()
}

fn sources(contents: &[WorkspaceContent]) -> BTreeSet<String> {
    contents.iter().map(|c| c.source.clone()).collect()
}

const BASELINE: [(&str, f64, u64); 4] = [
    ("A", 0.90, 0x6A72_0001),
    ("B", 0.80, 0x6A72_0002),
    ("C", 0.70, 0x6A72_0003),
    ("D", 0.60, 0x6A72_0004),
];

#[test]
fn capacity_two_workspace_admits_only_two_strongest_candidates() {
    let assessment = run_workspace(2, false, &BASELINE);

    assert_eq!(assessment.capacity.max_capacity, 2);
    assert_eq!(assessment.capacity.num_contents, 2);
    assert_eq!(assessment.conscious_contents.len(), 2);
    assert_eq!(assessment.competing_contents.len(), 2);

    assert_eq!(
        sources(&assessment.conscious_contents),
        BTreeSet::from(["A".to_string(), "B".to_string()]),
        "capacity-limited workspace did not admit the two strongest candidates",
    );
    assert_eq!(
        sources(&assessment.competing_contents),
        BTreeSet::from(["C".to_string(), "D".to_string()]),
        "excluded candidates were not retained as preconscious competitors",
    );
}

#[test]
fn changing_activation_changes_winner_identity_without_changing_capacity() {
    let baseline = run_workspace(2, false, &BASELINE);
    let salience_reversal = [
        ("A", 0.90, 0x6A72_0001),
        ("B", 0.80, 0x6A72_0002),
        ("C", 0.70, 0x6A72_0003),
        ("D", 0.95, 0x6A72_0004),
    ];
    let shifted = run_workspace(2, false, &salience_reversal);

    assert_eq!(baseline.capacity.max_capacity, shifted.capacity.max_capacity);
    assert_eq!(baseline.capacity.num_contents, shifted.capacity.num_contents);

    assert_eq!(
        sources(&baseline.conscious_contents),
        BTreeSet::from(["A".to_string(), "B".to_string()]),
    );
    assert_eq!(
        sources(&shifted.conscious_contents),
        BTreeSet::from(["A".to_string(), "D".to_string()]),
        "raising D's activation did not change selective workspace admission",
    );
}

#[test]
fn unrelated_broadcast_toggle_is_a_sham_for_workspace_admission() {
    let broadcast_off = run_workspace(2, false, &BASELINE);
    let broadcast_on = run_workspace(2, true, &BASELINE);

    assert_eq!(
        sources(&broadcast_off.conscious_contents),
        sources(&broadcast_on.conscious_contents),
        "broadcast setting unexpectedly changed competition/admission",
    );
    assert_eq!(
        sources(&broadcast_off.competing_contents),
        sources(&broadcast_on.competing_contents),
        "broadcast setting unexpectedly changed excluded competitors",
    );
    assert_eq!(broadcast_off.capacity.num_contents, 2);
    assert_eq!(broadcast_on.capacity.num_contents, 2);
}

#[test]
fn expanding_capacity_removes_the_bottleneck_and_rescues_excluded_content() {
    let limited = run_workspace(2, false, &BASELINE);
    let expanded = run_workspace(4, false, &BASELINE);

    assert_eq!(limited.conscious_contents.len(), 2);
    assert_eq!(expanded.conscious_contents.len(), 4);
    assert!(expanded.competing_contents.is_empty());
    assert_eq!(
        sources(&expanded.conscious_contents),
        BTreeSet::from([
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ]),
        "capacity expansion did not restore previously excluded content",
    );
}

#[test]
fn component_scope_refuses_gwt2_functional_promotion() {
    const CLAIM_SCOPE: &str =
        "component bottleneck/competition theorem only; no downstream cognitive-task consequence; no GWT-2 tier promotion; no consciousness claim";
    assert!(CLAIM_SCOPE.contains("no GWT-2 tier promotion"));
    assert!(CLAIM_SCOPE.contains("no consciousness claim"));
}
