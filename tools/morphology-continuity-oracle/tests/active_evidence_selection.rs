//! Finite action-selection oracle for what to inspect when target identity is ambiguous.
//!
//! This test does not implement an FEP policy, cryptographic verification, a sensor,
//! or an experiment scheduler. Provenance status is a frozen scenario input. The
//! fixture asks whether the action selector can distinguish evidence that truly
//! partitions the remaining hypotheses from repeated observations that cannot.

#[derive(Clone, Copy, Debug, PartialEq)]
struct CandidateAction {
    id: &'static str,
    /// Deterministic outcome under [damaged-chair history, intact-table history].
    outcomes: [&'static str; 2],
    /// Declared cost in integer milli-cost units. Must be nonzero.
    cost_milliunits: u32,
    /// Scenario annotation: only authenticated/validated sources can update belief.
    /// This boolean is supplied by the fixture, not calculated or cryptographically checked.
    provenance_validated: bool,
}

const PRIOR_ENTROPY_MILLIBITS: u32 = 1_000;

fn remaining_entropy_millibits(action: CandidateAction) -> u32 {
    if !action.provenance_validated || action.outcomes[0] == action.outcomes[1] {
        PRIOR_ENTROPY_MILLIBITS
    } else {
        // With two equally likely histories and a deterministic observation whose
        // outcomes are distinct, the posterior entropy is exactly zero.
        0
    }
}

fn information_gain_millibits(action: CandidateAction) -> u32 {
    PRIOR_ENTROPY_MILLIBITS - remaining_entropy_millibits(action)
}

/// Select the highest exact integer information-gain / cost ratio. Zero-information
/// actions are not promoted over abstention. Cross multiplication avoids float ordering;
/// an ID tie-break provides deterministic results.
fn choose_best_measurement(actions: &[CandidateAction]) -> Option<CandidateAction> {
    let mut best: Option<(CandidateAction, u64, u64)> = None;

    for action in actions.iter().copied() {
        if action.cost_milliunits == 0 {
            continue;
        }
        let gain = u64::from(information_gain_millibits(action));
        if gain == 0 {
            continue;
        }

        let cost = u64::from(action.cost_milliunits);
        let replace = match best {
            None => true,
            Some((current, best_gain, best_cost)) => {
                let left = gain * best_cost;
                let right = best_gain * cost;
                left > right || (left == right && action.id < current.id)
            }
        };

        if replace {
            best = Some((action, gain, cost));
        }
    }

    best.map(|(action, _, _)| action)
}

fn ambiguous_history_actions() -> [CandidateAction; 5] {
    [
        CandidateAction {
            id: "rescan-current-geometry",
            outcomes: ["table-shaped", "table-shaped"],
            cost_milliunits: 1_000,
            provenance_validated: true,
        },
        CandidateAction {
            id: "repeat-local-shape-classification",
            outcomes: ["table", "table"],
            cost_milliunits: 500,
            provenance_validated: true,
        },
        CandidateAction {
            id: "read-unverified-target-label",
            outcomes: ["chair", "table"],
            cost_milliunits: 250,
            provenance_validated: false,
        },
        CandidateAction {
            id: "read-validated-design-record",
            outcomes: ["chair-record", "table-record"],
            cost_milliunits: 2_000,
            provenance_validated: true,
        },
        CandidateAction {
            id: "inspect-validated-inventory-manifest",
            outcomes: ["chair-bom", "table-bom"],
            cost_milliunits: 3_000,
            provenance_validated: true,
        },
    ]
}

#[test]
fn repeated_geometry_cannot_resolve_historical_identity_alias() {
    let actions = ambiguous_history_actions();
    for action in actions.iter().filter(|item| {
        item.id == "rescan-current-geometry" || item.id == "repeat-local-shape-classification"
    }) {
        assert_eq!(information_gain_millibits(*action), 0, "{}", action.id);
        assert_eq!(
            remaining_entropy_millibits(*action),
            PRIOR_ENTROPY_MILLIBITS,
            "{}",
            action.id
        );
    }
}

#[test]
fn an_unverified_target_label_must_not_change_the_belief() {
    let unverified = ambiguous_history_actions()
        .into_iter()
        .find(|action| action.id == "read-unverified-target-label")
        .expect("unverified-label action exists");

    // Different strings alone are not enough to count as trustworthy evidence.
    assert_ne!(unverified.outcomes[0], unverified.outcomes[1]);
    assert_eq!(information_gain_millibits(unverified), 0);
    assert_eq!(
        remaining_entropy_millibits(unverified),
        PRIOR_ENTROPY_MILLIBITS
    );
}

#[test]
fn action_selector_prefers_validated_target_evidence_over_redundant_scans() {
    let actions = ambiguous_history_actions();
    let selected =
        choose_best_measurement(&actions).expect("at least one valid evidence action exists");
    assert_eq!(selected.id, "read-validated-design-record");
    assert_eq!(
        information_gain_millibits(selected),
        PRIOR_ENTROPY_MILLIBITS
    );
}

#[test]
fn absent_validated_evidence_action_means_abstain_not_guess() {
    let actions = ambiguous_history_actions();
    let unavailable: Vec<_> = actions
        .into_iter()
        .filter(|action| {
            action.id != "read-validated-design-record"
                && action.id != "inspect-validated-inventory-manifest"
        })
        .collect();

    assert_eq!(choose_best_measurement(&unavailable), None);
}

#[test]
fn equal_information_value_per_cost_uses_stable_id_tiebreak() {
    let actions = [
        CandidateAction {
            id: "source-b",
            outcomes: ["chair", "table"],
            cost_milliunits: 2_000,
            provenance_validated: true,
        },
        CandidateAction {
            id: "source-a",
            outcomes: ["chair", "table"],
            cost_milliunits: 2_000,
            provenance_validated: true,
        },
    ];
    assert_eq!(
        choose_best_measurement(&actions).map(|action| action.id),
        Some("source-a")
    );
}
