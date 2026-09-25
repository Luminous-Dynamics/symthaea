// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic structural import-leverage analysis.

use crate::{evaluate_reachability, ClosureProblemV1, ImportLeverageV1};
use std::collections::BTreeSet;

/// Evaluate the structural consequence of removing each declared import.
///
/// This reports graph reachability loss only. It does not assign economic,
/// social, geopolitical, mass, scarcity, or procurement value to an import.
pub fn evaluate_import_leverage(problem: &ClosureProblemV1) -> Vec<ImportLeverageV1> {
    let baseline = evaluate_reachability(problem);
    let baseline_reachable: BTreeSet<_> =
        baseline.operationally_reachable.iter().cloned().collect();

    problem
        .imports
        .iter()
        .map(|removed_import| {
            let mut without = problem.clone();
            without.imports.retain(|candidate| candidate != removed_import);

            let without_report = evaluate_reachability(&without);
            let without_reachable: BTreeSet<_> = without_report
                .operationally_reachable
                .iter()
                .cloned()
                .collect();

            debug_assert!(without_reachable.is_subset(&baseline_reachable));

            let targets_lost_if_removed = problem
                .targets
                .iter()
                .filter(|target| {
                    baseline_reachable.contains(*target) && !without_reachable.contains(*target)
                })
                .cloned()
                .collect();

            let capability_count_lost = baseline_reachable
                .difference(&without_reachable)
                .count();

            ImportLeverageV1::new(
                removed_import.clone(),
                targets_lost_if_removed,
                capability_count_lost,
            )
            .expect("lost targets are a canonical subset of validated problem targets")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapabilityId, RouteId, RouteV1};

    fn cap(value: &str) -> CapabilityId {
        CapabilityId::new(value).unwrap()
    }

    fn route_id(value: &str) -> RouteId {
        RouteId::new(value).unwrap()
    }

    fn route(id: &str, output: &str, requires: &[&str]) -> RouteV1 {
        RouteV1::new(
            route_id(id),
            cap(output),
            requires.iter().map(|value| cap(value)).collect(),
        )
        .unwrap()
    }

    fn leverage<'a>(
        report: &'a [ImportLeverageV1],
        import: &str,
    ) -> &'a ImportLeverageV1 {
        report
            .iter()
            .find(|entry| entry.import == cap(import))
            .unwrap()
    }

    #[test]
    fn upstream_bearing_import_exposes_downstream_structural_loss() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-copper"), cap("local-metal")],
            vec![cap("bearing")],
            vec![cap("frame"), cap("machine"), cap("motor")],
            vec![
                route("frame-route", "frame", &["local-metal"]),
                route("motor-route", "motor", &["bearing", "local-copper"]),
                route("machine-route", "machine", &["frame", "motor"]),
            ],
        )
        .unwrap();

        let report = evaluate_import_leverage(&problem);
        let bearing = leverage(&report, "bearing");
        assert_eq!(
            bearing.targets_lost_if_removed,
            vec![cap("machine"), cap("motor")]
        );
        assert_eq!(bearing.capability_count_lost, 3);
    }

    #[test]
    fn unrelated_import_can_have_zero_target_loss() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-metal")],
            vec![cap("spare-widget")],
            vec![cap("frame")],
            vec![route("frame-route", "frame", &["local-metal"])],
        )
        .unwrap();

        let report = evaluate_import_leverage(&problem);
        let unrelated = leverage(&report, "spare-widget");
        assert!(unrelated.targets_lost_if_removed.is_empty());
        assert_eq!(unrelated.capability_count_lost, 1);
    }

    #[test]
    fn local_alternative_route_reduces_import_leverage_to_zero() {
        let problem = ClosureProblemV1::new(
            vec![cap("ceramic-feed"), cap("local-copper")],
            vec![cap("bearing")],
            vec![cap("motor")],
            vec![
                route("bearing-local", "bearing", &["ceramic-feed"]),
                route("motor-route", "motor", &["bearing", "local-copper"]),
            ],
        )
        .unwrap();

        let report = evaluate_import_leverage(&problem);
        let bearing = leverage(&report, "bearing");
        assert!(bearing.targets_lost_if_removed.is_empty());
        assert_eq!(bearing.capability_count_lost, 0);
    }

    #[test]
    fn independent_import_branches_have_separate_loss_sets() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-copper"), cap("local-frame")],
            vec![cap("bearing"), cap("controller")],
            vec![cap("drive"), cap("motor")],
            vec![
                route("motor-route", "motor", &["bearing", "local-copper"]),
                route("drive-route", "drive", &["controller", "local-frame"]),
            ],
        )
        .unwrap();

        let report = evaluate_import_leverage(&problem);
        assert_eq!(
            leverage(&report, "bearing").targets_lost_if_removed,
            vec![cap("motor")]
        );
        assert_eq!(
            leverage(&report, "controller").targets_lost_if_removed,
            vec![cap("drive")]
        );
    }

    #[test]
    fn one_upstream_import_can_have_high_downstream_fanout() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-copper"), cap("local-metal")],
            vec![cap("bearing")],
            vec![cap("actuator"), cap("machine"), cap("motor"), cap("robot")],
            vec![
                route("motor-route", "motor", &["bearing", "local-copper"]),
                route("actuator-route", "actuator", &["motor"]),
                route("machine-route", "machine", &["actuator", "local-metal"]),
                route("robot-route", "robot", &["actuator", "local-metal"]),
            ],
        )
        .unwrap();

        let report = evaluate_import_leverage(&problem);
        let bearing = leverage(&report, "bearing");
        assert_eq!(
            bearing.targets_lost_if_removed,
            vec![cap("actuator"), cap("machine"), cap("motor"), cap("robot")]
        );
        assert_eq!(bearing.capability_count_lost, 5);
    }

    #[test]
    fn imported_finished_machine_has_operational_leverage_without_local_reproduction() {
        let problem = ClosureProblemV1::new(
            vec![],
            vec![cap("machine")],
            vec![cap("machine")],
            vec![],
        )
        .unwrap();

        let report = evaluate_import_leverage(&problem);
        let machine = leverage(&report, "machine");
        assert_eq!(machine.targets_lost_if_removed, vec![cap("machine")]);
        assert_eq!(machine.capability_count_lost, 1);
    }

    #[test]
    fn import_declaration_order_cannot_change_canonical_output() {
        let first = ClosureProblemV1::new(
            vec![cap("local")],
            vec![cap("z-import"), cap("a-import")],
            vec![cap("a-target"), cap("z-target")],
            vec![
                route("a-route", "a-target", &["a-import", "local"]),
                route("z-route", "z-target", &["local", "z-import"]),
            ],
        )
        .unwrap();
        let second = ClosureProblemV1::new(
            vec![cap("local")],
            vec![cap("a-import"), cap("z-import")],
            vec![cap("z-target"), cap("a-target")],
            vec![
                route("z-route", "z-target", &["z-import", "local"]),
                route("a-route", "a-target", &["local", "a-import"]),
            ],
        )
        .unwrap();

        let a = serde_json::to_string(&evaluate_import_leverage(&first)).unwrap();
        let b = serde_json::to_string(&evaluate_import_leverage(&second)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn removing_import_cannot_increase_reachability() {
        let problem = ClosureProblemV1::new(
            vec![cap("local")],
            vec![cap("imported")],
            vec![cap("target")],
            vec![route("target-route", "target", &["imported", "local"])],
        )
        .unwrap();

        let baseline = evaluate_reachability(&problem);
        let baseline_set: BTreeSet<_> = baseline.operationally_reachable.into_iter().collect();

        let mut without = problem.clone();
        without.imports.clear();
        let reduced = evaluate_reachability(&without);
        let reduced_set: BTreeSet<_> = reduced.operationally_reachable.into_iter().collect();

        assert!(reduced_set.is_subset(&baseline_set));
    }
}
