// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic structural reachability for productive-capability closure.

use crate::{CapabilityId, ClosureProblemV1, ClosureStatus, TargetReportV1};
use serde::Serialize;
use std::collections::BTreeSet;

/// Reachability-only result.
///
/// Import leverage is intentionally absent. CIV-BOOT-002C owns that theorem.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReachabilityReportV1 {
    pub locally_reproducible: Vec<CapabilityId>,
    pub operationally_reachable: Vec<CapabilityId>,
    pub targets: Vec<TargetReportV1>,
}

/// Evaluate local reproductive closure and import-enabled operational reachability.
///
/// This is structural graph reachability only. It does not establish quantities,
/// throughput, physical feasibility, maintenance, qualification, economics,
/// safety, or authority to manufacture.
pub fn evaluate_reachability(problem: &ClosureProblemV1) -> ReachabilityReportV1 {
    let local_seed: BTreeSet<_> = problem.local_primitives.iter().cloned().collect();
    let locally_reproducible = fixed_point(local_seed, problem);

    let mut operational_seed: BTreeSet<_> = problem.local_primitives.iter().cloned().collect();
    operational_seed.extend(problem.imports.iter().cloned());
    let operationally_reachable = fixed_point(operational_seed, problem);

    debug_assert!(locally_reproducible.is_subset(&operationally_reachable));

    let targets = problem
        .targets
        .iter()
        .cloned()
        .map(|target| {
            let status = if locally_reproducible.contains(&target) {
                ClosureStatus::LocallyClosed
            } else if operationally_reachable.contains(&target) {
                ClosureStatus::ImportDependent
            } else {
                ClosureStatus::Unavailable
            };
            TargetReportV1 { target, status }
        })
        .collect();

    ReachabilityReportV1 {
        locally_reproducible: locally_reproducible.into_iter().collect(),
        operationally_reachable: operationally_reachable.into_iter().collect(),
        targets,
    }
}

fn fixed_point(
    mut reachable: BTreeSet<CapabilityId>,
    problem: &ClosureProblemV1,
) -> BTreeSet<CapabilityId> {
    loop {
        let mut changed = false;

        for route in &problem.routes {
            if reachable.contains(&route.output) {
                continue;
            }
            if route
                .required_capabilities
                .iter()
                .all(|required| reachable.contains(required))
            {
                changed |= reachable.insert(route.output.clone());
            }
        }

        if !changed {
            return reachable;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RouteId, RouteV1};

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

    fn status(report: &ReachabilityReportV1, target: &str) -> ClosureStatus {
        report
            .targets
            .iter()
            .find(|entry| entry.target == cap(target))
            .unwrap()
            .status
    }

    #[test]
    fn local_import_and_downstream_statuses_match_reference_theorem() {
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

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "frame"), ClosureStatus::LocallyClosed);
        assert_eq!(status(&report, "motor"), ClosureStatus::ImportDependent);
        assert_eq!(status(&report, "machine"), ClosureStatus::ImportDependent);
        assert!(report.locally_reproducible.contains(&cap("frame")));
        assert!(!report.locally_reproducible.contains(&cap("motor")));
        assert!(report.operationally_reachable.contains(&cap("machine")));
    }

    #[test]
    fn missing_import_makes_downstream_targets_unavailable() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-copper"), cap("local-metal")],
            vec![],
            vec![cap("machine"), cap("motor")],
            vec![
                route("frame-route", "frame", &["local-metal"]),
                route("motor-route", "motor", &["bearing", "local-copper"]),
                route("machine-route", "machine", &["frame", "motor"]),
            ],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "motor"), ClosureStatus::Unavailable);
        assert_eq!(status(&report, "machine"), ClosureStatus::Unavailable);
    }

    #[test]
    fn complete_local_alternative_route_closes_downstream_chain() {
        let problem = ClosureProblemV1::new(
            vec![cap("ceramic-feed"), cap("local-copper"), cap("local-metal")],
            vec![cap("bearing")],
            vec![cap("machine"), cap("motor")],
            vec![
                route("bearing-local", "bearing", &["ceramic-feed"]),
                route("frame-route", "frame", &["local-metal"]),
                route("motor-route", "motor", &["bearing", "local-copper"]),
                route("machine-route", "machine", &["frame", "motor"]),
            ],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "motor"), ClosureStatus::LocallyClosed);
        assert_eq!(status(&report, "machine"), ClosureStatus::LocallyClosed);
    }

    #[test]
    fn unsupported_cycles_do_not_bootstrap_themselves() {
        let problem = ClosureProblemV1::new(
            vec![],
            vec![],
            vec![cap("a"), cap("b")],
            vec![route("a-from-b", "a", &["b"]), route("b-from-a", "b", &["a"])],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "a"), ClosureStatus::Unavailable);
        assert_eq!(status(&report, "b"), ClosureStatus::Unavailable);
    }

    #[test]
    fn imported_finished_capability_is_operational_not_local() {
        let problem = ClosureProblemV1::new(
            vec![],
            vec![cap("machine")],
            vec![cap("machine")],
            vec![],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "machine"), ClosureStatus::ImportDependent);
        assert!(report.locally_reproducible.is_empty());
        assert_eq!(report.operationally_reachable, vec![cap("machine")]);
    }

    #[test]
    fn undeclared_prerequisite_remains_unreachable() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-metal")],
            vec![],
            vec![cap("widget")],
            vec![route("widget-route", "widget", &["local-metal", "missing-seal"])],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "widget"), ClosureStatus::Unavailable);
    }

    #[test]
    fn local_or_route_wins_without_erasing_import_dependent_alternative() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-magnet"), cap("local-metal")],
            vec![cap("electronics")],
            vec![cap("actuator")],
            vec![
                route("electronic-route", "actuator", &["electronics", "local-metal"]),
                route("local-route", "actuator", &["local-magnet", "local-metal"]),
            ],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "actuator"), ClosureStatus::LocallyClosed);
    }

    #[test]
    fn local_closure_is_always_subset_of_operational_reachability() {
        let problem = ClosureProblemV1::new(
            vec![cap("metal")],
            vec![cap("bearing")],
            vec![cap("frame"), cap("motor")],
            vec![
                route("frame", "frame", &["metal"]),
                route("motor", "motor", &["bearing", "frame"]),
            ],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert!(report
            .locally_reproducible
            .iter()
            .all(|capability| report.operationally_reachable.contains(capability)));
    }

    #[test]
    fn canonical_problem_order_yields_byte_identical_report_serialization() {
        let first = ClosureProblemV1::new(
            vec![cap("metal"), cap("copper")],
            vec![cap("bearing")],
            vec![cap("machine"), cap("motor")],
            vec![
                route("motor-route", "motor", &["bearing", "copper"]),
                route("machine-route", "machine", &["metal", "motor"]),
            ],
        )
        .unwrap();
        let second = ClosureProblemV1::new(
            vec![cap("copper"), cap("metal")],
            vec![cap("bearing")],
            vec![cap("motor"), cap("machine")],
            vec![
                route("machine-route", "machine", &["motor", "metal"]),
                route("motor-route", "motor", &["copper", "bearing"]),
            ],
        )
        .unwrap();

        let a = serde_json::to_string(&evaluate_reachability(&first)).unwrap();
        let b = serde_json::to_string(&evaluate_reachability(&second)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn robot_housing_does_not_close_imported_actuator_stack() {
        let problem = ClosureProblemV1::new(
            vec![cap("local-polymer"), cap("printer")],
            vec![cap("controller"), cap("motor")],
            vec![cap("housing"), cap("robot")],
            vec![
                route("housing-route", "housing", &["local-polymer", "printer"]),
                route("robot-route", "robot", &["controller", "housing", "motor"]),
            ],
        )
        .unwrap();

        let report = evaluate_reachability(&problem);
        assert_eq!(status(&report, "housing"), ClosureStatus::LocallyClosed);
        assert_eq!(status(&report, "robot"), ClosureStatus::ImportDependent);
    }
}
