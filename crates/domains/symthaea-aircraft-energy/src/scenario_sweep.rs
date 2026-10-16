// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, multi-objective comparison of energy-system candidates on one
//! shared aircraft mission. This is an arithmetic screening aid, not a
//! propulsion optimizer, aircraft model, airworthiness assessment, or flight
//! authorization.

use std::collections::HashSet;

use super::{evaluate_mission, Disposition, EnergySystemEnvelope, MissionProfile, ScreeningResult};

/// One candidate energy-system envelope to evaluate against the same mission.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemCandidate {
    /// Stable, non-empty identifier unique within the comparison.
    pub id: String,
    pub system: EnergySystemEnvelope,
}

/// The screening result for one candidate, in stable identifier order.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioRow {
    pub candidate_id: String,
    pub result: ScreeningResult,
    /// True only when the candidate is arithmetically screen-feasible and no
    /// other feasible candidate dominates it on all reported objectives.
    pub on_pareto_front: bool,
}

/// Invalid comparison input. The per-candidate arithmetic outcome is instead
/// recorded as a normal ScreeningResult and remains fail-closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SweepInputError {
    EmptyCandidateSet,
    InvalidCandidateId { id: String },
    DuplicateCandidateId { id: String },
}

/// Results for alternative energy-system envelopes evaluated on one identical mission.
///
/// Pareto objectives are: minimize complete installed system mass, and maximize
/// energy margin after reserve, applicable continuous/peak power margins, and
/// peak-duration margin. These dimensions are deliberately not collapsed into
/// one weighted score. Only FeasibleWithinScreen rows can enter the frontier;
/// infeasible or indeterminate alternatives remain visible in rows.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioSweepReport {
    pub mission_id: String,
    pub rows: Vec<ScenarioRow>,
    pub pareto_front_ids: Vec<String>,
    pub feasible_count: usize,
    pub infeasible_count: usize,
    pub indeterminate_count: usize,
}

/// Compare alternative energy-system envelopes against one frozen mission.
///
/// Candidate identifiers are required to be unique, non-empty, and free of
/// leading/trailing whitespace. Output rows and frontier identifiers are
/// sorted lexicographically, independently of the input order. This function
/// does not mutate inputs and does not infer missing system properties.
///
/// A frontier membership means only "non-dominated within this set and this
/// crate's screening model". It does not imply aircraft feasibility or safety.
pub fn compare_systems_on_mission(
    mission: &MissionProfile,
    candidates: &[SystemCandidate],
) -> Result<ScenarioSweepReport, SweepInputError> {
    if candidates.is_empty() {
        return Err(SweepInputError::EmptyCandidateSet);
    }

    let mut seen = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        let id = candidate.id.as_str();
        if id.trim().is_empty() || id.trim() != id {
            return Err(SweepInputError::InvalidCandidateId {
                id: candidate.id.clone(),
            });
        }
        if !seen.insert(id.to_owned()) {
            return Err(SweepInputError::DuplicateCandidateId {
                id: candidate.id.clone(),
            });
        }
    }

    let mut rows: Vec<ScenarioRow> = candidates
        .iter()
        .map(|candidate| ScenarioRow {
            candidate_id: candidate.id.clone(),
            result: evaluate_mission(mission, &candidate.system),
            on_pareto_front: false,
        })
        .collect();
    rows.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));

    for index in 0..rows.len() {
        if rows[index].result.disposition != Disposition::FeasibleWithinScreen {
            continue;
        }
        let is_dominated = rows.iter().enumerate().any(|(other_index, other)| {
            other_index != index
                && other.result.disposition == Disposition::FeasibleWithinScreen
                && dominates(&other.result, &rows[index].result)
        });
        rows[index].on_pareto_front = !is_dominated;
    }

    let feasible_count = rows
        .iter()
        .filter(|row| row.result.disposition == Disposition::FeasibleWithinScreen)
        .count();
    let infeasible_count = rows
        .iter()
        .filter(|row| row.result.disposition == Disposition::InfeasibleWithinScreen)
        .count();
    let indeterminate_count = rows
        .iter()
        .filter(|row| row.result.disposition == Disposition::Indeterminate)
        .count();
    let pareto_front_ids = rows
        .iter()
        .filter(|row| row.on_pareto_front)
        .map(|row| row.candidate_id.clone())
        .collect();

    Ok(ScenarioSweepReport {
        mission_id: mission.id.clone(),
        rows,
        pareto_front_ids,
        feasible_count,
        infeasible_count,
        indeterminate_count,
    })
}

/// True when a is no worse than b on every objective and strictly better on at
/// least one. A missing dimension must match on both candidates; results for a
/// shared mission normally have the same applicable dimensions.
fn dominates(a: &ScreeningResult, b: &ScreeningResult) -> bool {
    let (Some(a_mass), Some(b_mass)) = (a.installed_mass_kg, b.installed_mass_kg) else {
        return false;
    };
    if a_mass > b_mass {
        return false;
    }
    let mut strictly_better = a_mass < b_mass;

    for (a_value, b_value) in [
        (a.energy_margin_wh, b.energy_margin_wh),
        (
            a.minimum_continuous_power_margin_w,
            b.minimum_continuous_power_margin_w,
        ),
        (a.peak_power_margin_w, b.peak_power_margin_w),
        (a.peak_duration_margin_s, b.peak_duration_margin_s),
    ] {
        match (a_value, b_value) {
            (Some(a_value), Some(b_value)) => {
                if a_value < b_value {
                    return false;
                }
                if a_value > b_value {
                    strictly_better = true;
                }
            }
            (None, None) => {}
            _ => return false,
        }
    }

    strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DutyClass, MissionSegment};

    fn mission() -> MissionProfile {
        MissionProfile {
            id: "road-air-reference-mission-v1".into(),
            segments: vec![
                MissionSegment::new("cruise", 3_600.0, 500.0, DutyClass::Continuous),
                MissionSegment::new("climb", 10.0, 1_000.0, DutyClass::Peak),
            ],
        }
    }

    fn candidate(
        id: &str,
        mass_kg: f64,
        energy_wh: f64,
        continuous_w: f64,
        peak_w: f64,
        peak_duration_s: f64,
    ) -> SystemCandidate {
        SystemCandidate {
            id: id.into(),
            system: EnergySystemEnvelope {
                installed_mass_kg: Some(mass_kg),
                usable_energy_at_bus_wh: Some(energy_wh),
                reserve_fraction: Some(0.1),
                max_continuous_bus_power_w: Some(continuous_w),
                max_peak_bus_power_w: Some(peak_w),
                max_peak_duration_s: Some(peak_duration_s),
            },
        }
    }

    #[test]
    fn pareto_front_preserves_mass_vs_margin_tradeoff() {
        let candidates = vec![
            candidate("light", 100.0, 1_500.0, 600.0, 1_200.0, 30.0),
            candidate("high-margin", 140.0, 2_500.0, 700.0, 1_300.0, 40.0),
            candidate("dominated", 150.0, 1_300.0, 550.0, 1_100.0, 15.0),
        ];

        let report = compare_systems_on_mission(&mission(), &candidates).unwrap();

        assert_eq!(report.feasible_count, 3);
        assert_eq!(report.infeasible_count, 0);
        assert_eq!(report.indeterminate_count, 0);
        assert_eq!(report.pareto_front_ids, vec!["high-margin", "light"]);
        let dominated = report.rows.iter().find(|row| row.candidate_id == "dominated").unwrap();
        assert!(!dominated.on_pareto_front);
    }

    #[test]
    fn input_order_does_not_change_rows_or_frontier() {
        let first = candidate("alpha", 100.0, 1_500.0, 600.0, 1_200.0, 30.0);
        let second = candidate("beta", 140.0, 2_500.0, 700.0, 1_300.0, 40.0);

        let forward = compare_systems_on_mission(&mission(), &[first.clone(), second.clone()]).unwrap();
        let reverse = compare_systems_on_mission(&mission(), &[second, first]).unwrap();

        assert_eq!(forward.rows, reverse.rows);
        assert_eq!(forward.pareto_front_ids, reverse.pareto_front_ids);
        assert_eq!(forward.pareto_front_ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn infeasible_and_indeterminate_candidates_remain_visible_but_never_rank() {
        let mut missing = candidate("unknown", 100.0, 1_500.0, 600.0, 1_200.0, 30.0);
        missing.system.usable_energy_at_bus_wh = None;
        let candidates = vec![
            candidate("infeasible", 100.0, 100.0, 600.0, 1_200.0, 30.0),
            missing,
            candidate("feasible", 120.0, 1_500.0, 600.0, 1_200.0, 30.0),
        ];

        let report = compare_systems_on_mission(&mission(), &candidates).unwrap();

        assert_eq!(report.feasible_count, 1);
        assert_eq!(report.infeasible_count, 1);
        assert_eq!(report.indeterminate_count, 1);
        assert_eq!(report.pareto_front_ids, vec!["feasible"]);
        assert!(report.rows.iter().any(|row| row.candidate_id == "infeasible"));
        assert!(report.rows.iter().any(|row| row.candidate_id == "unknown"));
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let candidates = vec![
            candidate("same", 100.0, 1_500.0, 600.0, 1_200.0, 30.0),
            candidate("same", 110.0, 1_600.0, 620.0, 1_220.0, 30.0),
        ];
        assert_eq!(
            compare_systems_on_mission(&mission(), &candidates),
            Err(SweepInputError::DuplicateCandidateId { id: "same".into() })
        );
    }

    #[test]
    fn empty_and_noncanonical_ids_are_rejected() {
        assert_eq!(
            compare_systems_on_mission(&mission(), &[]),
            Err(SweepInputError::EmptyCandidateSet)
        );
        assert_eq!(
            compare_systems_on_mission(
                &mission(),
                &[candidate(" alpha", 100.0, 1_500.0, 600.0, 1_200.0, 30.0)]
            ),
            Err(SweepInputError::InvalidCandidateId { id: " alpha".into() })
        );
    }
}
