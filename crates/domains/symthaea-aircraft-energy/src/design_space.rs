// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded, deterministic battery-system design-space screening.
//!
//! The sweep varies explicit pack/system assumptions against one frozen
//! MissionProfile. It does not re-compute aerodynamic or propulsive demand when
//! system mass changes. Therefore it is a first-pass sensitivity tool only;
//! a vehicle-level model must recompute mission loads for each candidate before
//! any mass-sensitive comparison is interpreted as a fair aircraft trade.

use std::collections::{BTreeMap, HashMap};

use super::{
    BatteryPackSpec, Disposition, EnergySystemEnvelope, Finding, MissionProfile,
    ScreeningResult, SystemCandidate, SweepInputError, compare_systems_on_mission,
    evaluate_mission,
};

/// Hard limit for the current pairwise Pareto implementation, whose comparison
/// cost is quadratic in the number of candidates. A larger study must use
/// reviewed batches and a separately verified frontier-merging procedure.
pub const MAX_BATTERY_DESIGN_POINTS: usize = 2_000;

/// Independent axes for an explicit battery-system sensitivity sweep.
///
/// Other BatteryPackSpec properties remain fixed at template values. Each
/// axis is canonicalized by total float order and exact-bit duplicate removal;
/// changing input axis order therefore does not change the generated cases.
#[derive(Debug, Clone, PartialEq)]
pub struct BatteryDesignSpace {
    pub pack_specific_energy_wh_per_kg: Vec<f64>,
    /// Complete installed energy-system mass divided by installed pack mass.
    /// Values below 1.0 are preserved as invalid designs, not silently removed.
    pub installed_system_mass_multiplier: Vec<f64>,
    pub temperature_capacity_fractions: Vec<f64>,
    pub end_of_life_soh_fractions: Vec<f64>,
    pub reserve_fractions: Vec<f64>,
    pub power_derating_fractions: Vec<f64>,
}

/// Failure to define or execute a bounded sweep. Individual invalid design
/// points are not errors at this level; they remain as rejected report rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DesignSpaceError {
    EmptyAxis { name: &'static str },
    TooManyDesignPoints { requested: usize, maximum: usize },
    CandidateComparisonRejected { cause: SweepInputError },
}

/// One generated design point and its complete arithmetic-screen disposition.
#[derive(Debug, Clone, PartialEq)]
pub struct BatteryDesignRow {
    pub candidate_id: String,
    pub spec: BatteryPackSpec,
    pub configuration_valid: bool,
    pub configuration_findings: Vec<Finding>,
    pub envelope: Option<EnergySystemEnvelope>,
    pub screening: ScreeningResult,
    /// Non-dominated only among valid candidates with the same reserve fraction.
    pub on_pareto_front: bool,
}

/// One frontier for a single common reserve fraction.
#[derive(Debug, Clone, PartialEq)]
pub struct ReservePolicyFrontier {
    pub reserve_fraction: f64,
    pub candidate_ids: Vec<String>,
}

/// Result of a bounded sweep. A frontier is valid only for this explicit,
/// frozen mission and this arithmetic model. It is not proof of aircraft
/// feasibility. The current model does not propagate changes in system mass
/// back into required power or energy.
#[derive(Debug, Clone, PartialEq)]
pub struct BatteryDesignSweepReport {
    pub mission_id: String,
    pub rows: Vec<BatteryDesignRow>,
    /// A separate frontier per reserve policy; candidates with different
    /// reserve fractions are not ranked against one another.
    pub pareto_frontiers: Vec<ReservePolicyFrontier>,
    pub feasible_count: usize,
    pub infeasible_count: usize,
    /// Includes invalid configurations and other missing/invalid evidence.
    pub indeterminate_count: usize,
    pub invalid_configuration_count: usize,
    pub valid_configuration_indeterminate_count: usize,
}

/// Generate and screen the Cartesian product of a bounded battery design space.
///
/// Invalid battery-spec combinations are retained in the report with their
/// explicit findings and are never considered for Pareto ranking. For valid
/// candidates the existing deterministic comparison engine defines the
/// frontier; no weighted composite score is invented.
///
/// Important model boundary: all candidates are checked against the same
/// already-specified mission bus-power profile. This function does not account
/// for how a different battery/system mass changes aircraft lift, drag,
/// climb, or propulsion demand. Before making aircraft-level decisions, build
/// a validated vehicle-level model and regenerate mission loads per candidate.
pub fn sweep_battery_design_space(
    mission: &MissionProfile,
    template: &BatteryPackSpec,
    design_space: &BatteryDesignSpace,
) -> Result<BatteryDesignSweepReport, DesignSpaceError> {
    let mut axes = [
        ("pack_specific_energy_wh_per_kg", design_space.pack_specific_energy_wh_per_kg.clone()),
        ("installed_system_mass_multiplier", design_space.installed_system_mass_multiplier.clone()),
        ("temperature_capacity_fractions", design_space.temperature_capacity_fractions.clone()),
        ("end_of_life_soh_fractions", design_space.end_of_life_soh_fractions.clone()),
        ("reserve_fractions", design_space.reserve_fractions.clone()),
        ("power_derating_fractions", design_space.power_derating_fractions.clone()),
    ];

    for (name, axis) in &mut axes {
        if axis.is_empty() {
            return Err(DesignSpaceError::EmptyAxis { name: *name });
        }
        // Negative zero and positive zero represent the same physical fraction;
        // canonicalize before bitwise deduplication to avoid duplicate designs.
        for value in axis.iter_mut() {
            if *value == 0.0 {
                *value = 0.0;
            }
        }
        axis.sort_by(f64::total_cmp);
        axis.dedup_by(|left, right| left.to_bits() == right.to_bits());
    }

    let requested = axes.iter().try_fold(1usize, |count, (_, axis)| {
        count.checked_mul(axis.len())
    }).ok_or(DesignSpaceError::TooManyDesignPoints {
        requested: usize::MAX,
        maximum: MAX_BATTERY_DESIGN_POINTS,
    })?;
    if requested > MAX_BATTERY_DESIGN_POINTS {
        return Err(DesignSpaceError::TooManyDesignPoints {
            requested,
            maximum: MAX_BATTERY_DESIGN_POINTS,
        });
    }

    let mut rows = Vec::with_capacity(requested);
    let mut candidates_by_reserve: BTreeMap<u64, Vec<SystemCandidate>> = BTreeMap::new();

    for linear_index in 0..requested {
        let mut remainder = linear_index;
        let mut coordinates = [0usize; 6];
        for axis_index in (0..axes.len()).rev() {
            coordinates[axis_index] = remainder % axes[axis_index].1.len();
            remainder /= axes[axis_index].1.len();
        }

        let value = |axis_index: usize| axes[axis_index].1[coordinates[axis_index]];
        let mut spec = template.clone();
        spec.pack_specific_energy_wh_per_kg = value(0);
        spec.installed_system_mass_kg = spec.installed_pack_mass_kg * value(1);
        spec.temperature_capacity_fraction = value(2);
        spec.end_of_life_soh_fraction = value(3);
        spec.reserve_fraction = value(4);
        spec.power_derating_fraction = value(5);
        let candidate_id = format!("case-{linear_index:05}");

        match spec.to_envelope() {
            Ok(envelope) => {
                let screening = evaluate_mission(mission, &envelope);
                if let Some(reserve) = envelope.reserve_fraction {
                    candidates_by_reserve
                        .entry(reserve.to_bits())
                        .or_default()
                        .push(SystemCandidate {
                            id: candidate_id.clone(),
                            system: envelope.clone(),
                        });
                } else {
                    // BatteryPackSpec::to_envelope currently always supplies
                    // reserve. Keep this fail-closed if that contract changes.
                    let finding = Finding::MissingInput {
                        field: "system.reserve_fraction".into(),
                    };
                    rows.push(BatteryDesignRow {
                        candidate_id,
                        spec,
                        configuration_valid: false,
                        configuration_findings: vec![finding.clone()],
                        envelope: None,
                        screening: ScreeningResult {
                            disposition: Disposition::Indeterminate,
                            required_mission_energy_wh: None,
                            available_energy_after_reserve_wh: None,
                            energy_margin_wh: None,
                            minimum_continuous_power_margin_w: None,
                            peak_power_margin_w: None,
                            peak_duration_margin_s: None,
                            installed_mass_kg: None,
                            findings: vec![finding],
                        },
                        on_pareto_front: false,
                    });
                    continue;
                }
                rows.push(BatteryDesignRow {
                    candidate_id,
                    spec,
                    configuration_valid: true,
                    configuration_findings: Vec::new(),
                    envelope: Some(envelope),
                    screening,
                    on_pareto_front: false,
                });
            }
            Err(findings) => {
                let invalid_screen = ScreeningResult {
                    disposition: Disposition::Indeterminate,
                    required_mission_energy_wh: None,
                    available_energy_after_reserve_wh: None,
                    energy_margin_wh: None,
                    minimum_continuous_power_margin_w: None,
                    peak_power_margin_w: None,
                    peak_duration_margin_s: None,
                    installed_mass_kg: None,
                    findings: findings.clone(),
                };
                rows.push(BatteryDesignRow {
                    candidate_id,
                    spec,
                    configuration_valid: false,
                    configuration_findings: findings,
                    envelope: None,
                    screening: invalid_screen,
                    on_pareto_front: false,
                });
            }
        }
    }

    let mut pareto_frontiers = Vec::with_capacity(candidates_by_reserve.len());
    for (reserve_bits, candidates) in candidates_by_reserve {
        let comparison = compare_systems_on_mission(mission, &candidates)
            .map_err(|cause| DesignSpaceError::CandidateComparisonRejected { cause })?;
        let ranked: HashMap<String, _> = comparison
            .rows
            .into_iter()
            .map(|row| (row.candidate_id.clone(), row))
            .collect();
        for row in &mut rows {
            if let Some(ranked_row) = ranked.get(&row.candidate_id) {
                row.screening = ranked_row.result.clone();
                row.on_pareto_front = ranked_row.on_pareto_front;
            }
        }
        pareto_frontiers.push(ReservePolicyFrontier {
            reserve_fraction: f64::from_bits(reserve_bits),
            candidate_ids: comparison.pareto_front_ids,
        });
    }
    pareto_frontiers.sort_by(|a, b| a.reserve_fraction.total_cmp(&b.reserve_fraction));

    rows.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let feasible_count = rows
        .iter()
        .filter(|row| row.screening.disposition == Disposition::FeasibleWithinScreen)
        .count();
    let infeasible_count = rows
        .iter()
        .filter(|row| row.screening.disposition == Disposition::InfeasibleWithinScreen)
        .count();
    let indeterminate_count = rows
        .iter()
        .filter(|row| row.screening.disposition == Disposition::Indeterminate)
        .count();
    let invalid_configuration_count = rows.iter().filter(|row| !row.configuration_valid).count();
    let valid_configuration_indeterminate_count = rows
        .iter()
        .filter(|row| {
            row.configuration_valid
                && row.screening.disposition == Disposition::Indeterminate
        })
        .count();
    Ok(BatteryDesignSweepReport {
        mission_id: mission.id.clone(),
        rows,
        pareto_frontiers,
        feasible_count,
        infeasible_count,
        indeterminate_count,
        invalid_configuration_count,
        valid_configuration_indeterminate_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DutyClass, MissionSegment};

    fn template() -> BatteryPackSpec {
        BatteryPackSpec {
            installed_pack_mass_kg: 100.0,
            installed_system_mass_kg: 130.0,
            pack_specific_energy_wh_per_kg: 300.0,
            usable_fraction: 0.9,
            temperature_capacity_fraction: 0.85,
            end_of_life_soh_fraction: 0.9,
            discharge_efficiency_fraction: 0.95,
            continuous_specific_power_w_per_kg: 300.0,
            peak_specific_power_w_per_kg: 600.0,
            power_derating_fraction: 0.8,
            reserve_fraction: 0.1,
            max_peak_duration_s: 60.0,
        }
    }

    fn mission() -> MissionProfile {
        MissionProfile {
            id: "frozen-reference-mission-v1".into(),
            segments: vec![
                MissionSegment::new("cruise", 3_600.0, 5_000.0, DutyClass::Continuous),
                MissionSegment::new("climb", 20.0, 15_000.0, DutyClass::Peak),
            ],
        }
    }

    fn space() -> BatteryDesignSpace {
        BatteryDesignSpace {
            pack_specific_energy_wh_per_kg: vec![300.0, 400.0],
            installed_system_mass_multiplier: vec![1.3, 1.4],
            temperature_capacity_fractions: vec![0.8, 0.9],
            end_of_life_soh_fractions: vec![0.8, 0.9],
            reserve_fractions: vec![0.1, 0.2],
            power_derating_fractions: vec![0.7, 0.8],
        }
    }

    #[test]
    fn cartesian_product_count_and_frontier_are_deterministic() {
        let report = sweep_battery_design_space(&mission(), &template(), &space()).unwrap();
        assert_eq!(report.rows.len(), 64);
        assert_eq!(report.invalid_configuration_count, 0);
        assert_eq!(report.feasible_count, 64);
        assert_eq!(report.pareto_frontiers.len(), 2);
        assert!(report.pareto_frontiers.iter().all(|frontier| !frontier.candidate_ids.is_empty()));
        assert!(report.pareto_frontiers[0].reserve_fraction < report.pareto_frontiers[1].reserve_fraction);
        assert!(report.rows.iter().all(|row| {
            row.configuration_valid
                && row.envelope.is_some()
                && row.screening.disposition == Disposition::FeasibleWithinScreen
        }));
        assert!(report.rows.windows(2).all(|pair| pair[0].candidate_id < pair[1].candidate_id));
    }

    #[test]
    fn reordering_axes_does_not_change_the_generated_report() {
        let first_space = space();
        let mut second_space = first_space.clone();
        second_space.pack_specific_energy_wh_per_kg.reverse();
        second_space.installed_system_mass_multiplier.reverse();
        second_space.temperature_capacity_fractions.reverse();
        second_space.end_of_life_soh_fractions.reverse();
        second_space.reserve_fractions.reverse();
        second_space.power_derating_fractions.reverse();

        let first = sweep_battery_design_space(&mission(), &template(), &first_space).unwrap();
        let second = sweep_battery_design_space(&mission(), &template(), &second_space).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_configurations_are_preserved_but_never_ranked() {
        let mut design_space = space();
        design_space.installed_system_mass_multiplier = vec![0.9];
        design_space.pack_specific_energy_wh_per_kg = vec![300.0];
        design_space.temperature_capacity_fractions = vec![0.8];
        design_space.end_of_life_soh_fractions = vec![0.8];
        design_space.reserve_fractions = vec![0.1];
        design_space.power_derating_fractions = vec![0.8];

        let report = sweep_battery_design_space(&mission(), &template(), &design_space).unwrap();
        assert_eq!(report.rows.len(), 1);
        assert_eq!(report.invalid_configuration_count, 1);
        assert_eq!(report.indeterminate_count, 1);
        assert!(report.pareto_frontiers.is_empty());
        assert!(!report.rows[0].configuration_valid);
        assert!(!report.rows[0].on_pareto_front);
        assert!(!report.rows[0].configuration_findings.is_empty());
    }

    #[test]
    fn signed_zero_is_canonicalized_before_axis_deduplication() {
        let mut design_space = space();
        design_space.reserve_fractions = vec![-0.0, 0.0];
        let report = sweep_battery_design_space(&mission(), &template(), &design_space).unwrap();
        assert_eq!(report.rows.len(), 32);
        assert_eq!(report.pareto_frontiers.len(), 1);
        assert_eq!(report.pareto_frontiers[0].reserve_fraction, 0.0);
    }

    #[test]
    fn empty_axis_is_rejected() {
        let mut design_space = space();
        design_space.reserve_fractions.clear();
        assert_eq!(
            sweep_battery_design_space(&mission(), &template(), &design_space),
            Err(DesignSpaceError::EmptyAxis { name: "reserve_fractions" })
        );
    }

    #[test]
    fn oversized_cartesian_product_is_rejected_before_allocation() {
        let design_space = BatteryDesignSpace {
            pack_specific_energy_wh_per_kg: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            installed_system_mass_multiplier: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            temperature_capacity_fractions: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            end_of_life_soh_fractions: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            reserve_fractions: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            power_derating_fractions: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };
        assert_eq!(
            sweep_battery_design_space(&mission(), &template(), &design_space),
            Err(DesignSpaceError::TooManyDesignPoints {
                requested: 1_000_000,
                maximum: MAX_BATTERY_DESIGN_POINTS,
            })
        );
    }
}
