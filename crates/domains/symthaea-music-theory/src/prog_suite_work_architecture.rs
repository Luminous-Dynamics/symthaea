// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical ProgSuite projection into work-scale tonal + metric architecture.
//!
//! This adapter makes no new native composition choices. It retains the
//! existing [`crate::prog_suite_work_bridge::ProgSuiteWorkBindingV1`], derives
//! both architecture views from that exact declaration, and requires the metric
//! architecture to project back to the already-authoritative temporal map.

use crate::prog_suite::{ProgSuitePlanV1, ProgSuitePlanErrorV1};
use crate::prog_suite_work_bridge::{
    ProgSuiteWorkBindingV1, ProgSuiteWorkBridgeErrorV1, bridge_prog_suite_plan,
};
use crate::rhythm::Duration;
use crate::temporal_map::TemporalMapErrorV1;
use crate::work_metric_architecture::{
    MetricArchitectureCoverageV1, MetricRegionRoleV1, MetricRegionV1,
    MetricTransitionRelationV1, MetricTransitionV1, WorkMetricArchitectureErrorV1,
    WorkMetricArchitectureV1, WORK_METRIC_ARCHITECTURE_VERSION,
};
use crate::work_tonal_trajectory::{
    TonalClosurePolicyV1, TonalRegionRoleV1, TonalRegionV1, TonalTrajectoryCoverageV1,
    TonalTransitionIntentV1, TonalTransitionV1, WorkTonalTrajectoryErrorV1,
    WorkTonalTrajectoryV1, WORK_TONAL_TRAJECTORY_VERSION, classify_tonal_relation,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PROG_SUITE_WORK_ARCHITECTURE_VERSION: &str =
    "melothaea-prog-suite-work-architecture-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteWorkArchitectureV1 {
    pub version: String,
    /// Complete canonical FORM declaration retained as source authority.
    pub binding: ProgSuiteWorkBindingV1,
    pub tonal_trajectory: WorkTonalTrajectoryV1,
    pub metric_architecture: WorkMetricArchitectureV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteWorkArchitectureErrorV1 {
    NativePlan(ProgSuitePlanErrorV1),
    WorkBridge(ProgSuiteWorkBridgeErrorV1),
    TemporalMap(TemporalMapErrorV1),
    TonalTrajectory(WorkTonalTrajectoryErrorV1),
    MetricArchitecture(WorkMetricArchitectureErrorV1),
    SectionBindingCountMismatch { expected: usize, found: usize },
    SectionBindingIndexMismatch { expected: usize, found: usize },
    DuplicateSectionWorkNode { work_node_id: String },
    TemporalAuthorityMismatch,
    WrongVersion { found: String },
    CanonicalArchitectureMismatch,
}

pub fn derive_prog_suite_work_architecture(
    plan: &ProgSuitePlanV1,
) -> Result<ProgSuiteWorkArchitectureV1, ProgSuiteWorkArchitectureErrorV1> {
    plan.validate()
        .map_err(ProgSuiteWorkArchitectureErrorV1::NativePlan)?;
    let binding = bridge_prog_suite_plan(plan)
        .map_err(ProgSuiteWorkArchitectureErrorV1::WorkBridge)?;
    if binding.section_bindings.len() != plan.sections.len() {
        return Err(
            ProgSuiteWorkArchitectureErrorV1::SectionBindingCountMismatch {
                expected: plan.sections.len(),
                found: binding.section_bindings.len(),
            },
        );
    }
    for (expected, section_binding) in binding.section_bindings.iter().enumerate() {
        if section_binding.section_index != expected {
            return Err(ProgSuiteWorkArchitectureErrorV1::SectionBindingIndexMismatch {
                expected,
                found: section_binding.section_index,
            });
        }
    }

    let tonal_trajectory = derive_tonal_trajectory(&binding)?;
    tonal_trajectory
        .validate(&binding.work_plan)
        .map_err(ProgSuiteWorkArchitectureErrorV1::TonalTrajectory)?;

    let metric_architecture = derive_metric_architecture(&binding)?;
    metric_architecture
        .validate(&binding.work_plan)
        .map_err(ProgSuiteWorkArchitectureErrorV1::MetricArchitecture)?;
    let projected_map = metric_architecture
        .to_temporal_map(&binding.work_plan)
        .map_err(ProgSuiteWorkArchitectureErrorV1::MetricArchitecture)?;
    if projected_map != binding.temporal_map {
        return Err(ProgSuiteWorkArchitectureErrorV1::TemporalAuthorityMismatch);
    }

    Ok(ProgSuiteWorkArchitectureV1 {
        version: PROG_SUITE_WORK_ARCHITECTURE_VERSION.into(),
        binding,
        tonal_trajectory,
        metric_architecture,
    })
}

impl ProgSuiteWorkArchitectureV1 {
    /// Rebuild the complete adapter from retained native declaration and require
    /// exact equality. Serialized architecture views therefore cannot become a
    /// second editable source of ProgSuite planning authority.
    pub fn validate(&self) -> Result<(), ProgSuiteWorkArchitectureErrorV1> {
        if self.version != PROG_SUITE_WORK_ARCHITECTURE_VERSION {
            return Err(ProgSuiteWorkArchitectureErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        let canonical = derive_prog_suite_work_architecture(&self.binding.native_plan)?;
        if &canonical != self {
            return Err(ProgSuiteWorkArchitectureErrorV1::CanonicalArchitectureMismatch);
        }
        Ok(())
    }
}

fn derive_tonal_trajectory(
    binding: &ProgSuiteWorkBindingV1,
) -> Result<WorkTonalTrajectoryV1, ProgSuiteWorkArchitectureErrorV1> {
    let plan = &binding.native_plan;
    let mut regions = BTreeMap::new();
    let mut region_order = Vec::with_capacity(plan.sections.len());

    for (index, section) in plan.sections.iter().enumerate() {
        let work_node_id = binding.section_bindings[index].work_node_id.clone();
        let role = if index == 0 {
            TonalRegionRoleV1::Home
        } else if index + 1 == plan.sections.len() && section.key == plan.home_key {
            TonalRegionRoleV1::Return
        } else if section.key == plan.home_key {
            TonalRegionRoleV1::Home
        } else {
            TonalRegionRoleV1::Departure
        };
        if regions
            .insert(
                work_node_id.clone(),
                TonalRegionV1 {
                    work_node_id: work_node_id.clone(),
                    start: section.start,
                    end: section.end,
                    key: section.key,
                    role,
                },
            )
            .is_some()
        {
            return Err(ProgSuiteWorkArchitectureErrorV1::DuplicateSectionWorkNode {
                work_node_id,
            });
        }
        region_order.push(work_node_id);
    }

    let mut transitions = Vec::with_capacity(region_order.len().saturating_sub(1));
    for index in 0..region_order.len().saturating_sub(1) {
        let from = &plan.sections[index];
        let to = &plan.sections[index + 1];
        let intent = if from.key != plan.home_key && to.key == plan.home_key {
            TonalTransitionIntentV1::ReturnHome
        } else if from.key == to.key {
            TonalTransitionIntentV1::Hold
        } else if from.key == plan.home_key {
            TonalTransitionIntentV1::Depart
        } else {
            TonalTransitionIntentV1::Develop
        };
        transitions.push(TonalTransitionV1 {
            from_region_id: region_order[index].clone(),
            to_region_id: region_order[index + 1].clone(),
            intent,
            relation: classify_tonal_relation(from.key, to.key),
        });
    }

    Ok(WorkTonalTrajectoryV1 {
        version: WORK_TONAL_TRAJECTORY_VERSION.into(),
        trajectory_id: "prog-suite:tonal-trajectory-v1".into(),
        home_key: plan.home_key,
        coverage: TonalTrajectoryCoverageV1::CompleteWork,
        closure: TonalClosurePolicyV1::ReturnToHome,
        regions,
        region_order,
        transitions,
    })
}

fn derive_metric_architecture(
    binding: &ProgSuiteWorkBindingV1,
) -> Result<WorkMetricArchitectureV1, ProgSuiteWorkArchitectureErrorV1> {
    let plan = &binding.native_plan;
    let mut regions = BTreeMap::new();
    let mut region_order = Vec::with_capacity(plan.sections.len());

    for (index, section) in plan.sections.iter().enumerate() {
        let work_node_id = binding.section_bindings[index].work_node_id.clone();
        let meter = binding
            .temporal_map
            .meter_at(section.start)
            .map_err(ProgSuiteWorkArchitectureErrorV1::TemporalMap)?;
        let tempo = binding
            .temporal_map
            .tempo_at(section.start)
            .map_err(ProgSuiteWorkArchitectureErrorV1::TemporalMap)?;
        let role = metric_role(index, plan);
        if regions
            .insert(
                work_node_id.clone(),
                MetricRegionV1 {
                    work_node_id: work_node_id.clone(),
                    start: section.start,
                    end: section.end,
                    meter,
                    tempo,
                    role,
                },
            )
            .is_some()
        {
            return Err(ProgSuiteWorkArchitectureErrorV1::DuplicateSectionWorkNode {
                work_node_id,
            });
        }
        region_order.push(work_node_id);
    }

    let quarter = Duration::new(1, 1);
    let transitions = region_order
        .windows(2)
        .map(|pair| MetricTransitionV1 {
            from_region_id: pair[0].clone(),
            to_region_id: pair[1].clone(),
            // Native ProgSuite changes written bar length while retaining one
            // global quarter-note tempo. State the exact surviving pulse rather
            // than relabeling it as a stronger metric modulation.
            relation: MetricTransitionRelationV1::PulseEquivalence {
                source_pulse: quarter,
                target_pulse: quarter,
            },
        })
        .collect();

    Ok(WorkMetricArchitectureV1 {
        version: WORK_METRIC_ARCHITECTURE_VERSION.into(),
        architecture_id: "prog-suite:metric-architecture-v1".into(),
        coverage: MetricArchitectureCoverageV1::CompleteWork,
        regions,
        region_order,
        transitions,
    })
}

fn metric_role(index: usize, plan: &ProgSuitePlanV1) -> MetricRegionRoleV1 {
    if index == 0 {
        return MetricRegionRoleV1::Stable;
    }
    let current = plan.sections[index].meter;
    if index + 1 == plan.sections.len() && current == plan.sections[0].meter {
        return MetricRegionRoleV1::Return;
    }
    let previous = plan.sections[index - 1].meter;
    if current > previous {
        MetricRegionRoleV1::Expansion
    } else if current < previous {
        MetricRegionRoleV1::Contraction
    } else {
        MetricRegionRoleV1::Stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Key, PitchClass, Style, TonalRelationV1};

    fn plan(seed: u64) -> ProgSuitePlanV1 {
        crate::plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            seed,
            &Style::ProgFolk.spec(),
        )
        .unwrap()
    }

    #[test]
    fn native_prog_suite_derives_complete_tonal_and_metric_architecture() {
        let architecture = derive_prog_suite_work_architecture(&plan(5)).unwrap();
        architecture.validate().unwrap();
        assert_eq!(architecture.tonal_trajectory.region_order.len(), 4);
        assert_eq!(architecture.metric_architecture.region_order.len(), 4);
        assert_eq!(
            architecture.metric_architecture.to_temporal_map(&architecture.binding.work_plan).unwrap(),
            architecture.binding.temporal_map
        );
    }

    #[test]
    fn tonal_arc_retains_home_home_relative_home_without_inventing_b_modulation() {
        let architecture = derive_prog_suite_work_architecture(&plan(5)).unwrap();
        let order = &architecture.tonal_trajectory.region_order;
        let tonal = &architecture.tonal_trajectory;
        assert_eq!(tonal.regions[&order[0]].key, architecture.binding.native_plan.home_key);
        assert_eq!(tonal.regions[&order[1]].key, architecture.binding.native_plan.home_key);
        assert_eq!(
            tonal.regions[&order[2]].key,
            architecture.binding.native_plan.home_key.relative()
        );
        assert_eq!(tonal.regions[&order[3]].key, architecture.binding.native_plan.home_key);
        assert_eq!(tonal.transitions[0].relation, TonalRelationV1::SameKey);
        assert_eq!(
            tonal.transitions[1].relation,
            TonalRelationV1::SourceToRelative
        );
        assert_eq!(
            tonal.transitions[2].relation,
            TonalRelationV1::SourceToRelative
        );
    }

    #[test]
    fn metric_arc_is_four_seven_five_four_with_quarter_pulse_continuity() {
        let architecture = derive_prog_suite_work_architecture(&plan(5)).unwrap();
        let metric = &architecture.metric_architecture;
        let numerators: Vec<_> = metric
            .region_order
            .iter()
            .map(|id| metric.regions[id].meter.numerator())
            .collect();
        assert_eq!(numerators, vec![4, 7, 5, 4]);
        assert_eq!(metric.regions[&metric.region_order[0]].role, MetricRegionRoleV1::Stable);
        assert_eq!(metric.regions[&metric.region_order[1]].role, MetricRegionRoleV1::Expansion);
        assert_eq!(metric.regions[&metric.region_order[2]].role, MetricRegionRoleV1::Contraction);
        assert_eq!(metric.regions[&metric.region_order[3]].role, MetricRegionRoleV1::Return);
        for transition in &metric.transitions {
            assert_eq!(
                transition.relation,
                MetricTransitionRelationV1::PulseEquivalence {
                    source_pulse: Duration::new(1, 1),
                    target_pulse: Duration::new(1, 1),
                }
            );
        }
    }

    #[test]
    fn metric_projection_reconciles_with_preexisting_temporal_authority() {
        let architecture = derive_prog_suite_work_architecture(&plan(17)).unwrap();
        let projected = architecture
            .metric_architecture
            .to_temporal_map(&architecture.binding.work_plan)
            .unwrap();
        assert_eq!(projected, architecture.binding.temporal_map);
    }

    #[test]
    fn serialized_tonal_tampering_cannot_become_native_authority() {
        let mut architecture = derive_prog_suite_work_architecture(&plan(5)).unwrap();
        architecture.tonal_trajectory.transitions[1].relation = TonalRelationV1::Other;
        assert_eq!(
            architecture.validate(),
            Err(ProgSuiteWorkArchitectureErrorV1::CanonicalArchitectureMismatch)
        );
    }
}
