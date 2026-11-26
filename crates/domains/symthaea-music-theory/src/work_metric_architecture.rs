// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Work-scale metric and pulse architecture.
//!
//! [`crate::temporal_map::TemporalMapV1`] is the exact timeline authority for
//! piecewise-constant written meter and quarter-note tempo. This module sits one
//! level above it: it freezes the large-scale metric *architecture* that gives
//! those changes narrative identity and, when requested, an exact pulse or bar
//! duration equivalence across a boundary.
//!
//! Equivalence is checked with rational integer arithmetic. V1 never uses a
//! floating tolerance and never promotes a valid plan into performed or
//! listener-perceived metric evidence.

use crate::meter::TimeSignature;
use crate::rhythm::Duration;
use crate::temporal_map::{TempoV1, TemporalMapErrorV1, TemporalMapV1};
use crate::work_plan::{HierarchicalWorkPlanV1, WorkPlanErrorV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const WORK_METRIC_ARCHITECTURE_VERSION: &str = "melothaea-work-metric-architecture-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricArchitectureCoverageV1 {
    CompleteWork,
    DeclaredRegionsOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MetricRegionRoleV1 {
    Stable,
    Expansion,
    Contraction,
    Asymmetry,
    Compound,
    Preparation,
    Return,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricRegionV1 {
    pub work_node_id: String,
    /// Exact half-open work span `[start, end)` in quarter-note beats.
    pub start: Duration,
    pub end: Duration,
    /// Full written-meter identity, including additive grouping.
    pub meter: TimeSignature,
    /// Exact rational quarter-note BPM.
    pub tempo: TempoV1,
    pub role: MetricRegionRoleV1,
}

/// Exact relation declared across one adjacent metric-region boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricTransitionRelationV1 {
    /// Written meter (including grouping) and quarter-note tempo are unchanged.
    Continuation,
    /// At least written meter/grouping or quarter-note tempo changes, with no
    /// stronger duration-equivalence claim.
    DirectChange,
    /// One source pulse duration equals one target pulse duration in seconds.
    /// Pulses are exact quarter-note-beat durations: quarter = 1/1,
    /// eighth = 1/2, dotted-quarter = 3/2, etc.
    PulseEquivalence {
        source_pulse: Duration,
        target_pulse: Duration,
    },
    /// The complete written source bar equals the complete target bar in
    /// elapsed time under the declared rational tempi.
    BarDurationEquivalence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricTransitionV1 {
    pub from_region_id: String,
    pub to_region_id: String,
    pub relation: MetricTransitionRelationV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkMetricArchitectureV1 {
    pub version: String,
    pub architecture_id: String,
    pub coverage: MetricArchitectureCoverageV1,
    pub regions: BTreeMap<String, MetricRegionV1>,
    pub region_order: Vec<String>,
    pub transitions: Vec<MetricTransitionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkMetricArchitectureErrorV1 {
    WrongVersion { found: String },
    InvalidWorkPlan(WorkPlanErrorV1),
    EmptyArchitectureId,
    EmptyRegions,
    RegionOrderCountMismatch { regions: usize, ordered: usize },
    EmptyRegionId { order_index: usize },
    UnknownOrderedRegion { region_id: String },
    DuplicateOrderedRegion { region_id: String },
    UnorderedRegion { region_id: String },
    EmptyWorkNode { region_id: String },
    MissingWorkNode { region_id: String, work_node_id: String },
    InvalidRegionSpan { region_id: String },
    RegionOutsideWorkNode { region_id: String, work_node_id: String },
    InvalidMeter { region_id: String },
    InvalidTempo { region_id: String },
    EmptyOtherRole { region_id: String },
    RegionOverlap { left_region_id: String, right_region_id: String },
    CompleteCoverageMustStartAtRoot,
    CompleteCoverageGap { left_region_id: String, right_region_id: String },
    CompleteCoverageMustEndAtRoot,
    TransitionCountMismatch { expected: usize, found: usize },
    TransitionAdjacencyMismatch { transition_index: usize },
    ContinuationChangedMetricState { transition_index: usize },
    DirectChangeDidNotChangeMetricState { transition_index: usize },
    NonPositivePulse { transition_index: usize },
    PulseEquivalenceMismatch { transition_index: usize },
    BarDurationEquivalenceMismatch { transition_index: usize },
    ProjectionRequiresCompleteCoverage,
    TemporalMap(TemporalMapErrorV1),
    QueryBeforeWork,
}

impl WorkMetricArchitectureV1 {
    pub fn validate(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
    ) -> Result<(), WorkMetricArchitectureErrorV1> {
        if self.version != WORK_METRIC_ARCHITECTURE_VERSION {
            return Err(WorkMetricArchitectureErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        work_plan
            .validate()
            .map_err(WorkMetricArchitectureErrorV1::InvalidWorkPlan)?;
        if self.architecture_id.trim().is_empty() {
            return Err(WorkMetricArchitectureErrorV1::EmptyArchitectureId);
        }
        if self.regions.is_empty() {
            return Err(WorkMetricArchitectureErrorV1::EmptyRegions);
        }
        if self.region_order.len() != self.regions.len() {
            return Err(WorkMetricArchitectureErrorV1::RegionOrderCountMismatch {
                regions: self.regions.len(),
                ordered: self.region_order.len(),
            });
        }

        let root = work_plan
            .nodes
            .get(&work_plan.root_id)
            .expect("validated work plan retains root");
        let mut seen = BTreeSet::new();
        let mut previous: Option<(&str, &MetricRegionV1)> = None;

        for (order_index, region_id) in self.region_order.iter().enumerate() {
            if region_id.trim().is_empty() {
                return Err(WorkMetricArchitectureErrorV1::EmptyRegionId { order_index });
            }
            let region = self.regions.get(region_id).ok_or_else(|| {
                WorkMetricArchitectureErrorV1::UnknownOrderedRegion {
                    region_id: region_id.clone(),
                }
            })?;
            if !seen.insert(region_id.clone()) {
                return Err(WorkMetricArchitectureErrorV1::DuplicateOrderedRegion {
                    region_id: region_id.clone(),
                });
            }
            if region.work_node_id.trim().is_empty() {
                return Err(WorkMetricArchitectureErrorV1::EmptyWorkNode {
                    region_id: region_id.clone(),
                });
            }
            let node = work_plan.nodes.get(&region.work_node_id).ok_or_else(|| {
                WorkMetricArchitectureErrorV1::MissingWorkNode {
                    region_id: region_id.clone(),
                    work_node_id: region.work_node_id.clone(),
                }
            })?;
            if compare_duration(region.start, region.end) != Ordering::Less {
                return Err(WorkMetricArchitectureErrorV1::InvalidRegionSpan {
                    region_id: region_id.clone(),
                });
            }
            if compare_duration(region.start, node.start) == Ordering::Less
                || compare_duration(region.end, node.end) == Ordering::Greater
            {
                return Err(WorkMetricArchitectureErrorV1::RegionOutsideWorkNode {
                    region_id: region_id.clone(),
                    work_node_id: region.work_node_id.clone(),
                });
            }
            if TimeSignature::with_grouping(
                region.meter.numerator(),
                region.meter.denominator(),
                region.meter.grouping().to_vec(),
            )
            .is_err()
            {
                return Err(WorkMetricArchitectureErrorV1::InvalidMeter {
                    region_id: region_id.clone(),
                });
            }
            let canonical_tempo = TempoV1::new(
                region.tempo.numerator_bpm(),
                region.tempo.denominator(),
            )
            .map_err(|_| WorkMetricArchitectureErrorV1::InvalidTempo {
                region_id: region_id.clone(),
            })?;
            if canonical_tempo != region.tempo {
                return Err(WorkMetricArchitectureErrorV1::InvalidTempo {
                    region_id: region_id.clone(),
                });
            }
            if matches!(&region.role, MetricRegionRoleV1::Other(label) if label.trim().is_empty()) {
                return Err(WorkMetricArchitectureErrorV1::EmptyOtherRole {
                    region_id: region_id.clone(),
                });
            }

            if let Some((previous_id, previous_region)) = previous {
                match compare_duration(previous_region.end, region.start) {
                    Ordering::Greater => {
                        return Err(WorkMetricArchitectureErrorV1::RegionOverlap {
                            left_region_id: previous_id.into(),
                            right_region_id: region_id.clone(),
                        });
                    }
                    Ordering::Less
                        if self.coverage == MetricArchitectureCoverageV1::CompleteWork =>
                    {
                        return Err(WorkMetricArchitectureErrorV1::CompleteCoverageGap {
                            left_region_id: previous_id.into(),
                            right_region_id: region_id.clone(),
                        });
                    }
                    _ => {}
                }
            }
            previous = Some((region_id.as_str(), region));
        }

        for region_id in self.regions.keys() {
            if !seen.contains(region_id) {
                return Err(WorkMetricArchitectureErrorV1::UnorderedRegion {
                    region_id: region_id.clone(),
                });
            }
        }

        let first = &self.regions[&self.region_order[0]];
        let last_id = self.region_order.last().expect("nonempty order");
        let last = &self.regions[last_id];
        if self.coverage == MetricArchitectureCoverageV1::CompleteWork {
            if first.start != root.start {
                return Err(WorkMetricArchitectureErrorV1::CompleteCoverageMustStartAtRoot);
            }
            if last.end != root.end {
                return Err(WorkMetricArchitectureErrorV1::CompleteCoverageMustEndAtRoot);
            }
        }

        let expected_transitions = self.region_order.len().saturating_sub(1);
        if self.transitions.len() != expected_transitions {
            return Err(WorkMetricArchitectureErrorV1::TransitionCountMismatch {
                expected: expected_transitions,
                found: self.transitions.len(),
            });
        }
        for (index, transition) in self.transitions.iter().enumerate() {
            let expected_from = &self.region_order[index];
            let expected_to = &self.region_order[index + 1];
            if &transition.from_region_id != expected_from || &transition.to_region_id != expected_to
            {
                return Err(WorkMetricArchitectureErrorV1::TransitionAdjacencyMismatch {
                    transition_index: index,
                });
            }
            let from = &self.regions[expected_from];
            let to = &self.regions[expected_to];
            validate_transition(index, &transition.relation, from, to)?;
        }
        Ok(())
    }

    /// Project a complete metric architecture into the exact temporal-map
    /// timeline authority. Declared-only architectures fail closed because gaps
    /// do not specify which meter/tempo remains authoritative.
    pub fn to_temporal_map(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
    ) -> Result<TemporalMapV1, WorkMetricArchitectureErrorV1> {
        self.validate(work_plan)?;
        if self.coverage != MetricArchitectureCoverageV1::CompleteWork {
            return Err(WorkMetricArchitectureErrorV1::ProjectionRequiresCompleteCoverage);
        }
        let first = &self.regions[&self.region_order[0]];
        let mut map = TemporalMapV1::new(first.tempo, first.meter.clone());
        let mut previous_meter = first.meter.clone();
        let mut previous_tempo = first.tempo;
        for region_id in self.region_order.iter().skip(1) {
            let region = &self.regions[region_id];
            let meter = (region.meter != previous_meter).then_some(region.meter.clone());
            let tempo = (region.tempo != previous_tempo).then_some(region.tempo);
            if meter.is_some() || tempo.is_some() {
                map.push_change(region.start, meter, tempo)
                    .map_err(WorkMetricArchitectureErrorV1::TemporalMap)?;
            }
            previous_meter = region.meter.clone();
            previous_tempo = region.tempo;
        }
        Ok(map)
    }

    pub fn region_at<'a>(
        &'a self,
        work_plan: &HierarchicalWorkPlanV1,
        at: Duration,
    ) -> Result<Option<(&'a str, &'a MetricRegionV1)>, WorkMetricArchitectureErrorV1> {
        self.validate(work_plan)?;
        if at.num() < 0 {
            return Err(WorkMetricArchitectureErrorV1::QueryBeforeWork);
        }
        for region_id in &self.region_order {
            let region = &self.regions[region_id];
            if compare_duration(region.start, at) != Ordering::Greater
                && compare_duration(at, region.end) == Ordering::Less
            {
                return Ok(Some((region_id.as_str(), region)));
            }
        }
        Ok(None)
    }
}

fn validate_transition(
    transition_index: usize,
    relation: &MetricTransitionRelationV1,
    from: &MetricRegionV1,
    to: &MetricRegionV1,
) -> Result<(), WorkMetricArchitectureErrorV1> {
    let state_changed = from.meter != to.meter || from.tempo != to.tempo;
    match relation {
        MetricTransitionRelationV1::Continuation if state_changed => Err(
            WorkMetricArchitectureErrorV1::ContinuationChangedMetricState { transition_index },
        ),
        MetricTransitionRelationV1::DirectChange if !state_changed => Err(
            WorkMetricArchitectureErrorV1::DirectChangeDidNotChangeMetricState {
                transition_index,
            },
        ),
        MetricTransitionRelationV1::PulseEquivalence {
            source_pulse,
            target_pulse,
        } => {
            if source_pulse.num() <= 0 || target_pulse.num() <= 0 {
                return Err(WorkMetricArchitectureErrorV1::NonPositivePulse {
                    transition_index,
                });
            }
            if !duration_seconds_equal_exact(
                *source_pulse,
                from.tempo,
                *target_pulse,
                to.tempo,
            ) {
                return Err(WorkMetricArchitectureErrorV1::PulseEquivalenceMismatch {
                    transition_index,
                });
            }
            Ok(())
        }
        MetricTransitionRelationV1::BarDurationEquivalence => {
            if !duration_seconds_equal_exact(
                from.meter.quarter_note_beats_per_bar(),
                from.tempo,
                to.meter.quarter_note_beats_per_bar(),
                to.tempo,
            ) {
                return Err(
                    WorkMetricArchitectureErrorV1::BarDurationEquivalenceMismatch {
                        transition_index,
                    },
                );
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Exact equality of elapsed time for two rational beat durations under two
/// rational quarter-note tempi. The common factor 60 seconds/minute cancels.
fn duration_seconds_equal_exact(
    left_duration: Duration,
    left_tempo: TempoV1,
    right_duration: Duration,
    right_tempo: TempoV1,
) -> bool {
    let left = i128::from(left_duration.num())
        * i128::from(left_tempo.denominator())
        * i128::from(right_duration.den())
        * i128::from(right_tempo.numerator_bpm());
    let right = i128::from(right_duration.num())
        * i128::from(right_tempo.denominator())
        * i128::from(left_duration.den())
        * i128::from(left_tempo.numerator_bpm());
    left == right
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work() -> HierarchicalWorkPlanV1 {
        HierarchicalWorkPlanV1::new("work", Duration::new(24, 1)).unwrap()
    }

    fn region(
        start: i64,
        end: i64,
        meter: TimeSignature,
        tempo: u32,
        role: MetricRegionRoleV1,
    ) -> MetricRegionV1 {
        MetricRegionV1 {
            work_node_id: "work".into(),
            start: Duration::new(start, 1),
            end: Duration::new(end, 1),
            meter,
            tempo: TempoV1::integer(tempo).unwrap(),
            role,
        }
    }

    fn architecture() -> WorkMetricArchitectureV1 {
        let four_four = TimeSignature::new(4, 4).unwrap();
        let six_eight = TimeSignature::new(6, 8).unwrap();
        let five_four = TimeSignature::new(5, 4).unwrap();
        WorkMetricArchitectureV1 {
            version: WORK_METRIC_ARCHITECTURE_VERSION.into(),
            architecture_id: "metric-arc".into(),
            coverage: MetricArchitectureCoverageV1::CompleteWork,
            regions: [
                (
                    "A".into(),
                    region(0, 8, four_four.clone(), 120, MetricRegionRoleV1::Stable),
                ),
                (
                    "B".into(),
                    region(8, 16, six_eight, 180, MetricRegionRoleV1::Compound),
                ),
                (
                    "C".into(),
                    region(16, 24, five_four, 150, MetricRegionRoleV1::Return),
                ),
            ]
            .into_iter()
            .collect(),
            region_order: vec!["A".into(), "B".into(), "C".into()],
            transitions: vec![
                MetricTransitionV1 {
                    from_region_id: "A".into(),
                    to_region_id: "B".into(),
                    relation: MetricTransitionRelationV1::PulseEquivalence {
                        source_pulse: Duration::new(1, 1),
                        target_pulse: Duration::new(3, 2),
                    },
                },
                MetricTransitionV1 {
                    from_region_id: "B".into(),
                    to_region_id: "C".into(),
                    relation: MetricTransitionRelationV1::DirectChange,
                },
            ],
        }
    }

    #[test]
    fn exact_metric_modulation_accepts_equal_elapsed_pulses() {
        // quarter @ 120 == dotted-quarter @ 180 == 0.5 seconds.
        architecture().validate(&work()).unwrap();
    }

    #[test]
    fn forged_pulse_equivalence_fails_exactly() {
        let mut architecture = architecture();
        architecture.transitions[0].relation = MetricTransitionRelationV1::PulseEquivalence {
            source_pulse: Duration::new(1, 1),
            target_pulse: Duration::new(1, 1),
        };
        assert_eq!(
            architecture.validate(&work()),
            Err(WorkMetricArchitectureErrorV1::PulseEquivalenceMismatch {
                transition_index: 0,
            })
        );
    }

    #[test]
    fn bar_duration_equivalence_is_exact() {
        let four_four = TimeSignature::new(4, 4).unwrap();
        let five_four = TimeSignature::new(5, 4).unwrap();
        let from = region(0, 8, four_four, 120, MetricRegionRoleV1::Stable);
        let to = region(8, 16, five_four, 150, MetricRegionRoleV1::Expansion);
        validate_transition(
            0,
            &MetricTransitionRelationV1::BarDurationEquivalence,
            &from,
            &to,
        )
        .unwrap();
    }

    #[test]
    fn complete_architecture_projects_to_temporal_map() {
        let architecture = architecture();
        let map = architecture.to_temporal_map(&work()).unwrap();
        assert_eq!(map.points.len(), 3);
        assert_eq!(map.points[0].at, Duration::zero());
        assert_eq!(map.points[1].at, Duration::new(8, 1));
        assert_eq!(map.points[2].at, Duration::new(16, 1));
        assert_eq!(map.meter_at(Duration::new(9, 1)).unwrap().numerator(), 6);
        assert_eq!(map.tempo_at(Duration::new(9, 1)).unwrap().bpm(), 180.0);
    }

    #[test]
    fn declared_only_architecture_cannot_mint_complete_temporal_authority() {
        let mut architecture = architecture();
        architecture.coverage = MetricArchitectureCoverageV1::DeclaredRegionsOnly;
        assert_eq!(
            architecture.to_temporal_map(&work()),
            Err(WorkMetricArchitectureErrorV1::ProjectionRequiresCompleteCoverage)
        );
    }

    #[test]
    fn complete_coverage_rejects_gaps() {
        let mut architecture = architecture();
        architecture.regions.get_mut("B").unwrap().start = Duration::new(9, 1);
        assert_eq!(
            architecture.validate(&work()),
            Err(WorkMetricArchitectureErrorV1::CompleteCoverageGap {
                left_region_id: "A".into(),
                right_region_id: "B".into(),
            })
        );
    }

    #[test]
    fn continuation_requires_identical_meter_grouping_and_tempo() {
        let mut architecture = architecture();
        architecture.transitions[0].relation = MetricTransitionRelationV1::Continuation;
        assert_eq!(
            architecture.validate(&work()),
            Err(WorkMetricArchitectureErrorV1::ContinuationChangedMetricState {
                transition_index: 0,
            })
        );
    }
}
