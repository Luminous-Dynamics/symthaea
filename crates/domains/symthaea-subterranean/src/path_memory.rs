// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded return-path memory and retreat-feasibility estimates.
//!
//! Underground autonomy cannot treat return as an abstract direction. The
//! route behind the platform accumulates water, slurry, roof damage,
//! localization uncertainty, and support history. This module records a
//! bounded, depth-binned operational history and estimates whether the current
//! battery reserve can traverse that route back to the surface.

use crate::types::{
    LOCALIZATION_CONFIDENCE, ROOF_STABILITY, SLURRY_LOAD, SubterraneanCommand, SubterraneanState,
    WATER_INGRESS_RATIO,
};
use serde::{Deserialize, Serialize};

pub const DEFAULT_PATH_CAPACITY: usize = 256;
pub const DEFAULT_SEGMENT_SPAN_M: f64 = 1.0;
pub const RETURN_RESERVE_RATIO: f64 = 0.04;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReturnPathSegment {
    pub bin_index: u16,
    pub minimum_depth_m: f64,
    pub maximum_depth_m: f64,
    pub minimum_roof_stability: f64,
    pub maximum_water_ingress: f64,
    pub maximum_slurry_load: f64,
    pub minimum_localization_confidence: f64,
    pub roof_supported: bool,
    pub traversal_count: u32,
}

impl ReturnPathSegment {
    fn new(bin_index: u16, before: &SubterraneanState, after: &SubterraneanState) -> Self {
        Self {
            bin_index,
            minimum_depth_m: before.depth_m().min(after.depth_m()),
            maximum_depth_m: before.depth_m().max(after.depth_m()),
            minimum_roof_stability: before.channels[ROOF_STABILITY]
                .min(after.channels[ROOF_STABILITY]),
            maximum_water_ingress: before.channels[WATER_INGRESS_RATIO]
                .max(after.channels[WATER_INGRESS_RATIO]),
            maximum_slurry_load: before.channels[SLURRY_LOAD].max(after.channels[SLURRY_LOAD]),
            minimum_localization_confidence: before.channels[LOCALIZATION_CONFIDENCE]
                .min(after.channels[LOCALIZATION_CONFIDENCE]),
            roof_supported: false,
            traversal_count: 1,
        }
    }

    fn observe(
        &mut self,
        before: &SubterraneanState,
        after: &SubterraneanState,
        command: &SubterraneanCommand,
    ) {
        self.minimum_depth_m = self
            .minimum_depth_m
            .min(before.depth_m().min(after.depth_m()));
        self.maximum_depth_m = self
            .maximum_depth_m
            .max(before.depth_m().max(after.depth_m()));
        self.minimum_roof_stability = self
            .minimum_roof_stability
            .min(before.channels[ROOF_STABILITY])
            .min(after.channels[ROOF_STABILITY]);
        self.maximum_water_ingress = self
            .maximum_water_ingress
            .max(before.channels[WATER_INGRESS_RATIO])
            .max(after.channels[WATER_INGRESS_RATIO]);
        self.maximum_slurry_load = self
            .maximum_slurry_load
            .max(before.channels[SLURRY_LOAD])
            .max(after.channels[SLURRY_LOAD]);
        self.minimum_localization_confidence = self
            .minimum_localization_confidence
            .min(before.channels[LOCALIZATION_CONFIDENCE])
            .min(after.channels[LOCALIZATION_CONFIDENCE]);
        self.roof_supported |= command.recovery.roof_support >= 0.5;
        self.traversal_count = self.traversal_count.saturating_add(1);
    }

    pub fn obstruction_risk(self) -> f64 {
        let support_credit = if self.roof_supported { 0.14 } else { 0.0 };
        ((1.0 - self.minimum_roof_stability) * 0.42
            + self.maximum_water_ingress * 0.24
            + self.maximum_slurry_load * 0.2
            + (1.0 - self.minimum_localization_confidence) * 0.14
            - support_credit)
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReturnPathAssessment {
    pub distance_home_m: f64,
    pub recorded_distance_m: f64,
    pub coverage_ratio: f64,
    pub path_confidence: f64,
    pub obstruction_risk: f64,
    pub estimated_battery_required: f64,
    pub battery_margin: f64,
    pub feasible: bool,
}

impl ReturnPathAssessment {
    pub const fn surface() -> Self {
        Self {
            distance_home_m: 0.0,
            recorded_distance_m: 0.0,
            coverage_ratio: 1.0,
            path_confidence: 1.0,
            obstruction_risk: 0.0,
            estimated_battery_required: 0.0,
            battery_margin: 1.0,
            feasible: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnPathMemory {
    segments: Vec<ReturnPathSegment>,
    capacity: usize,
    segment_span_m: f64,
    cumulative_outbound_m: f64,
    cumulative_retreat_m: f64,
}

impl ReturnPathMemory {
    pub fn new(capacity: usize, segment_span_m: f64) -> Self {
        Self {
            segments: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            segment_span_m: if segment_span_m.is_finite() && segment_span_m > 0.0 {
                segment_span_m
            } else {
                DEFAULT_SEGMENT_SPAN_M
            },
            cumulative_outbound_m: 0.0,
            cumulative_retreat_m: 0.0,
        }
    }

    pub fn observe(
        &mut self,
        before: &SubterraneanState,
        after: &SubterraneanState,
        command: &SubterraneanCommand,
    ) {
        let displacement = after.depth_m() - before.depth_m();
        if !displacement.is_finite() || displacement.abs() < 1e-6 {
            return;
        }
        if displacement > 0.0 {
            self.cumulative_outbound_m += displacement;
        } else {
            self.cumulative_retreat_m += -displacement;
        }

        let midpoint = (before.depth_m() + after.depth_m()) * 0.5;
        let raw_bin = (midpoint.max(0.0) / self.segment_span_m).floor();
        let bin_index = raw_bin.min(u16::MAX as f64) as u16;
        if let Some(segment) = self
            .segments
            .iter_mut()
            .find(|segment| segment.bin_index == bin_index)
        {
            segment.observe(before, after, command);
            return;
        }

        if self.segments.len() == self.capacity {
            self.segments.remove(0);
        }
        let mut segment = ReturnPathSegment::new(bin_index, before, after);
        segment.roof_supported = command.recovery.roof_support >= 0.5;
        self.segments.push(segment);
        self.segments.sort_by_key(|segment| segment.bin_index);
    }

    pub fn assess(&self, state: &SubterraneanState) -> ReturnPathAssessment {
        let distance_home_m = state.depth_m().max(0.0);
        if distance_home_m <= 1e-6 {
            return ReturnPathAssessment::surface();
        }

        let relevant: Vec<_> = self
            .segments
            .iter()
            .copied()
            .filter(|segment| segment.minimum_depth_m <= distance_home_m)
            .collect();
        let expected_bins = (distance_home_m / self.segment_span_m).ceil().max(1.0);
        let coverage_ratio = (relevant.len() as f64 / expected_bins).clamp(0.0, 1.0);
        let recorded_distance_m = relevant
            .iter()
            .map(|segment| segment.maximum_depth_m - segment.minimum_depth_m)
            .sum::<f64>()
            .min(distance_home_m);
        let obstruction_risk = relevant
            .iter()
            .copied()
            .map(ReturnPathSegment::obstruction_risk)
            .fold(0.0, f64::max);
        let mean_route_quality = if relevant.is_empty() {
            0.0
        } else {
            relevant
                .iter()
                .map(|segment| {
                    (segment.minimum_roof_stability * 0.45
                        + (1.0 - segment.maximum_water_ingress) * 0.2
                        + (1.0 - segment.maximum_slurry_load) * 0.15
                        + segment.minimum_localization_confidence * 0.2)
                        .clamp(0.0, 1.0)
                })
                .sum::<f64>()
                / relevant.len() as f64
        };
        let path_confidence = (mean_route_quality * 0.72 + coverage_ratio * 0.28).clamp(0.0, 1.0);
        let energy_per_meter = 0.0022 * (1.0 + obstruction_risk * 1.8);
        let estimated_battery_required =
            (distance_home_m * energy_per_meter + 0.025).clamp(0.0, 1.0);
        let battery_margin = state.battery_ratio() - estimated_battery_required;
        let feasible = battery_margin >= RETURN_RESERVE_RATIO
            && path_confidence >= 0.35
            && obstruction_risk < 0.92;
        ReturnPathAssessment {
            distance_home_m,
            recorded_distance_m,
            coverage_ratio,
            path_confidence,
            obstruction_risk,
            estimated_battery_required,
            battery_margin,
            feasible,
        }
    }

    pub fn segments(&self) -> &[ReturnPathSegment] {
        &self.segments
    }

    pub fn cumulative_outbound_m(&self) -> f64 {
        self.cumulative_outbound_m
    }

    pub fn cumulative_retreat_m(&self) -> f64 {
        self.cumulative_retreat_m
    }

    pub fn reset(&mut self) {
        self.segments.clear();
        self.cumulative_outbound_m = 0.0;
        self.cumulative_retreat_m = 0.0;
    }
}

impl Default for ReturnPathMemory {
    fn default() -> Self {
        Self::new(DEFAULT_PATH_CAPACITY, DEFAULT_SEGMENT_SPAN_M)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BATTERY_RATIO, ROOF_STABILITY, WATER_INGRESS_RATIO};

    fn transition(before_depth: f64, after_depth: f64) -> (SubterraneanState, SubterraneanState) {
        let mut before = SubterraneanState::home();
        let mut after = SubterraneanState::home();
        before.channels[crate::types::DEPTH_M] = before_depth;
        after.channels[crate::types::DEPTH_M] = after_depth;
        (before, after)
    }

    #[test]
    fn outbound_motion_builds_a_bounded_route_record() {
        let mut memory = ReturnPathMemory::new(4, 1.0);
        for index in 0..8 {
            let (before, after) = transition(index as f64, index as f64 + 0.8);
            memory.observe(&before, &after, &SubterraneanCommand::zero());
        }
        assert_eq!(memory.segments().len(), 4);
        assert!(memory.cumulative_outbound_m() > 6.0);
    }

    #[test]
    fn damaged_wet_route_reduces_return_confidence() {
        let mut healthy = ReturnPathMemory::default();
        let mut damaged = ReturnPathMemory::default();
        for index in 0..20 {
            let (before, after) = transition(index as f64, index as f64 + 1.0);
            healthy.observe(&before, &after, &SubterraneanCommand::zero());
            let mut damaged_after = after;
            damaged_after.channels[ROOF_STABILITY] = 0.18;
            damaged_after.channels[WATER_INGRESS_RATIO] = 0.8;
            damaged.observe(&before, &damaged_after, &SubterraneanCommand::zero());
        }
        let mut current = SubterraneanState::home();
        current.channels[crate::types::DEPTH_M] = 20.0;
        assert!(
            damaged.assess(&current).path_confidence < healthy.assess(&current).path_confidence
        );
        assert!(damaged.assess(&current).obstruction_risk > 0.5);
    }

    #[test]
    fn battery_budget_can_make_an_otherwise_known_route_infeasible() {
        let mut memory = ReturnPathMemory::default();
        for index in 0..60 {
            let (before, after) = transition(index as f64, index as f64 + 1.0);
            memory.observe(&before, &after, &SubterraneanCommand::zero());
        }
        let mut current = SubterraneanState::home();
        current.channels[crate::types::DEPTH_M] = 60.0;
        current.channels[BATTERY_RATIO] = 0.12;
        let assessment = memory.assess(&current);
        assert!(!assessment.feasible);
        assert!(assessment.battery_margin < RETURN_RESERVE_RATIO);
    }
}
