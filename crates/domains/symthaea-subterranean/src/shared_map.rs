// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministically merged, provenance-preserving tunnel observations.
//!
//! Peers do not overwrite one another's route knowledge. The map keeps the
//! newest observation per `(depth bin, source)` and computes a conservative
//! aggregate: minimum roof/localization confidence and maximum water/slurry.
//! Equal-version conflicting payloads are rejected as equivocation rather
//! than resolved by arrival order.

use crate::path_memory::{RETURN_RESERVE_RATIO, ReturnPathAssessment};
use crate::team::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_SHARED_MAP_CAPACITY: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SharedTunnelObservation {
    pub source: AgentId,
    pub epoch: u32,
    pub sequence: u64,
    pub observed_step: u64,
    pub bin_index: u16,
    pub minimum_depth_m: f64,
    pub maximum_depth_m: f64,
    pub roof_stability: f64,
    pub water_ingress: f64,
    pub slurry_load: f64,
    pub localization_confidence: f64,
    pub survey_confidence: f64,
    pub roof_supported: bool,
}

impl SharedTunnelObservation {
    pub fn is_valid(self) -> bool {
        self.source != AgentId::SURFACE_CONTROL
            && self.minimum_depth_m.is_finite()
            && self.maximum_depth_m.is_finite()
            && self.minimum_depth_m >= 0.0
            && self.maximum_depth_m >= self.minimum_depth_m
            && self.maximum_depth_m <= 200.0
            && [
                self.roof_stability,
                self.water_ingress,
                self.slurry_load,
                self.localization_confidence,
                self.survey_confidence,
            ]
            .into_iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }

    fn version_is_newer_than(self, other: Self) -> bool {
        self.epoch > other.epoch || (self.epoch == other.epoch && self.sequence > other.sequence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedMapRejection {
    Invalid,
    Replay,
    Equivocation,
    Capacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SharedTunnelBin {
    pub bin_index: u16,
    pub source_count: usize,
    pub minimum_depth_m: f64,
    pub maximum_depth_m: f64,
    pub minimum_roof_stability: f64,
    pub maximum_water_ingress: f64,
    pub maximum_slurry_load: f64,
    pub minimum_localization_confidence: f64,
    pub minimum_survey_confidence: f64,
    pub any_roof_support: bool,
    pub latest_observed_step: u64,
}

impl SharedTunnelBin {
    pub fn obstruction_risk(self) -> f64 {
        let support_credit = if self.any_roof_support { 0.12 } else { 0.0 };
        ((1.0 - self.minimum_roof_stability) * 0.42
            + self.maximum_water_ingress * 0.24
            + self.maximum_slurry_load * 0.18
            + (1.0 - self.minimum_localization_confidence) * 0.1
            + (1.0 - self.minimum_survey_confidence) * 0.06
            - support_credit)
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SharedRouteKnowledge {
    pub known_bins: usize,
    pub contributing_peers: usize,
    pub deepest_known_m: f64,
    pub minimum_confidence: f64,
    pub maximum_obstruction_risk: f64,
}

impl SharedRouteKnowledge {
    pub const fn empty() -> Self {
        Self {
            known_bins: 0,
            contributing_peers: 0,
            deepest_known_m: 0.0,
            minimum_confidence: 1.0,
            maximum_obstruction_risk: 0.0,
        }
    }

    /// Fuse peer route evidence without allowing it to make the locally
    /// measured return claim more optimistic. Shared evidence may lower
    /// confidence, raise obstruction/energy cost, or make a route infeasible;
    /// it cannot increase margin or clear a local failure.
    pub fn conservative_return_fusion(self, local: ReturnPathAssessment) -> ReturnPathAssessment {
        if self.known_bins == 0 {
            return local;
        }
        let path_confidence = local.path_confidence.min(self.minimum_confidence);
        let obstruction_risk = local.obstruction_risk.max(self.maximum_obstruction_risk);
        let uncertainty_cost = (1.0 - path_confidence) * 0.015;
        let obstruction_multiplier = 1.0 + obstruction_risk * 0.75;
        let estimated_battery_required = local
            .estimated_battery_required
            .max(local.estimated_battery_required * obstruction_multiplier + uncertainty_cost)
            .clamp(0.0, 1.0);
        let added_cost = estimated_battery_required - local.estimated_battery_required;
        let battery_margin = local.battery_margin - added_cost;
        let feasible = local.feasible
            && battery_margin >= RETURN_RESERVE_RATIO
            && path_confidence >= 0.35
            && obstruction_risk < 0.92;
        ReturnPathAssessment {
            distance_home_m: local.distance_home_m,
            recorded_distance_m: local.recorded_distance_m,
            coverage_ratio: local.coverage_ratio,
            path_confidence,
            obstruction_risk,
            estimated_battery_required,
            battery_margin,
            feasible,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedTunnelMap {
    capacity: usize,
    observations: BTreeMap<(u16, AgentId), SharedTunnelObservation>,
}

impl SharedTunnelMap {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            observations: BTreeMap::new(),
        }
    }

    pub fn merge(
        &mut self,
        observation: SharedTunnelObservation,
    ) -> Result<(), SharedMapRejection> {
        if !observation.is_valid() {
            return Err(SharedMapRejection::Invalid);
        }
        let key = (observation.bin_index, observation.source);
        if let Some(existing) = self.observations.get(&key).copied() {
            if observation == existing {
                return Err(SharedMapRejection::Replay);
            }
            if observation.epoch == existing.epoch && observation.sequence == existing.sequence {
                return Err(SharedMapRejection::Equivocation);
            }
            if !observation.version_is_newer_than(existing) {
                return Err(SharedMapRejection::Replay);
            }
        } else if self.observations.len() >= self.capacity {
            return Err(SharedMapRejection::Capacity);
        }
        self.observations.insert(key, observation);
        Ok(())
    }

    pub fn aggregate_bin(&self, bin_index: u16) -> Option<SharedTunnelBin> {
        let mut observations = self
            .observations
            .range((bin_index, AgentId::new(0))..=(bin_index, AgentId::new(u64::MAX)))
            .map(|(_, observation)| *observation);
        let first = observations.next()?;
        let mut aggregate = SharedTunnelBin {
            bin_index,
            source_count: 1,
            minimum_depth_m: first.minimum_depth_m,
            maximum_depth_m: first.maximum_depth_m,
            minimum_roof_stability: first.roof_stability,
            maximum_water_ingress: first.water_ingress,
            maximum_slurry_load: first.slurry_load,
            minimum_localization_confidence: first.localization_confidence,
            minimum_survey_confidence: first.survey_confidence,
            any_roof_support: first.roof_supported,
            latest_observed_step: first.observed_step,
        };
        for observation in observations {
            aggregate.source_count += 1;
            aggregate.minimum_depth_m = aggregate.minimum_depth_m.min(observation.minimum_depth_m);
            aggregate.maximum_depth_m = aggregate.maximum_depth_m.max(observation.maximum_depth_m);
            aggregate.minimum_roof_stability = aggregate
                .minimum_roof_stability
                .min(observation.roof_stability);
            aggregate.maximum_water_ingress = aggregate
                .maximum_water_ingress
                .max(observation.water_ingress);
            aggregate.maximum_slurry_load =
                aggregate.maximum_slurry_load.max(observation.slurry_load);
            aggregate.minimum_localization_confidence = aggregate
                .minimum_localization_confidence
                .min(observation.localization_confidence);
            aggregate.minimum_survey_confidence = aggregate
                .minimum_survey_confidence
                .min(observation.survey_confidence);
            aggregate.any_roof_support |= observation.roof_supported;
            aggregate.latest_observed_step = aggregate
                .latest_observed_step
                .max(observation.observed_step);
        }
        Some(aggregate)
    }

    pub fn route_knowledge(&self, maximum_depth_m: f64) -> SharedRouteKnowledge {
        if self.observations.is_empty() || !maximum_depth_m.is_finite() || maximum_depth_m <= 0.0 {
            return SharedRouteKnowledge::empty();
        }
        let mut bins = BTreeMap::new();
        let mut peers = BTreeMap::new();
        for ((bin_index, source), observation) in &self.observations {
            if observation.minimum_depth_m <= maximum_depth_m {
                bins.insert(*bin_index, ());
                peers.insert(*source, ());
            }
        }
        let mut result = SharedRouteKnowledge {
            known_bins: bins.len(),
            contributing_peers: peers.len(),
            deepest_known_m: 0.0,
            minimum_confidence: 1.0,
            maximum_obstruction_risk: 0.0,
        };
        for bin_index in bins.keys().copied() {
            if let Some(bin) = self.aggregate_bin(bin_index) {
                result.deepest_known_m = result.deepest_known_m.max(bin.maximum_depth_m);
                result.minimum_confidence = result
                    .minimum_confidence
                    .min(bin.minimum_survey_confidence)
                    .min(bin.minimum_localization_confidence);
                result.maximum_obstruction_risk =
                    result.maximum_obstruction_risk.max(bin.obstruction_risk());
            }
        }
        result
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn clear(&mut self) {
        self.observations.clear();
    }
}

impl Default for SharedTunnelMap {
    fn default() -> Self {
        Self::new(DEFAULT_SHARED_MAP_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(source: u64, bin: u16, sequence: u64) -> SharedTunnelObservation {
        SharedTunnelObservation {
            source: AgentId::new(source),
            epoch: 1,
            sequence,
            observed_step: sequence,
            bin_index: bin,
            minimum_depth_m: bin as f64,
            maximum_depth_m: bin as f64 + 1.0,
            roof_stability: 0.9,
            water_ingress: 0.1,
            slurry_load: 0.1,
            localization_confidence: 0.9,
            survey_confidence: 0.8,
            roof_supported: false,
        }
    }

    #[test]
    fn merge_is_order_independent_across_sources() {
        let mut a = SharedTunnelMap::default();
        let mut b = SharedTunnelMap::default();
        let mut wet = observation(2, 4, 1);
        wet.water_ingress = 0.7;
        let mut weak = observation(3, 4, 1);
        weak.roof_stability = 0.3;
        assert_eq!(a.merge(wet), Ok(()));
        assert_eq!(a.merge(weak), Ok(()));
        assert_eq!(b.merge(weak), Ok(()));
        assert_eq!(b.merge(wet), Ok(()));
        assert_eq!(a.aggregate_bin(4), b.aggregate_bin(4));
    }

    #[test]
    fn equal_version_conflict_is_rejected_as_equivocation() {
        let mut map = SharedTunnelMap::default();
        let first = observation(2, 4, 1);
        let mut conflict = first;
        conflict.water_ingress = 0.9;
        assert_eq!(map.merge(first), Ok(()));
        assert_eq!(map.merge(conflict), Err(SharedMapRejection::Equivocation));
    }

    #[test]
    fn aggregation_is_conservative() {
        let mut map = SharedTunnelMap::default();
        let first = observation(2, 8, 1);
        let mut second = observation(3, 8, 1);
        second.roof_stability = 0.2;
        second.water_ingress = 0.8;
        assert_eq!(map.merge(first), Ok(()));
        assert_eq!(map.merge(second), Ok(()));
        let aggregate = map.aggregate_bin(8);
        assert!(aggregate.is_some());
        let aggregate = aggregate.unwrap_or(SharedTunnelBin {
            bin_index: 0,
            source_count: 0,
            minimum_depth_m: 0.0,
            maximum_depth_m: 0.0,
            minimum_roof_stability: 1.0,
            maximum_water_ingress: 0.0,
            maximum_slurry_load: 0.0,
            minimum_localization_confidence: 1.0,
            minimum_survey_confidence: 1.0,
            any_roof_support: false,
            latest_observed_step: 0,
        });
        assert_eq!(aggregate.minimum_roof_stability, 0.2);
        assert_eq!(aggregate.maximum_water_ingress, 0.8);
        assert!(aggregate.obstruction_risk() > 0.5);
    }
    #[test]
    fn peer_route_evidence_can_only_make_return_claim_more_conservative() {
        let local = ReturnPathAssessment {
            distance_home_m: 30.0,
            recorded_distance_m: 30.0,
            coverage_ratio: 1.0,
            path_confidence: 0.9,
            obstruction_risk: 0.1,
            estimated_battery_required: 0.12,
            battery_margin: 0.3,
            feasible: true,
        };
        let peer = SharedRouteKnowledge {
            known_bins: 20,
            contributing_peers: 2,
            deepest_known_m: 30.0,
            minimum_confidence: 0.4,
            maximum_obstruction_risk: 0.8,
        };
        let fused = peer.conservative_return_fusion(local);
        assert!(fused.path_confidence <= local.path_confidence);
        assert!(fused.obstruction_risk >= local.obstruction_risk);
        assert!(fused.estimated_battery_required >= local.estimated_battery_required);
        assert!(fused.battery_margin <= local.battery_margin);

        let empty = SharedRouteKnowledge::empty().conservative_return_fusion(local);
        assert_eq!(empty, local);
    }
}
