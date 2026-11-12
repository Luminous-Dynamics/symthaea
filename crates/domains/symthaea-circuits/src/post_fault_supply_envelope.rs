//! Exact piecewise-constant post-fault electrical supply envelope.
//!
//! This module composes already-qualified post-isolation path evidence,
//! critical-load demand upper bounds, non-storage source-capability lower
//! bounds, storage discharge-power lower bounds, and usable-energy lower
//! bounds. It does not solve protection, network transients, electrochemistry,
//! thermal dynamics, thrust, or flight-control authority.

#![forbid(unsafe_code)]

/// Whether a critical load has an evidenced electrical path after isolation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostFaultPathEvidence {
    ConfirmedAvailable { path_digest: String },
    ConfirmedUnavailable { evidence_digest: String },
    Unknown,
}

/// Upper bound for one critical load during one constant-demand segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalLoadDemand {
    pub load_id: String,
    pub demand_upper_w: u64,
    pub path: PostFaultPathEvidence,
}

/// Piecewise-constant post-fault operating interval.
///
/// All times are relative to the post-fault profile start. Segments must be
/// contiguous, non-overlapping and start at t=0. Power values are hard bounds,
/// not estimates or probability distributions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostFaultSupplySegment {
    pub segment_id: String,
    pub start_seconds: u64,
    pub duration_seconds: u64,
    /// Sum of non-storage sources that remain available through confirmed paths.
    pub non_storage_source_capacity_lower_w: u64,
    /// Guaranteed aggregate storage discharge power for this segment.
    pub storage_discharge_power_lower_w: u64,
    pub critical_loads: Vec<CriticalLoadDemand>,
}

/// A digest-bound, post-isolation supply profile.
///
/// The profile-level storage energy lower bound is a single cumulative budget
/// shared across all segments. No charging, regeneration or replenishment
/// credit is assumed in V0.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostFaultSupplyProfile {
    pub schema_version: String,
    pub configuration_digest: String,
    pub operating_point_digest: String,
    pub post_fault_topology_digest: String,
    pub fault_event_digest: String,
    pub critical_load_manifest_digest: String,
    pub source_capability_digest: String,
    pub storage_state_digest: String,
    pub critical_load_ids: Vec<String>,
    pub usable_storage_energy_lower_j: u128,
    pub segments: Vec<PostFaultSupplySegment>,
}

/// Invalid evidence/profile structure. These failures are never converted to
/// a sustained-supply disposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupplyProfileError {
    EmptyIdentity(&'static str),
    NoCriticalLoads,
    DuplicateCriticalLoadId(String),
    NoSegments,
    EmptySegmentId,
    DuplicateSegmentId(String),
    ZeroDuration { segment_id: String },
    NonContiguousTimeline { segment_id: String, expected_start_s: u64 },
    EmptyLoadSet { segment_id: String },
    DuplicateLoadInSegment { segment_id: String, load_id: String },
    LoadManifestMismatch { segment_id: String, load_id: String },
    InvalidPathEvidence { segment_id: String, load_id: String },
    EnergyAccountingOverflow,
}

/// Result for a valid input profile.
///
/// SustainedElectricalSupply certifies only the supplied piecewise-constant
/// electrical power/energy envelope. This module makes no mission/flight
/// viability determination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElectricalSupplyDisposition {
    SustainedElectricalSupply {
        total_duration_seconds: u64,
        maximum_storage_power_required_w: u64,
        required_storage_energy_j: u128,
        remaining_storage_energy_lower_j: u128,
    },
    NotSustainedPath {
        segment_id: String,
        load_id: String,
        evidence_digest: String,
    },
    NotSustainedPowerLimit {
        segment_id: String,
        required_storage_power_w: u64,
        available_storage_power_lower_w: u64,
        unmet_power_w: u64,
    },
    NotSustainedEnergyLimit {
        segment_id: String,
        required_storage_energy_j: u128,
        usable_storage_energy_lower_j: u128,
        deficit_energy_j: u128,
        reserve_depletion_time_seconds: u64,
    },
    NotEvaluablePath {
        segment_id: String,
        load_id: String,
    },
}

/// Evaluate a post-fault supply profile with exact integer power and energy
/// arithmetic. Load demand is summed conservatively; any unknown path returns
/// NOT EVALUABLE, while a confirmed absent path is a non-sustained outcome.
pub fn evaluate_post_fault_supply(
    profile: &PostFaultSupplyProfile,
) -> Result<ElectricalSupplyDisposition, SupplyProfileError> {
    validate_profile(profile)?;

    let mut expected_start_s = 0_u64;
    let mut cumulative_storage_energy_j = 0_u128;
    let mut maximum_storage_power_required_w = 0_u64;

    for segment in &profile.segments {
        if segment.start_seconds != expected_start_s {
            return Err(SupplyProfileError::NonContiguousTimeline {
                segment_id: segment.segment_id.clone(),
                expected_start_s,
            });
        }

        let mut critical_load_power_upper_w = 0_u64;
        for load in &segment.critical_loads {
            match &load.path {
                PostFaultPathEvidence::ConfirmedAvailable { path_digest } => {
                    if path_digest.trim().is_empty() {
                        return Err(SupplyProfileError::InvalidPathEvidence {
                            segment_id: segment.segment_id.clone(),
                            load_id: load.load_id.clone(),
                        });
                    }
                }
                PostFaultPathEvidence::ConfirmedUnavailable { evidence_digest } => {
                    if evidence_digest.trim().is_empty() {
                        return Err(SupplyProfileError::InvalidPathEvidence {
                            segment_id: segment.segment_id.clone(),
                            load_id: load.load_id.clone(),
                        });
                    }
                    return Ok(ElectricalSupplyDisposition::NotSustainedPath {
                        segment_id: segment.segment_id.clone(),
                        load_id: load.load_id.clone(),
                        evidence_digest: evidence_digest.clone(),
                    });
                }
                PostFaultPathEvidence::Unknown => {
                    return Ok(ElectricalSupplyDisposition::NotEvaluablePath {
                        segment_id: segment.segment_id.clone(),
                        load_id: load.load_id.clone(),
                    });
                }
            }

            critical_load_power_upper_w = critical_load_power_upper_w
                .checked_add(load.demand_upper_w)
                .ok_or(SupplyProfileError::EnergyAccountingOverflow)?;
        }

        let storage_power_required_w = critical_load_power_upper_w
            .saturating_sub(segment.non_storage_source_capacity_lower_w);
        maximum_storage_power_required_w =
            maximum_storage_power_required_w.max(storage_power_required_w);

        if storage_power_required_w > segment.storage_discharge_power_lower_w {
            return Ok(ElectricalSupplyDisposition::NotSustainedPowerLimit {
                segment_id: segment.segment_id.clone(),
                required_storage_power_w: storage_power_required_w,
                available_storage_power_lower_w: segment.storage_discharge_power_lower_w,
                unmet_power_w: storage_power_required_w
                    - segment.storage_discharge_power_lower_w,
            });
        }

        let segment_storage_energy_j = u128::from(storage_power_required_w)
            .checked_mul(u128::from(segment.duration_seconds))
            .ok_or(SupplyProfileError::EnergyAccountingOverflow)?;
        let next_energy = cumulative_storage_energy_j
            .checked_add(segment_storage_energy_j)
            .ok_or(SupplyProfileError::EnergyAccountingOverflow)?;

        if next_energy > profile.usable_storage_energy_lower_j {
            let excess_energy_j = next_energy - profile.usable_storage_energy_lower_j;
            let remaining_before_segment =
                profile.usable_storage_energy_lower_j.saturating_sub(cumulative_storage_energy_j);
            let seconds_until_reserve_depletion = if storage_power_required_w == 0 {
                0
            } else {
                let power = u128::from(storage_power_required_w);
                let whole = remaining_before_segment / power;
                let remainder = remaining_before_segment % power;
                let seconds = whole + u128::from(remainder != 0);
                u64::try_from(seconds).unwrap_or(u64::MAX)
            };
            let reserve_depletion_time_seconds = segment
                .start_seconds
                .checked_add(seconds_until_reserve_depletion)
                .unwrap_or(u64::MAX);
            return Ok(ElectricalSupplyDisposition::NotSustainedEnergyLimit {
                segment_id: segment.segment_id.clone(),
                required_storage_energy_j: next_energy,
                usable_storage_energy_lower_j: profile.usable_storage_energy_lower_j,
                deficit_energy_j: excess_energy_j,
                reserve_depletion_time_seconds,
            });
        }

        cumulative_storage_energy_j = next_energy;
        expected_start_s = expected_start_s
            .checked_add(segment.duration_seconds)
            .ok_or(SupplyProfileError::EnergyAccountingOverflow)?;
    }

    Ok(ElectricalSupplyDisposition::SustainedElectricalSupply {
        total_duration_seconds: expected_start_s,
        maximum_storage_power_required_w,
        required_storage_energy_j: cumulative_storage_energy_j,
        remaining_storage_energy_lower_j: profile
            .usable_storage_energy_lower_j
            .saturating_sub(cumulative_storage_energy_j),
    })
}

fn validate_profile(profile: &PostFaultSupplyProfile) -> Result<(), SupplyProfileError> {
    for (field, value) in [
        ("schema_version", profile.schema_version.as_str()),
        ("configuration_digest", profile.configuration_digest.as_str()),
        ("operating_point_digest", profile.operating_point_digest.as_str()),
        ("post_fault_topology_digest", profile.post_fault_topology_digest.as_str()),
        ("fault_event_digest", profile.fault_event_digest.as_str()),
        ("critical_load_manifest_digest", profile.critical_load_manifest_digest.as_str()),
        ("source_capability_digest", profile.source_capability_digest.as_str()),
        ("storage_state_digest", profile.storage_state_digest.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(SupplyProfileError::EmptyIdentity(field));
        }
    }

    if profile.critical_load_ids.is_empty() {
        return Err(SupplyProfileError::NoCriticalLoads);
    }
    for (index, load_id) in profile.critical_load_ids.iter().enumerate() {
        if load_id.trim().is_empty() {
            return Err(SupplyProfileError::EmptyIdentity("critical_load_id"));
        }
        if profile.critical_load_ids[..index].contains(load_id) {
            return Err(SupplyProfileError::DuplicateCriticalLoadId(load_id.clone()));
        }
    }
    if profile.segments.is_empty() {
        return Err(SupplyProfileError::NoSegments);
    }

    let mut expected_start_s = 0_u64;
    let mut segment_ids = Vec::with_capacity(profile.segments.len());
    for segment in &profile.segments {
        if segment.segment_id.trim().is_empty() {
            return Err(SupplyProfileError::EmptySegmentId);
        }
        if segment_ids.contains(&segment.segment_id) {
            return Err(SupplyProfileError::DuplicateSegmentId(
                segment.segment_id.clone(),
            ));
        }
        segment_ids.push(segment.segment_id.clone());

        if segment.duration_seconds == 0 {
            return Err(SupplyProfileError::ZeroDuration {
                segment_id: segment.segment_id.clone(),
            });
        }
        if segment.start_seconds != expected_start_s {
            return Err(SupplyProfileError::NonContiguousTimeline {
                segment_id: segment.segment_id.clone(),
                expected_start_s,
            });
        }
        if segment.critical_loads.is_empty() {
            return Err(SupplyProfileError::EmptyLoadSet {
                segment_id: segment.segment_id.clone(),
            });
        }

        let mut observed_load_ids = Vec::with_capacity(segment.critical_loads.len());
        for load in &segment.critical_loads {
            if observed_load_ids.contains(&load.load_id) {
                return Err(SupplyProfileError::DuplicateLoadInSegment {
                    segment_id: segment.segment_id.clone(),
                    load_id: load.load_id.clone(),
                });
            }
            observed_load_ids.push(load.load_id.clone());
            if !profile.critical_load_ids.contains(&load.load_id) {
                return Err(SupplyProfileError::LoadManifestMismatch {
                    segment_id: segment.segment_id.clone(),
                    load_id: load.load_id.clone(),
                });
            }
        }
        for expected_load_id in &profile.critical_load_ids {
            if !observed_load_ids.contains(expected_load_id) {
                return Err(SupplyProfileError::LoadManifestMismatch {
                    segment_id: segment.segment_id.clone(),
                    load_id: expected_load_id.clone(),
                });
            }
        }

        expected_start_s = expected_start_s
            .checked_add(segment.duration_seconds)
            .ok_or(SupplyProfileError::EnergyAccountingOverflow)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available_load(load_id: &str, demand_upper_w: u64) -> CriticalLoadDemand {
        CriticalLoadDemand {
            load_id: load_id.to_string(),
            demand_upper_w,
            path: PostFaultPathEvidence::ConfirmedAvailable {
                path_digest: format!("sha256:path-{load_id}"),
            },
        }
    }

    fn profile(
        usable_energy_j: u128,
        segments: Vec<PostFaultSupplySegment>,
    ) -> PostFaultSupplyProfile {
        PostFaultSupplyProfile {
            schema_version: "0.1.0".to_string(),
            configuration_digest: "sha256:config-9".to_string(),
            operating_point_digest: "sha256:op-23".to_string(),
            post_fault_topology_digest: "sha256:topology-post-fault-4".to_string(),
            fault_event_digest: "sha256:fault-event-3".to_string(),
            critical_load_manifest_digest: "sha256:critical-loads-2".to_string(),
            source_capability_digest: "sha256:source-capability-5".to_string(),
            storage_state_digest: "sha256:storage-state-7".to_string(),
            critical_load_ids: vec!["critical-propulsion".to_string()],
            usable_storage_energy_lower_j: usable_energy_j,
            segments,
        }
    }

    #[test]
    fn adequate_source_storage_and_energy_are_sustained() {
        let p = profile(
            2_000_000,
            vec![
                PostFaultSupplySegment {
                    segment_id: "initial-recovery".to_string(),
                    start_seconds: 0,
                    duration_seconds: 60,
                    non_storage_source_capacity_lower_w: 70_000,
                    storage_discharge_power_lower_w: 30_000,
                    critical_loads: vec![available_load("critical-propulsion", 90_000)],
                },
                PostFaultSupplySegment {
                    segment_id: "continued-mission".to_string(),
                    start_seconds: 60,
                    duration_seconds: 120,
                    non_storage_source_capacity_lower_w: 70_000,
                    storage_discharge_power_lower_w: 10_000,
                    critical_loads: vec![available_load("critical-propulsion", 75_000)],
                },
            ],
        );
        assert_eq!(
            evaluate_post_fault_supply(&p).unwrap(),
            ElectricalSupplyDisposition::SustainedElectricalSupply {
                total_duration_seconds: 180,
                maximum_storage_power_required_w: 20_000,
                required_storage_energy_j: 1_800_000,
                remaining_storage_energy_lower_j: 200_000,
            }
        );
    }

    #[test]
    fn reachable_path_does_not_hide_a_storage_power_shortfall() {
        let p = profile(
            2_000_000,
            vec![PostFaultSupplySegment {
                segment_id: "power-limited".to_string(),
                start_seconds: 0,
                duration_seconds: 60,
                non_storage_source_capacity_lower_w: 70_000,
                storage_discharge_power_lower_w: 15_000,
                critical_loads: vec![available_load("critical-propulsion", 90_000)],
            }],
        );
        assert_eq!(
            evaluate_post_fault_supply(&p).unwrap(),
            ElectricalSupplyDisposition::NotSustainedPowerLimit {
                segment_id: "power-limited".to_string(),
                required_storage_power_w: 20_000,
                available_storage_power_lower_w: 15_000,
                unmet_power_w: 5_000,
            }
        );
    }

    #[test]
    fn adequate_instantaneous_power_does_not_hide_energy_exhaustion() {
        let p = profile(
            500_000,
            vec![PostFaultSupplySegment {
                segment_id: "energy-limited".to_string(),
                start_seconds: 0,
                duration_seconds: 360,
                non_storage_source_capacity_lower_w: 70_000,
                storage_discharge_power_lower_w: 10_000,
                critical_loads: vec![available_load("critical-propulsion", 80_000)],
            }],
        );
        assert_eq!(
            evaluate_post_fault_supply(&p).unwrap(),
            ElectricalSupplyDisposition::NotSustainedEnergyLimit {
                segment_id: "energy-limited".to_string(),
                required_storage_energy_j: 3_600_000,
                usable_storage_energy_lower_j: 500_000,
                deficit_energy_j: 3_100_000,
                reserve_depletion_time_seconds: 50,
            }
        );
    }

    #[test]
    fn confirmed_unavailable_path_is_not_sustained() {
        let mut load = available_load("critical-propulsion", 10_000);
        load.path = PostFaultPathEvidence::ConfirmedUnavailable {
            evidence_digest: "sha256:no-post-fault-path".to_string(),
        };
        let p = profile(
            0,
            vec![PostFaultSupplySegment {
                segment_id: "isolated-load".to_string(),
                start_seconds: 0,
                duration_seconds: 1,
                non_storage_source_capacity_lower_w: 100_000,
                storage_discharge_power_lower_w: 0,
                critical_loads: vec![load],
            }],
        );
        assert!(matches!(
            evaluate_post_fault_supply(&p).unwrap(),
            ElectricalSupplyDisposition::NotSustainedPath { .. }
        ));
    }

    #[test]
    fn unknown_path_is_not_evaluable() {
        let mut load = available_load("critical-propulsion", 10_000);
        load.path = PostFaultPathEvidence::Unknown;
        let p = profile(
            100_000,
            vec![PostFaultSupplySegment {
                segment_id: "unknown-path".to_string(),
                start_seconds: 0,
                duration_seconds: 1,
                non_storage_source_capacity_lower_w: 100_000,
                storage_discharge_power_lower_w: 0,
                critical_loads: vec![load],
            }],
        );
        assert!(matches!(
            evaluate_post_fault_supply(&p).unwrap(),
            ElectricalSupplyDisposition::NotEvaluablePath { .. }
        ));
    }

    #[test]
    fn missing_identity_or_mismatched_load_manifest_is_rejected() {
        let p = profile(
            100_000,
            vec![PostFaultSupplySegment {
                segment_id: "manifest-mismatch".to_string(),
                start_seconds: 0,
                duration_seconds: 1,
                non_storage_source_capacity_lower_w: 100_000,
                storage_discharge_power_lower_w: 0,
                critical_loads: vec![available_load("wrong-load", 10_000)],
            }],
        );
        assert!(matches!(
            evaluate_post_fault_supply(&p),
            Err(SupplyProfileError::LoadManifestMismatch { .. })
        ));

        let mut p = p;
        p.configuration_digest.clear();
        assert_eq!(
            evaluate_post_fault_supply(&p),
            Err(SupplyProfileError::EmptyIdentity("configuration_digest"))
        );
    }

    #[test]
    fn timeline_must_be_contiguous_and_segment_ids_unique() {
        let p = profile(
            100_000,
            vec![
                PostFaultSupplySegment {
                    segment_id: "s1".to_string(),
                    start_seconds: 0,
                    duration_seconds: 1,
                    non_storage_source_capacity_lower_w: 10_000,
                    storage_discharge_power_lower_w: 0,
                    critical_loads: vec![available_load("critical-propulsion", 10_000)],
                },
                PostFaultSupplySegment {
                    segment_id: "s2".to_string(),
                    start_seconds: 2,
                    duration_seconds: 1,
                    non_storage_source_capacity_lower_w: 10_000,
                    storage_discharge_power_lower_w: 0,
                    critical_loads: vec![available_load("critical-propulsion", 10_000)],
                },
            ],
        );
        assert!(matches!(
            evaluate_post_fault_supply(&p),
            Err(SupplyProfileError::NonContiguousTimeline { .. })
        ));
    }
}
