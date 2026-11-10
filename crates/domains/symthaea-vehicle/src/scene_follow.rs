//! Opaque scene-derived Follow targets.
//!
//! This module is an internal capability boundary, not a perception stack. The current
//! repository has a validated scene-evidence *contract* but no qualified physical sensor
//! admission producer. Consequently, no public constructor creates a scene receipt and
//! deployment Follow remains unavailable until an internal adapter consumes an independently
//! validated scene/provenance receipt. Unit tests exercise the boundary with synthetic data.

use std::collections::BTreeSet;

use crate::types::{FollowConsumerProfile, FollowEvidenceError, VehicleCommand, VehicleState};

/// Evidence role assigned by the validated scene-ingestion boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SceneSourceRole {
    LocalSensorObservation,
    SyntheticSensorObservation,
    SimulatorOracle,
    RemoteCooperativeClaim,
    MapPrior,
    Unknown,
}

/// Source record copied from one admitted scene snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SceneSourceReceipt {
    source_id: String,
    source_event_id: String,
    acquisition_tick_ns: u64,
    receive_tick_ns: u64,
    role: SceneSourceRole,
    current: bool,
    calibration_current: bool,
    frame_id: String,
    clock_domain_id: String,
    clock_transform_id: String,
    calibration_profile_id: String,
    extrinsic_profile_id: String,
}

/// Track state classification from the scene admission result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SceneTrackState {
    Observed,
    Estimated,
    Predicted,
    Occluded,
    Contradicted,
    Stale,
    Unknown,
}

/// Track-association status from the scene admission result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SceneAssociationState {
    Confirmed,
    Provisional,
    Ambiguous,
    Split,
    Merged,
    Unmatched,
}

/// Visibility state from the scene admission result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SceneVisibility {
    Visible,
    PartiallyOccluded,
    Occluded,
    OutOfView,
    Unknown,
}

/// One metric object hypothesis, already resolved against the scene snapshot.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SceneTrackReceipt {
    hypothesis_id: String,
    track_lineage_id: String,
    reference_frame_id: String,
    measurement_tick_ns: u64,
    position_x_m: f64,
    position_y_m: f64,
    velocity_x_mps: f64,
    extent_width_m: f64,
    distance_uncertainty_radius_m: f64,
    association: SceneAssociationState,
    state: SceneTrackState,
    visibility: SceneVisibility,
    source_ids: Vec<String>,
    /// Identifies how the upstream scene admission derived this uncertainty radius.
    uncertainty_semantics_id: String,
    /// Number of alternative hypotheses attached to this physical track lineage.
    hypotheses_for_lineage: usize,
}

/// Internal immutable result of scene admission.
///
/// Fields are private and this type deliberately does not implement Serialize or
/// Deserialize. The only non-test constructor is crate-private and must be called by the
/// future scene-ingestion adapter *after* applying the v0.1 scene contract, current clock /
/// transform / calibration checks, and the Mycelix source-provenance policy. A string ID or
/// serialized claim cannot recreate this type from outside the vehicle crate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SceneAdmissionReceipt {
    snapshot_id: String,
    schema_version: String,
    snapshot_digest: [u8; 32],
    provenance_receipt_digest: [u8; 32],
    sequence: u64,
    frame_id: String,
    frame_transform_id: String,
    clock_domain_id: String,
    clock_transform_id: String,
    event_tick_ns: u64,
    receive_tick_ns: u64,
    processing_tick_ns: u64,
    evaluated_at_tick_ns: u64,
    clock_alignment_admitted: bool,
    frame_transform_admitted: bool,
    calibration_currentness_admitted: bool,
    numeric_validity_admitted: bool,
    x_axis_is_forward: bool,
    ego_position_x_m: f64,
    ego_velocity_x_mps: f64,
    lane_min_y_m: f64,
    lane_max_y_m: f64,
    sources: Vec<SceneSourceReceipt>,
    tracks: Vec<SceneTrackReceipt>,
}

impl SceneAdmissionReceipt {
    /// Construct from the internal scene-admission result, never from wire/deserialized data.
    ///
    /// This constructor is crate-private. No production caller currently exists; the type
    /// therefore cannot yet be populated from a physical sensor pipeline.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // No qualified physical scene-admission producer exists yet.
    pub(crate) fn from_validated_scene_admission(
        snapshot_id: String,
        schema_version: String,
        snapshot_digest: [u8; 32],
        provenance_receipt_digest: [u8; 32],
        sequence: u64,
        frame_id: String,
        frame_transform_id: String,
        clock_domain_id: String,
        clock_transform_id: String,
        event_tick_ns: u64,
        receive_tick_ns: u64,
        processing_tick_ns: u64,
        evaluated_at_tick_ns: u64,
        clock_alignment_admitted: bool,
        frame_transform_admitted: bool,
        calibration_currentness_admitted: bool,
        numeric_validity_admitted: bool,
        x_axis_is_forward: bool,
        ego_position_x_m: f64,
        ego_velocity_x_mps: f64,
        lane_min_y_m: f64,
        lane_max_y_m: f64,
        sources: Vec<SceneSourceReceipt>,
        tracks: Vec<SceneTrackReceipt>,
    ) -> Result<Self, FollowEvidenceError> {
        if snapshot_id.is_empty()
            || schema_version != "0.1.0"
            || frame_id.is_empty()
            || frame_transform_id.is_empty()
            || clock_domain_id.is_empty()
            || clock_transform_id.is_empty()
            || snapshot_digest == [0; 32]
            || provenance_receipt_digest == [0; 32]
            || sequence == 0
            || !(event_tick_ns <= receive_tick_ns
                && receive_tick_ns <= processing_tick_ns
                && processing_tick_ns <= evaluated_at_tick_ns)
            || !ego_position_x_m.is_finite()
            || !ego_velocity_x_mps.is_finite()
            || !lane_min_y_m.is_finite()
            || !lane_max_y_m.is_finite()
            || lane_min_y_m >= lane_max_y_m
        {
            return Err(FollowEvidenceError::SceneAdmissionRejected);
        }

        Ok(Self {
            snapshot_id,
            schema_version,
            snapshot_digest,
            provenance_receipt_digest,
            sequence,
            frame_id,
            frame_transform_id,
            clock_domain_id,
            clock_transform_id,
            event_tick_ns,
            receive_tick_ns,
            processing_tick_ns,
            evaluated_at_tick_ns,
            clock_alignment_admitted,
            frame_transform_admitted,
            calibration_currentness_admitted,
            numeric_validity_admitted,
            x_axis_is_forward,
            ego_position_x_m,
            ego_velocity_x_mps,
            lane_min_y_m,
            lane_max_y_m,
            sources,
            tracks,
        })
    }
}

/// Local Follow capability. It is intentionally non-serializable and cannot be created by a
/// caller from a scene ID, track ID, source strings or scalar gap.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedLocalFollowTarget {
    scene_snapshot_id: String,
    schema_version: String,
    scene_digest: [u8; 32],
    provenance_receipt_digest: [u8; 32],
    scene_sequence: u64,
    scene_receive_tick_ns: u64,
    scene_processing_tick_ns: u64,
    track_hypothesis_id: String,
    track_lineage_id: String,
    source_ids: Vec<String>,
    frame_id: String,
    frame_transform_id: String,
    clock_domain_id: String,
    clock_transform_id: String,
    target_measurement_tick_ns: u64,
    uncertainty_semantics_id: String,
    observation_age_s: f64,
    gap_lower_bound_m: f64,
    relative_speed_mps: f64,
    distance_uncertainty_radius_m: f64,
    evaluated_at_s: f64,
    max_age_s: f64,
}

impl AdmittedLocalFollowTarget {
    pub fn scene_snapshot_id(&self) -> &str {
        &self.scene_snapshot_id
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn frame_transform_id(&self) -> &str {
        &self.frame_transform_id
    }

    pub fn clock_domain_id(&self) -> &str {
        &self.clock_domain_id
    }

    pub fn clock_transform_id(&self) -> &str {
        &self.clock_transform_id
    }

    pub fn target_measurement_tick_ns(&self) -> u64 {
        self.target_measurement_tick_ns
    }

    pub fn uncertainty_semantics_id(&self) -> &str {
        &self.uncertainty_semantics_id
    }

    pub fn scene_digest(&self) -> [u8; 32] {
        self.scene_digest
    }

    pub fn provenance_receipt_digest(&self) -> [u8; 32] {
        self.provenance_receipt_digest
    }

    pub fn scene_sequence(&self) -> u64 {
        self.scene_sequence
    }

    pub fn scene_receive_tick_ns(&self) -> u64 {
        self.scene_receive_tick_ns
    }

    pub fn scene_processing_tick_ns(&self) -> u64 {
        self.scene_processing_tick_ns
    }

    pub fn track_hypothesis_id(&self) -> &str {
        &self.track_hypothesis_id
    }

    pub fn track_lineage_id(&self) -> &str {
        &self.track_lineage_id
    }

    pub fn source_ids(&self) -> &[String] {
        &self.source_ids
    }

    pub fn frame_id(&self) -> &str {
        &self.frame_id
    }

    pub fn observation_age_s(&self) -> f64 {
        self.observation_age_s
    }

    pub fn gap_lower_bound_m(&self) -> f64 {
        self.gap_lower_bound_m
    }

    pub fn relative_speed_mps(&self) -> f64 {
        self.relative_speed_mps
    }

    pub fn evaluated_at_s(&self) -> f64 {
        self.evaluated_at_s
    }

    pub fn max_age_s(&self) -> f64 {
        self.max_age_s
    }

    /// This is the radius supplied by the upstream admitted uncertainty calculation. It is
    /// not independently promoted here into a calibrated probability or deterministic bound.
    pub fn distance_uncertainty_radius_m(&self) -> f64 {
        self.distance_uncertainty_radius_m
    }
}

/// Derive a local Follow target from one opaque scene-admission receipt.
///
/// Exactly one nearest, in-lane actor must be represented by a confirmed, visible and current
/// hypothesis. Ambiguous associations, competing hypotheses, unknown/stale sources and
/// incomplete scene admission are rejected. Gap and closing speed are derived from the ego
/// and actor metric positions/velocities in the same snapshot; caller-supplied geometry is
/// never accepted.
#[allow(dead_code)] // Reserved for the future internal scene-ingestion adapter.
pub(crate) fn admit_local_follow_target(
    receipt: &SceneAdmissionReceipt,
    profile: &FollowConsumerProfile,
) -> Result<AdmittedLocalFollowTarget, FollowEvidenceError> {
    let (expected_scene, expected_frame, max_age_s, max_uncertainty_m) = match profile {
        FollowConsumerProfile::LocalPerception {
            scene_snapshot_id,
            frame_id,
            max_age_s,
            max_distance_error_bound_m,
        }
        | FollowConsumerProfile::Deployment {
            scene_snapshot_id,
            frame_id,
            max_age_s,
            max_distance_error_bound_m,
        } => (
            scene_snapshot_id,
            frame_id,
            *max_age_s,
            *max_distance_error_bound_m,
        ),
        FollowConsumerProfile::SimulationBenchmark { .. } => {
            return Err(FollowEvidenceError::WrongConsumerProfile);
        }
    };

    if receipt.snapshot_id != *expected_scene || receipt.frame_id != *expected_frame {
        return Err(FollowEvidenceError::ContextMismatch);
    }
    if !receipt.clock_alignment_admitted
        || !receipt.frame_transform_admitted
        || !receipt.calibration_currentness_admitted
        || !receipt.numeric_validity_admitted
        || !receipt.x_axis_is_forward
    {
        return Err(FollowEvidenceError::SceneAdmissionRejected);
    }
    if !max_age_s.is_finite()
        || max_age_s < 0.0
        || !max_uncertainty_m.is_finite()
        || max_uncertainty_m < 0.0
    {
        return Err(FollowEvidenceError::InvalidNumericValue);
    }

    let snapshot_age_ns = receipt
        .evaluated_at_tick_ns
        .checked_sub(receipt.event_tick_ns)
        .ok_or(FollowEvidenceError::StaleOrFutureObservation)?;
    let snapshot_age_s = snapshot_age_ns as f64 / 1_000_000_000.0;
    if !snapshot_age_s.is_finite() || snapshot_age_s > max_age_s {
        return Err(FollowEvidenceError::StaleOrFutureObservation);
    }

    let mut source_ids = BTreeSet::new();
    if receipt.sources.is_empty() {
        return Err(FollowEvidenceError::MissingProvenance);
    }
    for source in &receipt.sources {
        if source.source_id.is_empty()
            || source.source_event_id.is_empty()
            || source.frame_id.is_empty()
            || source.clock_domain_id.is_empty()
            || source.clock_transform_id.is_empty()
            || source.calibration_profile_id.is_empty()
            || source.extrinsic_profile_id.is_empty()
            || source.acquisition_tick_ns > source.receive_tick_ns
            || source.receive_tick_ns > receipt.evaluated_at_tick_ns
            || !source_ids.insert(source.source_id.as_str())
        {
            return Err(FollowEvidenceError::SceneAdmissionRejected);
        }
    }

    // The closest positive-x actor whose lateral extent falls inside the admitted lane
    // corridor is the lead candidate. Do not skip an ambiguous/unknown closer candidate to
    // select a more convenient farther actor.
    let mut candidates: Vec<&SceneTrackReceipt> = receipt
        .tracks
        .iter()
        .filter(|track| {
            track.reference_frame_id == receipt.frame_id
                && track.position_x_m > receipt.ego_position_x_m
                && track.extent_width_m.is_finite()
                && track.extent_width_m >= 0.0
                && track.position_y_m - track.extent_width_m / 2.0 >= receipt.lane_min_y_m
                && track.position_y_m + track.extent_width_m / 2.0 <= receipt.lane_max_y_m
        })
        .collect();
    candidates.sort_by(|left, right| {
        left.position_x_m
            .total_cmp(&right.position_x_m)
            .then_with(|| left.hypothesis_id.cmp(&right.hypothesis_id))
    });
    let target = candidates
        .first()
        .copied()
        .ok_or(FollowEvidenceError::Missing)?;

    if target.association != SceneAssociationState::Confirmed
        || target.hypotheses_for_lineage != 1
        || target.state != SceneTrackState::Observed
            && target.state != SceneTrackState::Estimated
        || target.visibility != SceneVisibility::Visible
        || target.hypothesis_id.is_empty()
        || target.track_lineage_id.is_empty()
        || target.source_ids.is_empty()
        || target.measurement_tick_ns > receipt.event_tick_ns
        || target.measurement_tick_ns > receipt.evaluated_at_tick_ns
        || !target.position_x_m.is_finite()
        || !target.position_y_m.is_finite()
        || !target.velocity_x_mps.is_finite()
        || !target.extent_width_m.is_finite()
        || target.extent_width_m < 0.0
        || target.uncertainty_semantics_id.is_empty()
        || !target.distance_uncertainty_radius_m.is_finite()
        || target.distance_uncertainty_radius_m < 0.0
    {
        return Err(FollowEvidenceError::SceneAdmissionRejected);
    }
    let mut unique_target_sources = BTreeSet::new();
    for source_id in &target.source_ids {
        if !unique_target_sources.insert(source_id.as_str()) {
            return Err(FollowEvidenceError::MissingProvenance);
        }
        let Some(source) = receipt
            .sources
            .iter()
            .find(|source| source.source_id == *source_id)
        else {
            return Err(FollowEvidenceError::MissingProvenance);
        };
        if source.role != SceneSourceRole::LocalSensorObservation
            || !source.current
            || !source.calibration_current
            || receipt.evaluated_at_tick_ns.saturating_sub(source.acquisition_tick_ns)
                > (max_age_s * 1_000_000_000.0) as u64
        {
            return Err(FollowEvidenceError::SceneAdmissionRejected);
        }
    }
    if target.distance_uncertainty_radius_m > max_uncertainty_m {
        return Err(FollowEvidenceError::ExcessiveDistanceErrorBound);
    }

    let target_age_ns = receipt.evaluated_at_tick_ns - target.measurement_tick_ns;
    let observation_age_s = target_age_ns as f64 / 1_000_000_000.0;
    if !observation_age_s.is_finite() || observation_age_s > max_age_s {
        return Err(FollowEvidenceError::StaleOrFutureObservation);
    }

    let gap_m = target.position_x_m - receipt.ego_position_x_m;
    let relative_speed_mps = target.velocity_x_mps - receipt.ego_velocity_x_mps;
    if !gap_m.is_finite() || gap_m <= 0.0 || !relative_speed_mps.is_finite() {
        return Err(FollowEvidenceError::InvalidNumericValue);
    }

    Ok(AdmittedLocalFollowTarget {
        scene_snapshot_id: receipt.snapshot_id.clone(),
        schema_version: receipt.schema_version.clone(),
        scene_digest: receipt.snapshot_digest,
        provenance_receipt_digest: receipt.provenance_receipt_digest,
        scene_sequence: receipt.sequence,
        scene_receive_tick_ns: receipt.receive_tick_ns,
        scene_processing_tick_ns: receipt.processing_tick_ns,
        track_hypothesis_id: target.hypothesis_id.clone(),
        track_lineage_id: target.track_lineage_id.clone(),
        source_ids: target.source_ids.clone(),
        frame_id: receipt.frame_id.clone(),
        frame_transform_id: receipt.frame_transform_id.clone(),
        clock_domain_id: receipt.clock_domain_id.clone(),
        clock_transform_id: receipt.clock_transform_id.clone(),
        target_measurement_tick_ns: target.measurement_tick_ns,
        uncertainty_semantics_id: target.uncertainty_semantics_id.clone(),
        observation_age_s,
        gap_lower_bound_m: (gap_m - target.distance_uncertainty_radius_m).max(0.0),
        relative_speed_mps,
        distance_uncertainty_radius_m: target.distance_uncertainty_radius_m,
        evaluated_at_s: receipt.evaluated_at_tick_ns as f64 / 1_000_000_000.0,
        max_age_s,
    })
}

/// Follow PD command using the opaque, scene-derived local target.
pub fn pd_cruise_with_admitted_local_target(
    state: &VehicleState,
    target_speed: f64,
    road_heading: f64,
    target_gap: f64,
    target: &AdmittedLocalFollowTarget,
) -> Result<VehicleCommand, FollowEvidenceError> {
    if !target_speed.is_finite()
        || target_speed < 0.0
        || !road_heading.is_finite()
        || !target_gap.is_finite()
        || target_gap < 0.0
        || !target.gap_lower_bound_m.is_finite()
        || !target.relative_speed_mps.is_finite()
        || !state.timestamp.is_finite()
        || !target.evaluated_at_s.is_finite()
        || !target.max_age_s.is_finite()
        || target.max_age_s < 0.0
    {
        return Err(FollowEvidenceError::InvalidNumericValue);
    }
    if state.timestamp < target.evaluated_at_s
        || state.timestamp - target.evaluated_at_s > target.max_age_s
    {
        return Err(FollowEvidenceError::StaleOrFutureObservation);
    }
    let gap_error = target.gap_lower_bound_m - target_gap;
    let speed_cmd = (target_speed + gap_error * 0.3).clamp(0.0, target_speed * 1.2);
    Ok(crate::types::pd_cruise_baseline_with_road(
        state,
        speed_cmd,
        road_heading,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_receipt() -> SceneAdmissionReceipt {
        SceneAdmissionReceipt::from_validated_scene_admission(
            "scene-1".into(),
            "0.1.0".into(),
            [0x11; 32],
            [0x22; 32],
            12,
            "road-local".into(),
            "map-to-road-local-v1".into(),
            "vehicle-monotonic".into(),
            "clock-transform-v1".into(),
            12_000_000_000,
            12_001_000_000,
            12_002_000_000,
            12_020_000_000,
            true,
            true,
            true,
            true,
            true,
            0.0,
            4.0,
            -1.8,
            1.8,
            vec![SceneSourceReceipt {
                source_id: "front-lidar".into(),
                source_event_id: "lidar-scan-12".into(),
                acquisition_tick_ns: 12_000_000_000,
                receive_tick_ns: 12_001_000_000,
                role: SceneSourceRole::LocalSensorObservation,
                current: true,
                calibration_current: true,
                frame_id: "road-local".into(),
                clock_domain_id: "vehicle-monotonic".into(),
                clock_transform_id: "clock-transform-v1".into(),
                calibration_profile_id: "lidar-cal-v1".into(),
                extrinsic_profile_id: "lidar-extrinsic-v1".into(),
            }],
            vec![SceneTrackReceipt {
                hypothesis_id: "hyp-1".into(),
                track_lineage_id: "track-1".into(),
                reference_frame_id: "road-local".into(),
                measurement_tick_ns: 12_000_000_000,
                position_x_m: 15.0,
                position_y_m: 0.2,
                velocity_x_mps: 3.0,
                extent_width_m: 1.0,
                distance_uncertainty_radius_m: 2.0,
                association: SceneAssociationState::Confirmed,
                state: SceneTrackState::Observed,
                visibility: SceneVisibility::Visible,
                source_ids: vec!["front-lidar".into()],
                uncertainty_semantics_id: "scene-track-interval-bound-v1".into(),
                hypotheses_for_lineage: 1,
            }],
        )
        .expect("valid synthetic admitted-scene fixture")
    }

    fn deployment_profile() -> FollowConsumerProfile {
        FollowConsumerProfile::Deployment {
            scene_snapshot_id: "scene-1".into(),
            frame_id: "road-local".into(),
            max_age_s: 0.1,
            max_distance_error_bound_m: 3.0,
        }
    }

    #[test]
    fn derives_gap_and_relative_speed_from_one_opaque_scene_receipt() {
        let admitted = admit_local_follow_target(&valid_receipt(), &deployment_profile())
            .expect("synthetic fixture should exercise the internal capability path");
        assert!((admitted.gap_lower_bound_m() - 13.0).abs() < 1e-12);
        assert!((admitted.relative_speed_mps() + 1.0).abs() < 1e-12);
        assert_eq!(admitted.scene_sequence(), 12);
        assert_eq!(admitted.scene_receive_tick_ns(), 12_001_000_000);
        assert_eq!(admitted.scene_processing_tick_ns(), 12_002_000_000);
        assert_eq!(admitted.track_hypothesis_id(), "hyp-1");
        assert_eq!(admitted.track_lineage_id(), "track-1");
        assert_eq!(admitted.scene_digest(), [0x11; 32]);
        assert_eq!(admitted.schema_version(), "0.1.0");
        assert_eq!(admitted.frame_transform_id(), "map-to-road-local-v1");
        assert_eq!(admitted.clock_domain_id(), "vehicle-monotonic");
        assert_eq!(admitted.clock_transform_id(), "clock-transform-v1");
        assert_eq!(admitted.uncertainty_semantics_id(), "scene-track-interval-bound-v1");
    }

    #[test]
    fn rejects_a_receipt_from_another_snapshot() {
        let mut profile = deployment_profile();
        if let FollowConsumerProfile::Deployment {
            scene_snapshot_id, ..
        } = &mut profile
        {
            *scene_snapshot_id = "scene-other".into();
        }
        assert_eq!(
            admit_local_follow_target(&valid_receipt(), &profile),
            Err(FollowEvidenceError::ContextMismatch)
        );
    }

    #[test]
    fn rejects_stale_scene_receipt() {
        let mut receipt = valid_receipt();
        receipt.evaluated_at_tick_ns = 12_500_000_000;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::StaleOrFutureObservation)
        );
    }

    #[test]
    fn rejects_invalid_clock_transform_or_calibration() {
        let mut receipt = valid_receipt();
        receipt.calibration_currentness_admitted = false;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
        let mut receipt = valid_receipt();
        receipt.clock_alignment_admitted = false;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
    }

    #[test]
    fn rejects_remote_or_synthetic_source_roles() {
        for role in [
            SceneSourceRole::SyntheticSensorObservation,
            SceneSourceRole::SimulatorOracle,
            SceneSourceRole::RemoteCooperativeClaim,
            SceneSourceRole::MapPrior,
            SceneSourceRole::Unknown,
        ] {
            let mut receipt = valid_receipt();
            receipt.sources[0].role = role;
            assert_eq!(
                admit_local_follow_target(&receipt, &deployment_profile()),
                Err(FollowEvidenceError::SceneAdmissionRejected)
            );
        }
    }

    #[test]
    fn rejects_ambiguous_track_lineage_and_hidden_actor() {
        let mut receipt = valid_receipt();
        receipt.tracks[0].hypotheses_for_lineage = 2;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
        let mut receipt = valid_receipt();
        receipt.tracks[0].visibility = SceneVisibility::PartiallyOccluded;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
    }

    #[test]
    fn rejects_unknown_or_unregistered_source_reference() {
        let mut receipt = valid_receipt();
        receipt.tracks[0].source_ids = vec!["unregistered".into()];
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::MissingProvenance)
        );
    }

    #[test]
    fn rejects_uncertainty_radius_above_consumer_limit() {
        let mut receipt = valid_receipt();
        receipt.tracks[0].distance_uncertainty_radius_m = 3.1;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::ExcessiveDistanceErrorBound)
        );
    }

    #[test]
    fn chooses_nearest_lead_candidate_and_never_skips_ambiguous_nearer_target() {
        let mut receipt = valid_receipt();
        let mut farther = receipt.tracks[0].clone();
        farther.hypothesis_id = "hyp-2".into();
        farther.track_lineage_id = "track-2".into();
        farther.position_x_m = 25.0;
        receipt.tracks[0].position_x_m = 15.0;
        receipt.tracks.push(farther);
        let admitted = admit_local_follow_target(&receipt, &deployment_profile()).unwrap();
        assert_eq!(admitted.track_hypothesis_id(), "hyp-1");

        receipt.tracks[0].association = SceneAssociationState::Ambiguous;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
    }

    #[test]
    fn unrelated_advisories_do_not_replace_a_local_sensor_track() {
        let mut receipt = valid_receipt();
        receipt.sources.push(SceneSourceReceipt {
            source_id: "v2x-peer".into(),
            source_event_id: "peer-event-4".into(),
            acquisition_tick_ns: 12_000_000_000,
            receive_tick_ns: 12_001_000_000,
            role: SceneSourceRole::RemoteCooperativeClaim,
            current: true,
            calibration_current: true,
            frame_id: "peer-frame".into(),
            clock_domain_id: "peer-clock".into(),
            clock_transform_id: "peer-to-vehicle-clock-v1".into(),
            calibration_profile_id: "peer-claim-only".into(),
            extrinsic_profile_id: "peer-claim-only".into(),
        });
        assert!(admit_local_follow_target(&receipt, &deployment_profile()).is_ok());
        receipt.tracks[0].source_ids = vec!["v2x-peer".into()];
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
    }

    #[test]
    fn command_consumer_rejects_reused_or_future_scene_capability() {
        let admitted =
            admit_local_follow_target(&valid_receipt(), &deployment_profile()).unwrap();
        let mut state = VehicleState::cruising(4.0);
        state.timestamp = admitted.evaluated_at_s() + admitted.max_age_s() + 0.001;
        assert_eq!(
            pd_cruise_with_admitted_local_target(&state, 10.0, 0.0, 20.0, &admitted),
            Err(FollowEvidenceError::StaleOrFutureObservation)
        );
        state.timestamp = admitted.evaluated_at_s() - 0.001;
        assert_eq!(
            pd_cruise_with_admitted_local_target(&state, 10.0, 0.0, 20.0, &admitted),
            Err(FollowEvidenceError::StaleOrFutureObservation)
        );
    }

    #[test]
    fn rejects_stale_or_future_source_evidence() {
        let mut receipt = valid_receipt();
        receipt.sources[0].acquisition_tick_ns = 11_000_000_000;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
        let mut receipt = valid_receipt();
        receipt.sources[0].receive_tick_ns = 12_030_000_000;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
    }

    #[test]
    fn rejects_scene_hypotheses_with_unknown_uncertainty_semantics() {
        let mut receipt = valid_receipt();
        receipt.tracks[0].uncertainty_semantics_id.clear();
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
    }

    #[test]
    fn rejects_non_forward_road_frame() {
        let mut receipt = valid_receipt();
        receipt.x_axis_is_forward = false;
        assert_eq!(
            admit_local_follow_target(&receipt, &deployment_profile()),
            Err(FollowEvidenceError::SceneAdmissionRejected)
        );
    }
}
