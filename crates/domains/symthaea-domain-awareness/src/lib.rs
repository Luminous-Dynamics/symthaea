// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-first domain awareness primitives for safety-critical physical systems.
//!
//! This crate deliberately models `Observation -> Track -> Hypotheses -> Risk`.
//! It does **not** define a `Target` or `Hostile` primitive and does not grant
//! physical authority. Cooperative identity, classifier output, and human reports
//! remain evidence rather than truth. Downstream actuation must pass through an
//! independent authority and safety boundary.

#![deny(unsafe_code)]

pub mod operational_domain;
pub mod safety_case;
pub mod spatial;
pub use safety_case::domain_awareness_safety_case;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Physical operating domain in which an observation or track exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    Air,
    Surface,
    Subsurface,
    Land,
    Space,
    CrossDomain,
}

/// Sensor or evidence modality. A modality is evidence provenance, not identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Radar,
    ElectroOptical,
    Infrared,
    Lidar,
    Acoustic,
    RadioFrequency,
    Sonar,
    Gnss,
    CooperativeAirIdentity,
    CooperativeMaritimeIdentity,
    Weather,
    HumanReport,
    ExternalAuthority,
    Other,
}

/// Integrity status supplied by a verifier outside this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntegrityStatus {
    Unverified,
    Verified,
    Failed,
}

/// Current health of the producing sensor or evidence source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorHealth {
    Nominal,
    Degraded,
    Unavailable,
    Suspect,
}

impl SensorHealth {
    /// Upper bound on how much trust a consumer should assign to this health state.
    /// This is intentionally monotonic: degradation can never increase the cap.
    pub const fn trust_cap(self) -> f64 {
        match self {
            Self::Nominal => 1.0,
            Self::Degraded => 0.70,
            Self::Suspect => 0.35,
            Self::Unavailable => 0.0,
        }
    }
}

/// Explicit epistemic state. Unknown and conflicting states are first-class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicState {
    Unknown,
    InsufficientEvidence,
    ConflictingEvidence,
    OutOfDistribution,
    Known,
}

/// Clock provenance and timing uncertainty for an observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeEvidence {
    pub observed_at_ms: u64,
    pub received_at_ms: u64,
    pub clock_source: String,
    pub clock_uncertainty_ms: u64,
    pub maximum_valid_age_ms: u64,
}

impl TimeEvidence {
    pub fn validate(&self) -> bool {
        !self.clock_source.trim().is_empty()
            && self.maximum_valid_age_ms > 0
            && self.observed_at_ms <= self.received_at_ms.saturating_add(self.clock_uncertainty_ms)
    }

    /// True when the observation remains inside its declared freshness envelope.
    pub fn is_fresh_at(&self, now_ms: u64) -> bool {
        self.validate()
            && now_ms >= self.observed_at_ms
            && now_ms.saturating_sub(self.observed_at_ms)
                <= self
                    .maximum_valid_age_ms
                    .saturating_add(self.clock_uncertainty_ms)
    }
}

/// Fault-domain lineage used to avoid counting correlated evidence as independent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EvidenceLineage {
    /// Physical sensing element or originating evidence source.
    pub physical_source_id: String,
    /// Processor/classifier that transformed the source observation.
    pub processor_id: String,
    /// Network or transport path by which the observation arrived.
    pub network_path: String,
    /// Clock domain used for time attribution.
    pub clock_domain: String,
}

impl EvidenceLineage {
    pub fn validate(&self) -> bool {
        !self.physical_source_id.trim().is_empty()
            && !self.processor_id.trim().is_empty()
            && !self.network_path.trim().is_empty()
            && !self.clock_domain.trim().is_empty()
    }

    /// Conservative independence key: observations from the same physical source
    /// are not independent even when processed or transported differently.
    pub fn physical_fault_domain(&self) -> &str {
        self.physical_source_id.as_str()
    }
}

/// Measurement uncertainty carried with every observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasurementUncertainty {
    /// One-sigma position uncertainty in meters where applicable.
    pub position_sigma_m: Option<f64>,
    /// One-sigma velocity uncertainty in meters/second where applicable.
    pub velocity_sigma_mps: Option<f64>,
    /// One-sigma bearing uncertainty in degrees where applicable.
    pub bearing_sigma_deg: Option<f64>,
}

impl MeasurementUncertainty {
    pub fn validate(&self) -> bool {
        [
            self.position_sigma_m,
            self.velocity_sigma_mps,
            self.bearing_sigma_deg,
        ]
        .into_iter()
        .flatten()
        .all(|value| value.is_finite() && value >= 0.0)
    }
}

/// Generic measurement payload. Domain-specific crates may wrap richer raw data
/// while publishing a normalized observation through one of these variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Measurement {
    /// Position and optional velocity in the declared coordinate frame.
    Kinematic {
        position: [f64; 3],
        velocity: Option<[f64; 3]>,
    },
    /// Bearing/elevation observation when range is not known.
    Bearing {
        azimuth_deg: f64,
        elevation_deg: Option<f64>,
    },
    /// Cooperative identity assertion such as Remote ID, ADS-B-like, AIS, or VDES.
    /// Presence of this assertion never establishes intent.
    CooperativeIdentity {
        scheme: String,
        asserted_id: String,
    },
    /// Bounded scalar evidence such as weather visibility or sensor-derived quality.
    Scalar {
        quantity: String,
        value: f64,
        unit: String,
    },
}

impl Measurement {
    pub fn validate(&self) -> bool {
        match self {
            Self::Kinematic { position, velocity } => {
                position.iter().all(|v| v.is_finite())
                    && velocity
                        .as_ref()
                        .is_none_or(|v| v.iter().all(|x| x.is_finite()))
            }
            Self::Bearing {
                azimuth_deg,
                elevation_deg,
            } => {
                azimuth_deg.is_finite()
                    && (-360.0..=360.0).contains(azimuth_deg)
                    && elevation_deg.is_none_or(|v| v.is_finite() && (-90.0..=90.0).contains(&v))
            }
            Self::CooperativeIdentity { scheme, asserted_id } => {
                !scheme.trim().is_empty() && !asserted_id.trim().is_empty()
            }
            Self::Scalar {
                quantity,
                value,
                unit,
            } => !quantity.trim().is_empty() && value.is_finite() && !unit.trim().is_empty(),
        }
    }
}

/// Canonical evidence envelope for physical-domain observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationEnvelope {
    pub observation_id: Uuid,
    pub source_id: String,
    pub sequence: u64,
    pub domain: Domain,
    pub modality: Modality,
    pub coordinate_frame: String,
    pub measurement: Measurement,
    pub uncertainty: MeasurementUncertainty,
    pub time: TimeEvidence,
    pub lineage: EvidenceLineage,
    pub integrity: IntegrityStatus,
    pub sensor_health: SensorHealth,
    /// Confidence assigned by the producing subsystem, bounded by sensor health.
    pub confidence: f64,
    /// References into an evidence store / DKG / audit log.
    pub evidence_refs: Vec<String>,
}

impl ObservationEnvelope {
    pub fn validate(&self) -> bool {
        !self.source_id.trim().is_empty()
            && !self.coordinate_frame.trim().is_empty()
            && self.measurement.validate()
            && self.uncertainty.validate()
            && self.time.validate()
            && self.lineage.validate()
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && self.confidence <= self.sensor_health.trust_cap()
            && self.evidence_refs.iter().all(|v| !v.trim().is_empty())
    }

    pub fn is_fresh_at(&self, now_ms: u64) -> bool {
        self.validate() && self.time.is_fresh_at(now_ms)
    }

    /// Returns a copy with health degraded while ensuring confidence cannot rise.
    pub fn with_health(mut self, new_health: SensorHealth) -> Self {
        self.sensor_health = new_health;
        self.confidence = self.confidence.min(new_health.trust_cap());
        self
    }
}

/// Count distinct physical fault domains in a set of otherwise valid observations.
/// Multiple classifiers or network feeds derived from one physical sensor count once.
pub fn independent_physical_sources(observations: &[ObservationEnvelope]) -> usize {
    observations
        .iter()
        .filter(|observation| observation.validate())
        .map(|observation| observation.lineage.physical_fault_domain())
        .collect::<BTreeSet<_>>()
        .len()
}

/// Kinematic estimate associated with a track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KinematicEstimate {
    pub coordinate_frame: String,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub position_sigma_m: f64,
    pub velocity_sigma_mps: f64,
    pub valid_at_ms: u64,
}

impl KinematicEstimate {
    pub fn validate(&self) -> bool {
        !self.coordinate_frame.trim().is_empty()
            && self.position.iter().all(|v| v.is_finite())
            && self.velocity.iter().all(|v| v.is_finite())
            && self.position_sigma_m.is_finite()
            && self.position_sigma_m >= 0.0
            && self.velocity_sigma_mps.is_finite()
            && self.velocity_sigma_mps >= 0.0
    }
}

/// Competing explanation for identity. Labels are intentionally open-ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityHypothesis {
    pub label: String,
    pub confidence: f64,
    pub evidence_ids: Vec<Uuid>,
    pub evidence_refs: Vec<String>,
}

impl IdentityHypothesis {
    pub fn validate(&self) -> bool {
        !self.label.trim().is_empty()
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && self.evidence_refs.iter().all(|v| !v.trim().is_empty())
    }
}

/// Competing explanation for observed behavior. This represents motion/context,
/// not moral intent and never grants authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorHypothesis {
    pub label: String,
    pub confidence: f64,
    pub evidence_ids: Vec<Uuid>,
}

impl BehaviorHypothesis {
    pub fn validate(&self) -> bool {
        !self.label.trim().is_empty()
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }
}

/// Evidence-backed track with uncertainty preserved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub track_id: Uuid,
    pub domain: Domain,
    pub epistemic_state: EpistemicState,
    pub kinematics: Option<KinematicEstimate>,
    pub observation_ids: Vec<Uuid>,
    pub identity_hypotheses: Vec<IdentityHypothesis>,
    pub behavior_hypotheses: Vec<BehaviorHypothesis>,
    pub last_updated_ms: u64,
}

impl Track {
    pub fn new(domain: Domain, now_ms: u64) -> Self {
        Self {
            track_id: Uuid::new_v4(),
            domain,
            epistemic_state: EpistemicState::Unknown,
            kinematics: None,
            observation_ids: Vec::new(),
            identity_hypotheses: Vec::new(),
            behavior_hypotheses: Vec::new(),
            last_updated_ms: now_ms,
        }
    }

    pub fn validate(&self) -> bool {
        self.kinematics.as_ref().is_none_or(KinematicEstimate::validate)
            && self.identity_hypotheses.iter().all(IdentityHypothesis::validate)
            && self.behavior_hypotheses.iter().all(BehaviorHypothesis::validate)
            && {
                let ids = self.observation_ids.iter().copied().collect::<BTreeSet<_>>();
                ids.len() == self.observation_ids.len()
            }
    }

    /// Attach evidence without silently changing identity or intent.
    pub fn attach_observation(&mut self, observation: &ObservationEnvelope, now_ms: u64) -> bool {
        if !observation.validate() || self.observation_ids.contains(&observation.observation_id) {
            return false;
        }
        self.observation_ids.push(observation.observation_id);
        self.last_updated_ms = now_ms;
        if self.epistemic_state == EpistemicState::Unknown {
            self.epistemic_state = EpistemicState::InsufficientEvidence;
        }
        true
    }

    /// Replace identity hypotheses only when all supplied hypotheses are valid.
    /// Conflicting non-trivial hypotheses remain explicitly conflicting.
    pub fn set_identity_hypotheses(&mut self, hypotheses: Vec<IdentityHypothesis>) -> bool {
        if hypotheses.iter().any(|h| !h.validate()) {
            return false;
        }
        self.epistemic_state = epistemic_state_for_hypotheses(&hypotheses);
        self.identity_hypotheses = hypotheses;
        true
    }
}

/// Determine epistemic state without forcing a winner from ambiguous evidence.
pub fn epistemic_state_for_hypotheses(hypotheses: &[IdentityHypothesis]) -> EpistemicState {
    if hypotheses.is_empty() {
        return EpistemicState::InsufficientEvidence;
    }
    let mut meaningful = hypotheses
        .iter()
        .filter(|h| h.validate() && h.confidence >= 0.20)
        .map(|h| h.confidence)
        .collect::<Vec<_>>();
    if meaningful.is_empty() {
        return EpistemicState::InsufficientEvidence;
    }
    meaningful.sort_by(|a, b| b.total_cmp(a));
    if meaningful.len() >= 2 && meaningful[1] >= meaningful[0] * 0.60 {
        EpistemicState::ConflictingEvidence
    } else if meaningful[0] >= 0.80 {
        EpistemicState::Known
    } else {
        EpistemicState::InsufficientEvidence
    }
}

/// Consequence-oriented risk assessment. This intentionally avoids hostility or
/// engagement semantics: risk is about potential harm to protected people/assets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub track_id: Uuid,
    /// Estimated probability of a harmful outcome in [0, 1].
    pub harmful_outcome_probability: f64,
    /// Consequence severity in [0, 1].
    pub consequence_severity: f64,
    /// Confidence in the risk model itself in [0, 1].
    pub model_confidence: f64,
    pub epistemic_state: EpistemicState,
    pub evidence_refs: Vec<String>,
}

impl RiskAssessment {
    pub fn validate(&self) -> bool {
        [
            self.harmful_outcome_probability,
            self.consequence_severity,
            self.model_confidence,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            && self.evidence_refs.iter().all(|v| !v.trim().is_empty())
    }

    /// Conservative scalar for prioritizing protective review. It is not authority.
    pub fn review_priority(&self) -> f64 {
        if !self.validate() {
            return 0.0;
        }
        self.harmful_outcome_probability * self.consequence_severity * self.model_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(source: &str, physical: &str, confidence: f64) -> ObservationEnvelope {
        ObservationEnvelope {
            observation_id: Uuid::new_v4(),
            source_id: source.to_string(),
            sequence: 1,
            domain: Domain::Air,
            modality: Modality::Radar,
            coordinate_frame: "local-enu".to_string(),
            measurement: Measurement::Kinematic {
                position: [10.0, 20.0, 30.0],
                velocity: Some([1.0, 0.0, 0.0]),
            },
            uncertainty: MeasurementUncertainty {
                position_sigma_m: Some(2.0),
                velocity_sigma_mps: Some(0.5),
                bearing_sigma_deg: None,
            },
            time: TimeEvidence {
                observed_at_ms: 1_000,
                received_at_ms: 1_010,
                clock_source: "ptp-a".to_string(),
                clock_uncertainty_ms: 5,
                maximum_valid_age_ms: 500,
            },
            lineage: EvidenceLineage {
                physical_source_id: physical.to_string(),
                processor_id: format!("processor-{source}"),
                network_path: format!("path-{source}"),
                clock_domain: "clock-a".to_string(),
            },
            integrity: IntegrityStatus::Verified,
            sensor_health: SensorHealth::Nominal,
            confidence,
            evidence_refs: vec![format!("evidence:{source}")],
        }
    }

    #[test]
    fn invalid_confidence_is_rejected() {
        assert!(!observation("radar-a", "sensor-a", 1.1).validate());
    }

    #[test]
    fn freshness_is_explicit_and_bounded() {
        let obs = observation("radar-a", "sensor-a", 0.9);
        assert!(obs.is_fresh_at(1_400));
        assert!(!obs.is_fresh_at(2_000));
    }

    #[test]
    fn degradation_never_increases_confidence() {
        let obs = observation("radar-a", "sensor-a", 0.9);
        let degraded = obs.clone().with_health(SensorHealth::Degraded);
        let suspect = degraded.clone().with_health(SensorHealth::Suspect);
        assert_eq!(degraded.confidence, 0.70);
        assert_eq!(suspect.confidence, 0.35);
        assert!(suspect.confidence <= degraded.confidence);
        assert!(degraded.confidence <= obs.confidence);
    }

    #[test]
    fn correlated_derivatives_count_as_one_physical_witness() {
        let a = observation("radar-classifier-a", "radar-physical-1", 0.8);
        let b = observation("radar-classifier-b", "radar-physical-1", 0.8);
        let c = observation("eo-a", "camera-physical-9", 0.7);
        assert_eq!(independent_physical_sources(&[a, b, c]), 2);
    }

    #[test]
    fn track_does_not_infer_identity_from_observation_presence() {
        let obs = observation("radar-a", "sensor-a", 0.9);
        let mut track = Track::new(Domain::Air, 1_010);
        assert!(track.attach_observation(&obs, 1_020));
        assert_eq!(track.epistemic_state, EpistemicState::InsufficientEvidence);
        assert!(track.identity_hypotheses.is_empty());
    }

    #[test]
    fn competing_identity_hypotheses_remain_conflicting() {
        let mut track = Track::new(Domain::Air, 0);
        let hypotheses = vec![
            IdentityHypothesis {
                label: "authorized-small-aircraft".to_string(),
                confidence: 0.55,
                evidence_ids: Vec::new(),
                evidence_refs: vec!["classifier:a".to_string()],
            },
            IdentityHypothesis {
                label: "bird-or-clutter".to_string(),
                confidence: 0.40,
                evidence_ids: Vec::new(),
                evidence_refs: vec!["classifier:b".to_string()],
            },
        ];
        assert!(track.set_identity_hypotheses(hypotheses));
        assert_eq!(track.epistemic_state, EpistemicState::ConflictingEvidence);
    }

    #[test]
    fn cooperative_identity_is_only_an_observation() {
        let mut obs = observation("remote-id-a", "radio-1", 0.8);
        obs.modality = Modality::CooperativeAirIdentity;
        obs.measurement = Measurement::CooperativeIdentity {
            scheme: "remote-id".to_string(),
            asserted_id: "example-aircraft".to_string(),
        };
        assert!(obs.validate());
        let mut track = Track::new(Domain::Air, 1_010);
        assert!(track.attach_observation(&obs, 1_020));
        assert!(track.identity_hypotheses.is_empty());
    }

    #[test]
    fn risk_priority_is_for_review_not_authority() {
        let risk = RiskAssessment {
            track_id: Uuid::new_v4(),
            harmful_outcome_probability: 0.5,
            consequence_severity: 0.8,
            model_confidence: 0.5,
            epistemic_state: EpistemicState::InsufficientEvidence,
            evidence_refs: vec!["twin:run-1".to_string()],
        };
        assert!(risk.validate());
        assert!((risk.review_priority() - 0.2).abs() < 1e-12);
    }
}
