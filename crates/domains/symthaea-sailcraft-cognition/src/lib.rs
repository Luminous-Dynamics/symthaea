// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware sailcraft perception and HDC/LTC temporal representation.
//!
//! This crate encodes typed sailcraft measurements into ContinuousHV and evolves
//! those representations through HdcLtcUnifiedNetwork. It deliberately has no
//! policy head, actuator command type, route mutation, or control authority.
//!
//! Non-qualified observations contribute an explicit status vector only. Their
//! numeric payload is never encoded. Unknown, stale and conflicted evidence is
//! not converted to a healthy zero or silently clamped into a physical range.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, UnifiedConfig,
    UnifiedNetworkConfig,
};
use symthaea_sailcraft::{Evidence, EvidenceState};

const MIN_LEVELS: usize = 2;
const MAX_LEVELS: usize = 64;
const DEFAULT_LEVELS: usize = 16;

const SCALAR_CHANNELS: &[&str] = &[
    "true_wind_speed_mps",
    "apparent_wind_speed_mps",
    "speed_through_water_mps",
    "speed_over_ground_mps",
    "current_speed_mps",
    "heel_degrees",
    "leeway_degrees",
    "rudder_angle_degrees",
    "sheet_angle_degrees",
    "waypoint_range_m",
    "actuator_latency_ms",
    "significant_wave_height_m",
];

const ANGLE_CHANNELS: &[&str] = &[
    "true_wind_from_degrees",
    "apparent_wind_from_degrees",
    "heading_degrees",
    "current_toward_degrees",
    "waypoint_bearing_degrees",
];

#[derive(Clone, Debug)]
pub struct SailObservation {
    /// Exact hull/rig/sail-plan/loading/software configuration identity.
    pub configuration_digest: [u8; 32],
    /// Declared common timebase identity; this is an identity binding, not a
    /// clock qualification or trusted-time attestation.
    pub common_timebase_id: String,

    pub true_wind_speed_mps: Evidence<f64>,
    pub true_wind_from_degrees: Evidence<f64>,
    pub apparent_wind_speed_mps: Evidence<f64>,
    pub apparent_wind_from_degrees: Evidence<f64>,
    pub heading_degrees: Evidence<f64>,
    pub speed_through_water_mps: Evidence<f64>,
    pub speed_over_ground_mps: Evidence<f64>,
    pub current_speed_mps: Evidence<f64>,
    pub current_toward_degrees: Evidence<f64>,
    pub heel_degrees: Evidence<f64>,
    pub leeway_degrees: Evidence<f64>,
    pub rudder_angle_degrees: Evidence<f64>,
    pub sheet_angle_degrees: Evidence<f64>,
    pub waypoint_range_m: Evidence<f64>,
    pub waypoint_bearing_degrees: Evidence<f64>,
    pub actuator_latency_ms: Evidence<f64>,
    pub significant_wave_height_m: Evidence<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationAdmission {
    /// Every declared channel was qualified and in-domain.
    Complete,
    /// At least one noncritical channel is not qualified.
    Degraded,
    /// At least one navigation-critical channel is not qualified.
    CriticalEvidenceIncomplete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelEvidenceState {
    pub channel: &'static str,
    pub state: EvidenceState,
    pub critical: bool,
    /// Source identity is retained as receipt metadata, not treated as signer
    /// authentication or physical-truth proof.
    pub source_artifact_digest: [u8; 32],
    pub source_timebase_id: String,
    pub age_seconds: f64,
    pub maximum_age_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct EncodedSailObservation {
    pub sensor_hv: ContinuousHV,
    pub admission: ObservationAdmission,
    pub channel_states: Vec<ChannelEvidenceState>,
    pub timebase_id: String,
    pub configuration_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub enum EncodeError {
    InvalidEncoderConfiguration,
    ConfigurationMismatch,
    InvalidConfigurationDigest,
    MissingCommonTimebase,
    QualifiedValueMissing { channel: &'static str },
    InvalidQualifiedMetadata { channel: &'static str, reason: &'static str },
    OutOfDomain {
        channel: &'static str,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for EncodeError {}

pub struct SailcraftEncoder {
    genesis: GenesisSeed,
    expected_configuration_digest: [u8; 32],
    level_vectors: Vec<ContinuousHV>,
    scalar_bases: BTreeMap<&'static str, ContinuousHV>,
    angle_sine_bases: BTreeMap<&'static str, ContinuousHV>,
    angle_cosine_bases: BTreeMap<&'static str, ContinuousHV>,
}

impl SailcraftEncoder {
    pub fn new(
        genesis: &GenesisSeed,
        expected_configuration_digest: [u8; 32],
        level_count: usize,
    ) -> Result<Self, EncodeError> {
        if expected_configuration_digest == [0; 32]
            || !(MIN_LEVELS..=MAX_LEVELS).contains(&level_count)
        {
            return Err(EncodeError::InvalidEncoderConfiguration);
        }

        let level_vectors = (0..level_count)
            .map(|index| {
                genesis.hv(
                    &format!("sailcraft::level::{index}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let scalar_bases = SCALAR_CHANNELS
            .iter()
            .map(|channel| {
                (
                    *channel,
                    genesis.hv(
                        &format!("sailcraft::channel::{channel}::scalar"),
                        HDC_DIMENSION,
                    ),
                )
            })
            .collect();

        let angle_sine_bases = ANGLE_CHANNELS
            .iter()
            .map(|channel| {
                (
                    *channel,
                    genesis.hv(
                        &format!("sailcraft::channel::{channel}::sin"),
                        HDC_DIMENSION,
                    ),
                )
            })
            .collect();

        let angle_cosine_bases = ANGLE_CHANNELS
            .iter()
            .map(|channel| {
                (
                    *channel,
                    genesis.hv(
                        &format!("sailcraft::channel::{channel}::cos"),
                        HDC_DIMENSION,
                    ),
                )
            })
            .collect();

        Ok(Self {
            genesis: genesis.clone(),
            expected_configuration_digest,
            level_vectors,
            scalar_bases,
            angle_sine_bases,
            angle_cosine_bases,
        })
    }

    fn level_encode(&self, normalized: f32) -> ContinuousHV {
        let index = ((normalized * self.level_vectors.len() as f32) as usize)
            .min(self.level_vectors.len() - 1);
        if index == 0 {
            self.level_vectors[0].clone()
        } else {
            let references: Vec<&ContinuousHV> =
                self.level_vectors[..=index].iter().collect();
            ContinuousHV::bundle(&references)
        }
    }

    fn evidence_value<'a>(
        &self,
        channel: &'static str,
        evidence: &'a Evidence<f64>,
        common_timebase_id: &str,
    ) -> Result<Option<f64>, EncodeError> {
        match evidence.state {
            EvidenceState::Qualified => {
                let Some(value) = evidence.value else {
                    return Err(EncodeError::QualifiedValueMissing { channel });
                };
                if evidence.source_artifact_digest == [0; 32] {
                    return Err(EncodeError::InvalidQualifiedMetadata {
                        channel,
                        reason: "zero source digest",
                    });
                }
                if evidence.timebase_id.trim().is_empty()
                    || evidence.timebase_id != common_timebase_id
                {
                    return Err(EncodeError::InvalidQualifiedMetadata {
                        channel,
                        reason: "timebase identity is missing or mismatched",
                    });
                }
                if !evidence.age_seconds.is_finite()
                    || !evidence.maximum_age_seconds.is_finite()
                    || evidence.age_seconds < 0.0
                    || evidence.maximum_age_seconds < 0.0
                    || evidence.age_seconds > evidence.maximum_age_seconds
                {
                    return Err(EncodeError::InvalidQualifiedMetadata {
                        channel,
                        reason: "invalid or exceeded freshness bound",
                    });
                }
                if !value.is_finite() {
                    return Err(EncodeError::InvalidQualifiedMetadata {
                        channel,
                        reason: "qualified value is non-finite",
                    });
                }
                Ok(Some(value))
            }
            EvidenceState::Missing
            | EvidenceState::Stale
            | EvidenceState::Unverified
            | EvidenceState::Conflicted => Ok(None),
        }
    }

    fn append_status(
        &self,
        channel: &'static str,
        evidence: &Evidence<f64>,
        critical: bool,
        components: &mut Vec<ContinuousHV>,
        weights: &mut Vec<f32>,
        states: &mut Vec<ChannelEvidenceState>,
    ) {
        let label = state_label(evidence.state);
        components.push(self.genesis.hv(
            &format!("sailcraft::channel::{channel}::status::{label}"),
            HDC_DIMENSION,
        ));
        weights.push(0.45);
        states.push(ChannelEvidenceState {
            channel,
            state: evidence.state,
            critical,
            source_artifact_digest: evidence.source_artifact_digest,
            source_timebase_id: evidence.timebase_id.clone(),
            age_seconds: evidence.age_seconds,
            maximum_age_seconds: evidence.maximum_age_seconds,
        });
    }

    fn append_scalar(
        &self,
        channel: &'static str,
        evidence: &Evidence<f64>,
        minimum: f64,
        maximum: f64,
        weight: f32,
        critical: bool,
        common_timebase_id: &str,
        components: &mut Vec<ContinuousHV>,
        weights: &mut Vec<f32>,
        states: &mut Vec<ChannelEvidenceState>,
    ) -> Result<(), EncodeError> {
        let value = self.evidence_value(channel, evidence, common_timebase_id)?;
        self.append_status(channel, evidence, critical, components, weights, states);

        if let Some(value) = value {
            if value < minimum || value > maximum {
                return Err(EncodeError::OutOfDomain {
                    channel,
                    value,
                    minimum,
                    maximum,
                });
            }
            let normalized = ((value - minimum) / (maximum - minimum)) as f32;
            let level = self.level_encode(normalized);
            let base = self.scalar_bases.get(channel).expect("scalar base initialized");
            components.push(base.bind(&level));
            weights.push(weight);
        }
        Ok(())
    }

    fn append_angle(
        &self,
        channel: &'static str,
        evidence: &Evidence<f64>,
        weight: f32,
        critical: bool,
        common_timebase_id: &str,
        components: &mut Vec<ContinuousHV>,
        weights: &mut Vec<f32>,
        states: &mut Vec<ChannelEvidenceState>,
    ) -> Result<(), EncodeError> {
        let value = self.evidence_value(channel, evidence, common_timebase_id)?;
        self.append_status(channel, evidence, critical, components, weights, states);

        if let Some(degrees) = value {
            if !(0.0..360.0).contains(&degrees) {
                return Err(EncodeError::OutOfDomain {
                    channel,
                    value: degrees,
                    minimum: 0.0,
                    maximum_exclusive: 360.0,
                });
            }
            let radians = degrees.to_radians();
            let sine = ((radians.sin() + 1.0) * 0.5) as f32;
            let cosine = ((radians.cos() + 1.0) * 0.5) as f32;
            let sine_level = self.level_encode(sine);
            let cosine_level = self.level_encode(cosine);
            let sine_base = self.angle_sine_bases.get(channel).expect("sine base initialized");
            let cosine_base = self.angle_cosine_bases.get(channel).expect("cosine base initialized");
            components.push(sine_base.bind(&sine_level));
            weights.push(weight * 0.5);
            components.push(cosine_base.bind(&cosine_level));
            weights.push(weight * 0.5);
        }
        Ok(())
    }

    pub fn encode(
        &self,
        observation: &SailObservation,
    ) -> Result<EncodedSailObservation, EncodeError> {
        if observation.configuration_digest != self.expected_configuration_digest {
            return Err(EncodeError::ConfigurationMismatch);
        }
        if observation.configuration_digest == [0; 32] {
            return Err(EncodeError::InvalidConfigurationDigest);
        }
        if observation.common_timebase_id.trim().is_empty() {
            return Err(EncodeError::MissingCommonTimebase);
        }

        let mut components = Vec::with_capacity(48);
        let mut weights = Vec::with_capacity(48);
        let mut states = Vec::with_capacity(17);

        // Circular bearings use sin/cos features; 359° and 1° remain adjacent.
        self.append_scalar(
            "true_wind_speed_mps", &observation.true_wind_speed_mps,
            0.0, 40.0, 1.2, true, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_angle(
            "true_wind_from_degrees", &observation.true_wind_from_degrees,
            1.2, true, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "apparent_wind_speed_mps", &observation.apparent_wind_speed_mps,
            0.0, 50.0, 0.8, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_angle(
            "apparent_wind_from_degrees", &observation.apparent_wind_from_degrees,
            0.8, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_angle(
            "heading_degrees", &observation.heading_degrees,
            1.2, true, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "speed_through_water_mps", &observation.speed_through_water_mps,
            0.0, 20.0, 1.1, true, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "speed_over_ground_mps", &observation.speed_over_ground_mps,
            0.0, 20.0, 0.8, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "current_speed_mps", &observation.current_speed_mps,
            0.0, 10.0, 1.0, true, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_angle(
            "current_toward_degrees", &observation.current_toward_degrees,
            1.0, true, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "heel_degrees", &observation.heel_degrees,
            -60.0, 60.0, 1.0, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "leeway_degrees", &observation.leeway_degrees,
            -60.0, 60.0, 0.9, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "rudder_angle_degrees", &observation.rudder_angle_degrees,
            -90.0, 90.0, 0.9, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "sheet_angle_degrees", &observation.sheet_angle_degrees,
            0.0, 180.0, 0.8, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "waypoint_range_m", &observation.waypoint_range_m,
            0.0, 100_000.0, 1.0, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_angle(
            "waypoint_bearing_degrees", &observation.waypoint_bearing_degrees,
            1.1, true, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "actuator_latency_ms", &observation.actuator_latency_ms,
            0.0, 5_000.0, 0.8, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;
        self.append_scalar(
            "significant_wave_height_m", &observation.significant_wave_height_m,
            0.0, 20.0, 0.7, false, &observation.common_timebase_id,
            &mut components, &mut weights, &mut states,
        )?;

        components.push(self.genesis.hv(
            &format!("sailcraft::timebase::{}", observation.common_timebase_id),
            HDC_DIMENSION,
        ));
        weights.push(0.25);

        let all_qualified = states.iter().all(|state| state.state == EvidenceState::Qualified);
        let critical_missing = states
            .iter()
            .any(|state| state.critical && state.state != EvidenceState::Qualified);
        let admission = if critical_missing {
            ObservationAdmission::CriticalEvidenceIncomplete
        } else if all_qualified {
            ObservationAdmission::Complete
        } else {
            ObservationAdmission::Degraded
        };

        let references: Vec<&ContinuousHV> = components.iter().collect();
        let sensor_hv = ContinuousHV::weighted_bundle(&references, &weights).normalize();
        Ok(EncodedSailObservation {
            sensor_hv,
            admission,
            channel_states: states,
            timebase_id: observation.common_timebase_id.clone(),
            configuration_digest: observation.configuration_digest,
        })
    }
}

fn state_label(state: EvidenceState) -> &'static str {
    match state {
        EvidenceState::Qualified => "qualified",
        EvidenceState::Missing => "missing",
        EvidenceState::Stale => "stale",
        EvidenceState::Unverified => "unverified",
        EvidenceState::Conflicted => "conflicted",
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TemporalError {
    InvalidNetworkConfiguration,
    InvalidTimestep,
    TimebaseChanged,
    InvalidInputDimension,
    NonFiniteLatentOutput,
}

impl std::fmt::Display for TemporalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TemporalError {}

#[derive(Clone, Debug)]
pub struct SailTemporalRepresentation {
    /// 16,384D latent state only; no action or actuator projection exists.
    pub latent_hv: ContinuousHV,
    pub admission: ObservationAdmission,
    pub channel_states: Vec<ChannelEvidenceState>,
    pub timebase_id: String,
    pub configuration_digest: [u8; 32],
}

pub struct SailcraftTemporalState {
    network: HdcLtcUnifiedNetwork,
    timebase_id: Option<String>,
}

impl SailcraftTemporalState {
    pub fn new(
        genesis: &GenesisSeed,
        layers: usize,
        neurons_per_layer: usize,
        learning_rate: f32,
    ) -> Result<Self, TemporalError> {
        if !(1..=8).contains(&layers)
            || !(1..=256).contains(&neurons_per_layer)
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || learning_rate > 1.0
        {
            return Err(TemporalError::InvalidNetworkConfiguration);
        }

        let neuron_config = UnifiedConfig {
            tau_base: 0.02,
            backbone_tau: 0.4,
            dimension: HDC_DIMENSION,
            learning_rate,
            ..UnifiedConfig::default()
        };
        let network_config = UnifiedNetworkConfig {
            layer_sizes: vec![neurons_per_layer; layers],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };
        Ok(Self {
            network: HdcLtcUnifiedNetwork::from_genesis(network_config, genesis),
            timebase_id: None,
        })
    }

    /// Evolve the temporal representation only. This does not select or issue
    /// a sailing action and does not imply a trained policy or physical skill.
    pub fn step(
        &mut self,
        observation: &EncodedSailObservation,
        dt_seconds: f32,
    ) -> Result<SailTemporalRepresentation, TemporalError> {
        if !dt_seconds.is_finite() || dt_seconds <= 0.0 || dt_seconds > 2.0 {
            return Err(TemporalError::InvalidTimestep);
        }
        if observation.sensor_hv.as_slice().len() != HDC_DIMENSION {
            return Err(TemporalError::InvalidInputDimension);
        }
        if let Some(previous) = &self.timebase_id {
            if previous != &observation.timebase_id {
                return Err(TemporalError::TimebaseChanged);
            }
        }

        self.network.evolve_closed_form(dt_seconds, &observation.sensor_hv);
        let latent_hv = self.network.output().normalize();
        if latent_hv.as_slice().len() != HDC_DIMENSION
            || latent_hv.as_slice().iter().any(|value| !value.is_finite())
        {
            return Err(TemporalError::NonFiniteLatentOutput);
        }
        self.timebase_id = Some(observation.timebase_id.clone());

        Ok(SailTemporalRepresentation {
            latent_hv,
            admission: observation.admission,
            channel_states: observation.channel_states.clone(),
            timebase_id: observation.timebase_id.clone(),
            configuration_digest: observation.configuration_digest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: [u8; 32] = [0x42; 32];
    const TIMEBASE: &str = "synthetic-common-timebase";

    fn qualified(value: f64) -> Evidence<f64> {
        Evidence::qualified_with_freshness(
            value,
            [0xA5; 32],
            TIMEBASE,
            0.1,
            1.0,
        )
    }

    fn observation() -> SailObservation {
        SailObservation {
            configuration_digest: CONFIG,
            common_timebase_id: TIMEBASE.into(),
            true_wind_speed_mps: qualified(8.0),
            true_wind_from_degrees: qualified(359.0),
            apparent_wind_speed_mps: qualified(10.0),
            apparent_wind_from_degrees: qualified(345.0),
            heading_degrees: qualified(15.0),
            speed_through_water_mps: qualified(3.0),
            speed_over_ground_mps: qualified(3.2),
            current_speed_mps: qualified(0.4),
            current_toward_degrees: qualified(90.0),
            heel_degrees: qualified(8.0),
            leeway_degrees: qualified(3.0),
            rudder_angle_degrees: qualified(-4.0),
            sheet_angle_degrees: qualified(35.0),
            waypoint_range_m: qualified(1_200.0),
            waypoint_bearing_degrees: qualified(42.0),
            actuator_latency_ms: qualified(85.0),
            significant_wave_height_m: qualified(0.7),
        }
    }

    fn encoder() -> SailcraftEncoder {
        let genesis = GenesisSeed::from_phrase("sailcraft-cognition-test-genesis");
        SailcraftEncoder::new(&genesis, CONFIG, DEFAULT_LEVELS).unwrap()
    }

    fn cosine(left: &ContinuousHV, right: &ContinuousHV) -> f64 {
        let a = left.as_slice();
        let b = right.as_slice();
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let norm_a: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        dot / (norm_a * norm_b)
    }

    #[test]
    fn complete_qualified_observation_is_admitted() {
        let encoded = encoder().encode(&observation()).unwrap();
        assert_eq!(encoded.admission, ObservationAdmission::Complete);
        assert_eq!(encoded.channel_states.len(), 17);
        assert_eq!(encoded.sensor_hv.as_slice().len(), HDC_DIMENSION);
    }

    #[test]
    fn circular_bearings_encode_zero_crossing_as_neighbors() {
        let encoder = encoder();
        let mut north_a = observation();
        let mut north_b = observation();
        let mut opposed = observation();
        north_a.true_wind_from_degrees = qualified(359.0);
        north_b.true_wind_from_degrees = qualified(1.0);
        opposed.true_wind_from_degrees = qualified(180.0);
        let a = encoder.encode(&north_a).unwrap();
        let b = encoder.encode(&north_b).unwrap();
        let c = encoder.encode(&opposed).unwrap();
        assert!(cosine(&a.sensor_hv, &b.sensor_hv) > cosine(&a.sensor_hv, &c.sensor_hv));
    }

    #[test]
    fn stale_numeric_payload_is_not_encoded() {
        let encoder = encoder();
        let mut stale_a = observation();
        let mut stale_b = observation();
        stale_a.true_wind_speed_mps = Evidence::stale(8.0);
        stale_b.true_wind_speed_mps = Evidence::stale(18.0);
        let a = encoder.encode(&stale_a).unwrap();
        let b = encoder.encode(&stale_b).unwrap();
        assert_eq!(a.sensor_hv.as_slice(), b.sensor_hv.as_slice());
        assert_eq!(a.admission, ObservationAdmission::CriticalEvidenceIncomplete);
    }

    #[test]
    fn evidence_states_remain_distinguishable() {
        let encoder = encoder();
        let mut missing = observation();
        let mut stale = observation();
        let mut unverified = observation();
        let mut conflicted = observation();
        missing.true_wind_speed_mps = Evidence::missing();
        stale.true_wind_speed_mps = Evidence::stale(9.0);
        unverified.true_wind_speed_mps = Evidence::unverified(9.0);
        conflicted.true_wind_speed_mps = Evidence::conflicted(9.0);
        let encoded = [
            encoder.encode(&missing).unwrap(),
            encoder.encode(&stale).unwrap(),
            encoder.encode(&unverified).unwrap(),
            encoder.encode(&conflicted).unwrap(),
        ];
        for i in 0..encoded.len() {
            for j in i + 1..encoded.len() {
                assert_ne!(encoded[i].sensor_hv.as_slice(), encoded[j].sensor_hv.as_slice());
            }
        }
    }

    #[test]
    fn invalid_qualified_digest_and_timebase_are_rejected() {
        let encoder = encoder();
        let mut bad_digest = observation();
        bad_digest.true_wind_speed_mps = Evidence::qualified_with_freshness(
            8.0, [0; 32], TIMEBASE, 0.1, 1.0,
        );
        assert!(matches!(
            encoder.encode(&bad_digest),
            Err(EncodeError::InvalidQualifiedMetadata { .. })
        ));

        let mut bad_clock = observation();
        bad_clock.true_wind_speed_mps = Evidence::qualified_with_freshness(
            8.0, [0xA5; 32], "unmapped-clock", 0.1, 1.0,
        );
        assert!(matches!(
            encoder.encode(&bad_clock),
            Err(EncodeError::InvalidQualifiedMetadata { .. })
        ));
    }

    #[test]
    fn stale_freshness_and_nonfinite_measurements_are_rejected() {
        let encoder = encoder();
        let mut too_old = observation();
        too_old.true_wind_speed_mps = Evidence::qualified_with_freshness(
            8.0, [0xA5; 32], TIMEBASE, 2.0, 1.0,
        );
        assert!(matches!(
            encoder.encode(&too_old),
            Err(EncodeError::InvalidQualifiedMetadata { .. })
        ));

        let mut nonfinite = observation();
        nonfinite.heel_degrees = qualified(f64::NAN);
        assert!(matches!(
            encoder.encode(&nonfinite),
            Err(EncodeError::InvalidQualifiedMetadata { .. })
        ));
    }

    #[test]
    fn out_of_domain_values_are_rejected_without_clamping() {
        let encoder = encoder();
        let mut invalid_wind = observation();
        invalid_wind.true_wind_speed_mps = qualified(41.0);
        assert!(matches!(
            encoder.encode(&invalid_wind),
            Err(EncodeError::OutOfDomain { .. })
        ));

        let mut invalid_bearing = observation();
        invalid_bearing.current_toward_degrees = qualified(360.0);
        assert!(matches!(
            encoder.encode(&invalid_bearing),
            Err(EncodeError::OutOfDomain { .. })
        ));
    }

    #[test]
    fn configuration_mismatch_is_rejected() {
        let encoder = encoder();
        let mut other = observation();
        other.configuration_digest = [0x33; 32];
        assert!(matches!(
            encoder.encode(&other),
            Err(EncodeError::ConfigurationMismatch)
        ));
    }

    #[test]
    fn deterministic_genesis_replays_identical_vectors() {
        let genesis_a = GenesisSeed::from_phrase("sailcraft-cognition-determinism");
        let genesis_b = GenesisSeed::from_phrase("sailcraft-cognition-determinism");
        let encoder_a = SailcraftEncoder::new(&genesis_a, CONFIG, DEFAULT_LEVELS).unwrap();
        let encoder_b = SailcraftEncoder::new(&genesis_b, CONFIG, DEFAULT_LEVELS).unwrap();
        let a = encoder_a.encode(&observation()).unwrap();
        let b = encoder_b.encode(&observation()).unwrap();
        assert_eq!(a.sensor_hv.as_slice(), b.sensor_hv.as_slice());
    }

    #[test]
    fn channel_receipt_preserves_source_identity_without_claiming_authenticity() {
        let encoded = encoder().encode(&observation()).unwrap();
        let wind = encoded.channel_states.iter()
            .find(|state| state.channel == "true_wind_speed_mps").unwrap();
        assert_eq!(wind.source_artifact_digest, [0xA5; 32]);
        assert_eq!(wind.source_timebase_id, TIMEBASE);
        assert_eq!(wind.age_seconds, 0.1);
        assert_eq!(wind.maximum_age_seconds, 1.0);
    }

    #[test]
    fn critical_missingness_blocks_policy_input_but_keeps_representation() {
        let mut partial = observation();
        partial.current_toward_degrees = Evidence::missing();
        let encoded = encoder().encode(&partial).unwrap();
        assert_eq!(encoded.admission, ObservationAdmission::CriticalEvidenceIncomplete);
        assert_eq!(encoded.sensor_hv.as_slice().len(), HDC_DIMENSION);
    }

    #[test]
    fn temporal_state_produces_finite_latent_without_action_output() {
        let genesis = GenesisSeed::from_phrase("sailcraft-temporal-state-test");
        let mut temporal = SailcraftTemporalState::new(&genesis, 3, 8, 0.001).unwrap();
        let encoded = encoder().encode(&observation()).unwrap();
        let latent = temporal.step(&encoded, 0.02).unwrap();
        assert_eq!(latent.latent_hv.as_slice().len(), HDC_DIMENSION);
        assert!(latent.latent_hv.as_slice().iter().all(|value| value.is_finite()));
        assert_eq!(latent.admission, ObservationAdmission::Complete);
    }

    #[test]
    fn temporal_state_rejects_timebase_discontinuity() {
        let genesis = GenesisSeed::from_phrase("sailcraft-temporal-timebase-test");
        let mut temporal = SailcraftTemporalState::new(&genesis, 2, 8, 0.001).unwrap();
        let encoded = encoder().encode(&observation()).unwrap();
        temporal.step(&encoded, 0.02).unwrap();
        let mut changed_clock = encoded.clone();
        changed_clock.timebase_id = "new-clock-domain".into();
        assert_eq!(
            temporal.step(&changed_clock, 0.02).unwrap_err(),
            TemporalError::TimebaseChanged
        );
    }

    #[test]
    fn invalid_timestep_does_not_advance_temporal_state() {
        let genesis = GenesisSeed::from_phrase("sailcraft-temporal-dt-test");
        let mut temporal = SailcraftTemporalState::new(&genesis, 2, 8, 0.001).unwrap();
        let encoded = encoder().encode(&observation()).unwrap();
        assert_eq!(temporal.step(&encoded, 0.0).unwrap_err(), TemporalError::InvalidTimestep);
        assert_eq!(temporal.step(&encoded, 3.0).unwrap_err(), TemporalError::InvalidTimestep);
    }
}
