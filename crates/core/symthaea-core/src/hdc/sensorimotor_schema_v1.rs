// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Universal role-bound sensorimotor semantics for HDC.
//!
//! This module defines a portable semantic address for physical observations.
//! It deliberately does not encode operator authority, qualification, execution
//! permits, or safety approval. HDC here is representation only.

use super::{BinaryHV, ContinuousHV, HDC_DIMENSION};
use serde::{Deserialize, Serialize};

const SENSORIMOTOR_ADDRESS_DOMAIN_V1: &[u8] = b"symthaea.sensorimotor.address.v1\0";
const SENSORIMOTOR_VALUE_DOMAIN_V1: &[u8] = b"symthaea.sensorimotor.value.v1\0";

/// Physical subject addressed by one sensorimotor fact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorimotorSubjectV1 {
    BodyRoot,
    Joint(String),
    Actuator(String),
    EndEffector(String),
    ContactSite(String),
    Sensor(String),
    Object(String),
    Custom(String),
}

/// Reference frame in which a physical quantity is expressed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorimotorFrameV1 {
    World,
    Body,
    ParentJoint,
    Joint(String),
    Tool(String),
    Sensor(String),
    Map(String),
    Custom(String),
}

/// Physical quantity represented by one observation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorimotorQuantityV1 {
    Position,
    Orientation,
    LinearVelocity,
    AngularVelocity,
    LinearAcceleration,
    AngularAcceleration,
    JointPosition,
    JointVelocity,
    Force,
    Moment,
    Torque,
    ContactState,
    Height,
    Energy,
    StateOfCharge,
    Temperature,
    Pressure,
    Custom(String),
}

/// Component within a scalar/vector/quaternion-valued quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorimotorComponentV1 {
    Scalar,
    X,
    Y,
    Z,
    W,
}

/// Unit contract carried by the semantic address.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorimotorUnitV1 {
    Meter,
    Radian,
    MeterPerSecond,
    RadianPerSecond,
    MeterPerSecondSquared,
    RadianPerSecondSquared,
    Newton,
    NewtonMeter,
    Joule,
    Ratio,
    Kelvin,
    Pascal,
    Unitless,
    Boolean,
    Custom(String),
}

/// Why an observation is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingObservationReasonV1 {
    NotPresent,
    NotObserved,
    Stale,
    Invalid,
    Disabled,
    Unknown,
}

/// Numeric quantization contract for one physical role.
///
/// `min`/`max` are expressed in `unit`; values outside the range saturate at
/// encoding time rather than silently changing the semantic range contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorimotorValueContractV1 {
    pub unit: SensorimotorUnitV1,
    pub min: f64,
    pub max: f64,
    pub bins: u16,
}

impl SensorimotorValueContractV1 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.min.is_finite() || !self.max.is_finite() {
            return Err("sensorimotor value range must be finite");
        }
        if self.max <= self.min {
            return Err("sensorimotor value range requires max > min");
        }
        if !(2..=4096).contains(&self.bins) {
            return Err("sensorimotor value bins must be in 2..=4096");
        }
        Ok(())
    }
}

/// Stable semantic address for one physical quantity.
///
/// The address intentionally contains no embodiment/platform ID for common
/// physical roles. `BodyRoot + AngularVelocity + Body + X`, for example, can
/// therefore be shared by a humanoid, multirotor, quadruped, or vehicle when
/// their unit/range contracts agree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorimotorAddressV1 {
    pub schema_id: String,
    pub subject: SensorimotorSubjectV1,
    pub quantity: SensorimotorQuantityV1,
    pub frame: SensorimotorFrameV1,
    pub component: SensorimotorComponentV1,
    pub value_contract: SensorimotorValueContractV1,
}

impl SensorimotorAddressV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.sensorimotor.address.v1";

    pub fn new(
        subject: SensorimotorSubjectV1,
        quantity: SensorimotorQuantityV1,
        frame: SensorimotorFrameV1,
        component: SensorimotorComponentV1,
        value_contract: SensorimotorValueContractV1,
    ) -> Self {
        Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            subject,
            quantity,
            frame,
            component,
            value_contract,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err("unsupported sensorimotor address schema");
        }
        self.value_contract.validate()?;
        if !unit_matches_quantity(&self.quantity, &self.value_contract.unit) {
            return Err("sensorimotor unit is incompatible with quantity");
        }
        Ok(())
    }

    /// Exact 256-bit semantic identity of this physical role and value contract.
    pub fn semantic_digest(&self) -> Result<[u8; 32], &'static str> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SENSORIMOTOR_ADDRESS_DOMAIN_V1);
        feed_token(&mut hasher, self.schema_id.as_bytes());
        feed_subject(&mut hasher, &self.subject);
        feed_quantity(&mut hasher, &self.quantity);
        feed_frame(&mut hasher, &self.frame);
        feed_component(&mut hasher, self.component);
        feed_unit(&mut hasher, &self.value_contract.unit);
        hasher.update(&self.value_contract.min.to_bits().to_le_bytes());
        hasher.update(&self.value_contract.max.to_bits().to_le_bytes());
        hasher.update(&self.value_contract.bins.to_le_bytes());
        Ok(*hasher.finalize().as_bytes())
    }
}

/// A measured value at a semantic sensorimotor address.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorimotorMeasurementV1 {
    pub address: SensorimotorAddressV1,
    pub value: f64,
}

impl SensorimotorMeasurementV1 {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.address.validate()?;
        if !self.value.is_finite() {
            return Err("sensorimotor measurement must be finite");
        }
        Ok(())
    }
}

/// Observation state. Missing is deliberately distinct from measured zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SensorimotorObservationV1 {
    Measured(SensorimotorMeasurementV1),
    Missing {
        address: SensorimotorAddressV1,
        reason: MissingObservationReasonV1,
    },
}

impl SensorimotorObservationV1 {
    pub fn address(&self) -> &SensorimotorAddressV1 {
        match self {
            Self::Measured(measurement) => &measurement.address,
            Self::Missing { address, .. } => address,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Measured(measurement) => measurement.validate(),
            Self::Missing { address, .. } => address.validate(),
        }
    }
}

/// Versioned identity of the HDC representation geometry used for a physical
/// sensorimotor fact. This is deliberately separate from the physical address
/// digest so an optimized or future encoding kernel cannot silently change what
/// the fact *means*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorimotorHdcEncodingProfileV1 {
    ThermometerPrefixBinaryV1,
}

impl SensorimotorHdcEncodingProfileV1 {
    pub const fn schema_id(self) -> &'static str {
        match self {
            Self::ThermometerPrefixBinaryV1 => {
                "symthaea.sensorimotor.hdc.thermometer-prefix-binary.v1"
            }
        }
    }
}

/// Deterministic role/filler encoder for the v1 sensorimotor schema.
#[derive(Debug, Clone, Copy, Default)]
pub struct SensorimotorHdcEncoderV1;

impl SensorimotorHdcEncoderV1 {
    pub const fn profile(&self) -> SensorimotorHdcEncodingProfileV1 {
        SensorimotorHdcEncodingProfileV1::ThermometerPrefixBinaryV1
    }

    /// Encode one observation. Missing observations intentionally produce no
    /// measurement vector rather than being collapsed to numeric zero.
    pub fn encode_observation(
        &self,
        observation: &SensorimotorObservationV1,
    ) -> Result<Option<ContinuousHV>, &'static str> {
        observation.validate()?;
        match observation {
            SensorimotorObservationV1::Measured(measurement) => {
                Ok(Some(self.encode_measurement(measurement)?))
            }
            SensorimotorObservationV1::Missing { .. } => Ok(None),
        }
    }

    /// Encode a measured physical fact as ROLE(address) ⊗ VALUE(level).
    ///
    /// VALUE uses deterministic thermometer-style bundling so nearby bins retain
    /// more representational similarity than distant bins.
    pub fn encode_measurement(
        &self,
        measurement: &SensorimotorMeasurementV1,
    ) -> Result<ContinuousHV, &'static str> {
        measurement.validate()?;
        let address_digest = measurement.address.semantic_digest()?;
        let role = sensorimotor_role_hv_v1(&address_digest);
        let bin = sensorimotor_value_bin_v1(&measurement.address.value_contract, measurement.value);

        let levels: Vec<BinaryHV> = (0..=bin)
            .map(|level| sensorimotor_value_level_hv_v1(&address_digest, level))
            .collect();
        let value = BinaryHV::bundle(&levels);
        Ok(role.bind(&value).to_continuous())
    }

    /// Bundle all measured observations. Order does not carry semantics.
    /// Returns `None` when every observation is missing.
    pub fn encode_observations(
        &self,
        observations: &[SensorimotorObservationV1],
    ) -> Result<Option<ContinuousHV>, &'static str> {
        let mut encoded = Vec::new();
        for observation in observations {
            if let Some(hv) = self.encode_observation(observation)? {
                encoded.push(hv);
            }
        }
        if encoded.is_empty() {
            return Ok(None);
        }
        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        Ok(Some(ContinuousHV::bundle(&refs)))
    }
}

pub(crate) fn sensorimotor_role_hv_v1(address_digest: &[u8; 32]) -> BinaryHV {
    BinaryHV::random(sensorimotor_seed64_v1(address_digest))
}

pub(crate) fn sensorimotor_value_level_hv_v1(
    address_digest: &[u8; 32],
    level: u16,
) -> BinaryHV {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SENSORIMOTOR_VALUE_DOMAIN_V1);
    hasher.update(address_digest);
    hasher.update(&level.to_le_bytes());
    BinaryHV::random(sensorimotor_seed64_v1(hasher.finalize().as_bytes()))
}

pub(crate) fn sensorimotor_value_bin_v1(
    contract: &SensorimotorValueContractV1,
    value: f64,
) -> u16 {
    let clamped = value.clamp(contract.min, contract.max);
    let normalized = (clamped - contract.min) / (contract.max - contract.min);
    ((normalized * f64::from(contract.bins - 1)).round() as u16).min(contract.bins - 1)
}

fn sensorimotor_seed64_v1(digest: &[u8; 32]) -> u64 {
    u64::from_le_bytes(digest[..8].try_into().expect("8-byte digest prefix"))
}

fn feed_token(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn feed_subject(hasher: &mut blake3::Hasher, subject: &SensorimotorSubjectV1) {
    match subject {
        SensorimotorSubjectV1::BodyRoot => feed_token(hasher, b"body_root"),
        SensorimotorSubjectV1::Joint(v) => feed_tagged(hasher, b"joint", v),
        SensorimotorSubjectV1::Actuator(v) => feed_tagged(hasher, b"actuator", v),
        SensorimotorSubjectV1::EndEffector(v) => feed_tagged(hasher, b"end_effector", v),
        SensorimotorSubjectV1::ContactSite(v) => feed_tagged(hasher, b"contact_site", v),
        SensorimotorSubjectV1::Sensor(v) => feed_tagged(hasher, b"sensor", v),
        SensorimotorSubjectV1::Object(v) => feed_tagged(hasher, b"object", v),
        SensorimotorSubjectV1::Custom(v) => feed_tagged(hasher, b"custom", v),
    }
}

fn feed_quantity(hasher: &mut blake3::Hasher, quantity: &SensorimotorQuantityV1) {
    match quantity {
        SensorimotorQuantityV1::Position => feed_token(hasher, b"position"),
        SensorimotorQuantityV1::Orientation => feed_token(hasher, b"orientation"),
        SensorimotorQuantityV1::LinearVelocity => feed_token(hasher, b"linear_velocity"),
        SensorimotorQuantityV1::AngularVelocity => feed_token(hasher, b"angular_velocity"),
        SensorimotorQuantityV1::LinearAcceleration => feed_token(hasher, b"linear_acceleration"),
        SensorimotorQuantityV1::AngularAcceleration => feed_token(hasher, b"angular_acceleration"),
        SensorimotorQuantityV1::JointPosition => feed_token(hasher, b"joint_position"),
        SensorimotorQuantityV1::JointVelocity => feed_token(hasher, b"joint_velocity"),
        SensorimotorQuantityV1::Force => feed_token(hasher, b"force"),
        SensorimotorQuantityV1::Moment => feed_token(hasher, b"moment"),
        SensorimotorQuantityV1::Torque => feed_token(hasher, b"torque"),
        SensorimotorQuantityV1::ContactState => feed_token(hasher, b"contact_state"),
        SensorimotorQuantityV1::Height => feed_token(hasher, b"height"),
        SensorimotorQuantityV1::Energy => feed_token(hasher, b"energy"),
        SensorimotorQuantityV1::StateOfCharge => feed_token(hasher, b"state_of_charge"),
        SensorimotorQuantityV1::Temperature => feed_token(hasher, b"temperature"),
        SensorimotorQuantityV1::Pressure => feed_token(hasher, b"pressure"),
        SensorimotorQuantityV1::Custom(v) => feed_tagged(hasher, b"custom_quantity", v),
    }
}

fn feed_frame(hasher: &mut blake3::Hasher, frame: &SensorimotorFrameV1) {
    match frame {
        SensorimotorFrameV1::World => feed_token(hasher, b"world"),
        SensorimotorFrameV1::Body => feed_token(hasher, b"body"),
        SensorimotorFrameV1::ParentJoint => feed_token(hasher, b"parent_joint"),
        SensorimotorFrameV1::Joint(v) => feed_tagged(hasher, b"joint_frame", v),
        SensorimotorFrameV1::Tool(v) => feed_tagged(hasher, b"tool_frame", v),
        SensorimotorFrameV1::Sensor(v) => feed_tagged(hasher, b"sensor_frame", v),
        SensorimotorFrameV1::Map(v) => feed_tagged(hasher, b"map_frame", v),
        SensorimotorFrameV1::Custom(v) => feed_tagged(hasher, b"custom_frame", v),
    }
}

fn feed_component(hasher: &mut blake3::Hasher, component: SensorimotorComponentV1) {
    feed_token(
        hasher,
        match component {
            SensorimotorComponentV1::Scalar => b"scalar",
            SensorimotorComponentV1::X => b"x",
            SensorimotorComponentV1::Y => b"y",
            SensorimotorComponentV1::Z => b"z",
            SensorimotorComponentV1::W => b"w",
        },
    );
}

fn feed_unit(hasher: &mut blake3::Hasher, unit: &SensorimotorUnitV1) {
    match unit {
        SensorimotorUnitV1::Meter => feed_token(hasher, b"meter"),
        SensorimotorUnitV1::Radian => feed_token(hasher, b"radian"),
        SensorimotorUnitV1::MeterPerSecond => feed_token(hasher, b"meter_per_second"),
        SensorimotorUnitV1::RadianPerSecond => feed_token(hasher, b"radian_per_second"),
        SensorimotorUnitV1::MeterPerSecondSquared => {
            feed_token(hasher, b"meter_per_second_squared")
        }
        SensorimotorUnitV1::RadianPerSecondSquared => {
            feed_token(hasher, b"radian_per_second_squared")
        }
        SensorimotorUnitV1::Newton => feed_token(hasher, b"newton"),
        SensorimotorUnitV1::NewtonMeter => feed_token(hasher, b"newton_meter"),
        SensorimotorUnitV1::Joule => feed_token(hasher, b"joule"),
        SensorimotorUnitV1::Ratio => feed_token(hasher, b"ratio"),
        SensorimotorUnitV1::Kelvin => feed_token(hasher, b"kelvin"),
        SensorimotorUnitV1::Pascal => feed_token(hasher, b"pascal"),
        SensorimotorUnitV1::Unitless => feed_token(hasher, b"unitless"),
        SensorimotorUnitV1::Boolean => feed_token(hasher, b"boolean"),
        SensorimotorUnitV1::Custom(v) => feed_tagged(hasher, b"custom_unit", v),
    }
}

fn feed_tagged(hasher: &mut blake3::Hasher, tag: &[u8], value: &str) {
    feed_token(hasher, tag);
    feed_token(hasher, value.as_bytes());
}

fn unit_matches_quantity(quantity: &SensorimotorQuantityV1, unit: &SensorimotorUnitV1) -> bool {
    use SensorimotorQuantityV1 as Q;
    use SensorimotorUnitV1 as U;
    match quantity {
        Q::Position | Q::Height => matches!(unit, U::Meter),
        Q::Orientation => matches!(unit, U::Radian | U::Unitless),
        Q::LinearVelocity => matches!(unit, U::MeterPerSecond),
        Q::AngularVelocity | Q::JointVelocity => matches!(unit, U::RadianPerSecond),
        Q::LinearAcceleration => matches!(unit, U::MeterPerSecondSquared),
        Q::AngularAcceleration => matches!(unit, U::RadianPerSecondSquared),
        Q::JointPosition => matches!(unit, U::Radian),
        Q::Force => matches!(unit, U::Newton),
        Q::Moment | Q::Torque => matches!(unit, U::NewtonMeter),
        Q::ContactState => matches!(unit, U::Boolean),
        Q::Energy => matches!(unit, U::Joule),
        Q::StateOfCharge => matches!(unit, U::Ratio),
        Q::Temperature => matches!(unit, U::Kelvin),
        Q::Pressure => matches!(unit, U::Pascal),
        Q::Custom(_) => true,
    }
}

#[path = "sensorimotor_codebook_v1.rs"]
mod codebook_v1;
pub use codebook_v1::*;

#[cfg(test)]
mod sensorimotor_schema_tests {
    use super::*;

    fn body_ang_vel_x(frame: SensorimotorFrameV1) -> SensorimotorAddressV1 {
        SensorimotorAddressV1::new(
            SensorimotorSubjectV1::BodyRoot,
            SensorimotorQuantityV1::AngularVelocity,
            frame,
            SensorimotorComponentV1::X,
            SensorimotorValueContractV1 {
                unit: SensorimotorUnitV1::RadianPerSecond,
                min: -20.0,
                max: 20.0,
                bins: 401,
            },
        )
    }

    fn measured(value: f64) -> SensorimotorObservationV1 {
        SensorimotorObservationV1::Measured(SensorimotorMeasurementV1 {
            address: body_ang_vel_x(SensorimotorFrameV1::Body),
            value,
        })
    }

    #[test]
    fn same_fact_encoding_is_deterministic() {
        let encoder = SensorimotorHdcEncoderV1;
        let a = encoder.encode_observation(&measured(0.75)).unwrap().unwrap();
        let b = encoder.encode_observation(&measured(0.75)).unwrap().unwrap();
        assert_eq!(a.values, b.values);
        assert_eq!(a.values.len(), HDC_DIMENSION);
    }

    #[test]
    fn observation_order_does_not_define_semantics() {
        let encoder = SensorimotorHdcEncoderV1;
        let a = measured(0.5);
        let b = measured(-0.5);
        let ab = encoder.encode_observations(&[a.clone(), b.clone()]).unwrap().unwrap();
        let ba = encoder.encode_observations(&[b, a]).unwrap().unwrap();
        assert_eq!(ab.values, ba.values);
    }

    #[test]
    fn frame_identity_is_explicit() {
        let body = body_ang_vel_x(SensorimotorFrameV1::Body);
        let world = body_ang_vel_x(SensorimotorFrameV1::World);
        assert_ne!(body.semantic_digest().unwrap(), world.semantic_digest().unwrap());
    }

    #[test]
    fn missing_is_not_measured_zero() {
        let encoder = SensorimotorHdcEncoderV1;
        let address = body_ang_vel_x(SensorimotorFrameV1::Body);
        let missing = SensorimotorObservationV1::Missing {
            address,
            reason: MissingObservationReasonV1::NotObserved,
        };
        assert!(encoder.encode_observation(&missing).unwrap().is_none());
        assert!(encoder.encode_observation(&measured(0.0)).unwrap().is_some());
    }

    #[test]
    fn unit_and_range_participate_in_schema_identity() {
        let base = body_ang_vel_x(SensorimotorFrameV1::Body);
        let mut changed_range = base.clone();
        changed_range.value_contract.max = 30.0;
        assert_ne!(base.semantic_digest().unwrap(), changed_range.semantic_digest().unwrap());

        let mut invalid_unit = base;
        invalid_unit.value_contract.unit = SensorimotorUnitV1::MeterPerSecond;
        assert!(invalid_unit.validate().is_err());
    }

    #[test]
    fn shared_body_role_is_embodiment_independent() {
        // Two adapters constructing the same typed physical role obtain the
        // same exact semantic identity; no platform/channel number participates.
        let humanoid = body_ang_vel_x(SensorimotorFrameV1::Body);
        let multirotor = body_ang_vel_x(SensorimotorFrameV1::Body);
        assert_eq!(humanoid.semantic_digest().unwrap(), multirotor.semantic_digest().unwrap());
    }

    #[test]
    fn serialization_round_trip_preserves_schema_identity() {
        let address = body_ang_vel_x(SensorimotorFrameV1::Body);
        let before = address.semantic_digest().unwrap();
        let wire = bincode::serialize(&address).unwrap();
        let restored: SensorimotorAddressV1 = bincode::deserialize(&wire).unwrap();
        assert_eq!(before, restored.semantic_digest().unwrap());
    }
}
