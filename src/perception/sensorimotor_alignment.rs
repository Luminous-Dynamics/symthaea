// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-embodiment sensorimotor alignment measurements.
//!
//! This module is compiled only when both `sensor-fusion` and `humanoid` are
//! enabled. It measures claims that can be established directly from the two
//! independent adapters; it does not infer policy transfer, safety, authority,
//! or calibrated uncertainty from HDC similarity.

use super::sensor_fusion::{ImuReading, SemanticImuFrameV1};
use symthaea_core::hdc::sensorimotor_contingencies::{
    SensorimotorComponentV1, SensorimotorHdcEncoderV1, SensorimotorObservationV1,
    SensorimotorQuantityV1, SensorimotorSubjectV1,
};
use symthaea_humanoid::{HumanoidMorphology, HumanoidState, SemanticHumanoidEncoderV1};

/// Compatibility view retained for the original R3.4 report contract.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuHumanoidAlignmentReportV1 {
    pub schema_id: String,
    /// Number of physically shared roles considered. V1 is angular velocity XYZ.
    pub shared_role_count: usize,
    /// Roles whose exact R3.1 observation-contract digests match.
    pub exact_semantic_identity_matches: usize,
    /// Same-value roles whose encoded `ContinuousHV` values are exactly equal.
    pub equal_value_hdc_matches: usize,
}

impl ImuHumanoidAlignmentReportV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.sensorimotor.imu-humanoid-alignment.v1";

    pub fn exact_semantic_alignment(&self) -> bool {
        self.shared_role_count > 0
            && self.exact_semantic_identity_matches == self.shared_role_count
    }

    pub fn exact_equal_value_representation_alignment(&self) -> bool {
        self.shared_role_count > 0 && self.equal_value_hdc_matches == self.shared_role_count
    }
}

/// R3.7 measurement view that keeps physical-role, exact-contract, and current
/// V1 representation claims separate.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuHumanoidAlignmentReportV2 {
    pub schema_id: String,
    /// Number of candidate roles compared. V2 currently compares angular velocity XYZ.
    pub shared_role_count: usize,
    /// Roles with equal `physical_role_digest()` values.
    pub physical_role_matches: usize,
    /// Roles with equal exact R3.1/R3.6 contract digests.
    pub exact_contract_matches: usize,
    /// Same-value roles whose current V1 HDC vectors are bit-for-bit equal.
    pub equal_value_hdc_matches: usize,
}

impl ImuHumanoidAlignmentReportV2 {
    pub const SCHEMA_ID: &'static str = "symthaea.sensorimotor.imu-humanoid-alignment.v2";

    pub fn physical_role_alignment(&self) -> bool {
        self.shared_role_count > 0 && self.physical_role_matches == self.shared_role_count
    }

    pub fn exact_contract_alignment(&self) -> bool {
        self.shared_role_count > 0 && self.exact_contract_matches == self.shared_role_count
    }

    pub fn exact_equal_value_representation_alignment(&self) -> bool {
        self.shared_role_count > 0 && self.equal_value_hdc_matches == self.shared_role_count
    }
}

/// Compare independently constructed IMU and humanoid semantic observations for
/// the physical intersection they currently share: body-frame angular velocity.
///
/// The humanoid state receives the gyroscope values exactly (`f32 -> f64`) so
/// equal-value representation can be tested without numerical-conversion noise.
/// Acceleration is intentionally excluded because `HumanoidState` does not carry
/// body-root linear acceleration; root linear velocity is a different quantity.
pub fn measure_imu_humanoid_angular_velocity_alignment_v2(
    gyro_rad_s: [f32; 3],
) -> Result<ImuHumanoidAlignmentReportV2, String> {
    let imu = SemanticImuFrameV1::from_reading(&ImuReading {
        accel: [0.0; 3],
        gyro: gyro_rad_s,
        timestamp_us: 0,
    });

    let morphology = HumanoidMorphology::Dmc21;
    let mut humanoid_state = HumanoidState::default_for(morphology);
    humanoid_state.root_angular_velocity = gyro_rad_s.map(f64::from);
    let humanoid = SemanticHumanoidEncoderV1::new()
        .frame(&humanoid_state, morphology)
        .map_err(|error| error.to_string())?;

    let hdc = SensorimotorHdcEncoderV1;
    let mut physical_role_matches = 0;
    let mut exact_contract_matches = 0;
    let mut equal_value_hdc_matches = 0;

    for component in [
        SensorimotorComponentV1::X,
        SensorimotorComponentV1::Y,
        SensorimotorComponentV1::Z,
    ] {
        let imu_observation = find_body_angular_velocity(&imu.observations, component)
            .ok_or_else(|| format!("IMU missing {component:?} angular-velocity role"))?;
        let humanoid_observation =
            find_body_angular_velocity(&humanoid.policy_observations, component)
                .ok_or_else(|| format!("humanoid missing {component:?} angular-velocity role"))?;

        if imu_observation
            .address()
            .same_physical_role(humanoid_observation.address())
            .map_err(str::to_string)?
        {
            physical_role_matches += 1;
        }

        let imu_digest = imu_observation
            .address()
            .contract_digest()
            .map_err(str::to_string)?;
        let humanoid_digest = humanoid_observation
            .address()
            .contract_digest()
            .map_err(str::to_string)?;
        if imu_digest == humanoid_digest {
            exact_contract_matches += 1;
        }

        let imu_hv = hdc
            .encode_observation(imu_observation)
            .map_err(str::to_string)?
            .ok_or_else(|| format!("IMU {component:?} observation is missing"))?;
        let humanoid_hv = hdc
            .encode_observation(humanoid_observation)
            .map_err(str::to_string)?
            .ok_or_else(|| format!("humanoid {component:?} observation is missing"))?;
        if imu_hv.values == humanoid_hv.values {
            equal_value_hdc_matches += 1;
        }
    }

    Ok(ImuHumanoidAlignmentReportV2 {
        schema_id: ImuHumanoidAlignmentReportV2::SCHEMA_ID.to_string(),
        shared_role_count: 3,
        physical_role_matches,
        exact_contract_matches,
        equal_value_hdc_matches,
    })
}

/// Compatibility wrapper preserving the original R3.4 report shape.
pub fn measure_imu_humanoid_angular_velocity_alignment(
    gyro_rad_s: [f32; 3],
) -> Result<ImuHumanoidAlignmentReportV1, String> {
    let v2 = measure_imu_humanoid_angular_velocity_alignment_v2(gyro_rad_s)?;
    Ok(ImuHumanoidAlignmentReportV1 {
        schema_id: ImuHumanoidAlignmentReportV1::SCHEMA_ID.to_string(),
        shared_role_count: v2.shared_role_count,
        exact_semantic_identity_matches: v2.exact_contract_matches,
        equal_value_hdc_matches: v2.equal_value_hdc_matches,
    })
}

fn find_body_angular_velocity<'a>(
    observations: &'a [SensorimotorObservationV1],
    component: SensorimotorComponentV1,
) -> Option<&'a SensorimotorObservationV1> {
    observations.iter().find(|observation| {
        let address = observation.address();
        matches!(&address.subject, SensorimotorSubjectV1::BodyRoot)
            && matches!(&address.quantity, SensorimotorQuantityV1::AngularVelocity)
            && address.component == component
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::sensorimotor_contingencies::{
        SensorimotorContractRelationV1, SensorimotorMeasurementV1, SensorimotorRangeRelationV1,
    };

    #[test]
    fn actual_imu_and_humanoid_adapters_share_three_exact_angular_velocity_roles() {
        let report = measure_imu_humanoid_angular_velocity_alignment([0.25, -1.0, 2.5]).unwrap();
        assert_eq!(report.schema_id, ImuHumanoidAlignmentReportV1::SCHEMA_ID);
        assert_eq!(report.shared_role_count, 3);
        assert_eq!(report.exact_semantic_identity_matches, 3);
        assert!(report.exact_semantic_alignment());
    }

    #[test]
    fn v2_separates_physical_contract_and_representation_alignment() {
        let report =
            measure_imu_humanoid_angular_velocity_alignment_v2([0.25, -1.0, 2.5]).unwrap();
        assert_eq!(report.schema_id, ImuHumanoidAlignmentReportV2::SCHEMA_ID);
        assert_eq!(report.shared_role_count, 3);
        assert_eq!(report.physical_role_matches, 3);
        assert_eq!(report.exact_contract_matches, 3);
        assert_eq!(report.equal_value_hdc_matches, 3);
        assert!(report.physical_role_alignment());
        assert!(report.exact_contract_alignment());
        assert!(report.exact_equal_value_representation_alignment());
    }

    #[test]
    fn equal_physical_values_encode_identically_across_actual_adapters() {
        let report = measure_imu_humanoid_angular_velocity_alignment([0.0, 0.1, -3.2]).unwrap();
        assert_eq!(report.equal_value_hdc_matches, 3);
        assert!(report.exact_equal_value_representation_alignment());
    }

    #[test]
    fn widened_sensor_contract_keeps_role_but_not_exact_contract_or_v1_hdc() {
        let imu = SemanticImuFrameV1::from_reading(&ImuReading {
            accel: [0.0; 3],
            gyro: [0.25, 0.0, 0.0],
            timestamp_us: 0,
        });
        let original = find_body_angular_velocity(
            &imu.observations,
            SensorimotorComponentV1::X,
        )
        .unwrap();
        let SensorimotorObservationV1::Measured(original_measurement) = original else {
            panic!("finite IMU gyro must be measured");
        };

        let mut widened_measurement = original_measurement.clone();
        widened_measurement.address.value_contract.min = -40.0;
        widened_measurement.address.value_contract.max = 40.0;
        widened_measurement.address.value_contract.bins = 801;
        let widened = SensorimotorObservationV1::Measured(SensorimotorMeasurementV1 {
            address: widened_measurement.address.clone(),
            value: widened_measurement.value,
        });

        assert!(
            original
                .address()
                .same_physical_role(widened.address())
                .unwrap()
        );
        assert_ne!(
            original.address().contract_digest().unwrap(),
            widened.address().contract_digest().unwrap()
        );
        assert_eq!(
            original.address().contract_relation(widened.address()).unwrap(),
            SensorimotorContractRelationV1::SamePhysicalRole {
                range_relation: SensorimotorRangeRelationV1::ContainedByOther,
                same_bins: false,
            }
        );

        let hdc = SensorimotorHdcEncoderV1;
        let original_hv = hdc.encode_observation(original).unwrap().unwrap();
        let widened_hv = hdc.encode_observation(&widened).unwrap().unwrap();
        assert_ne!(original_hv.values, widened_hv.values);
    }

    #[test]
    fn alignment_measurement_does_not_conflate_acceleration_with_velocity() {
        let report = measure_imu_humanoid_angular_velocity_alignment_v2([1.0, 2.0, 3.0]).unwrap();
        // IMU has six observations, but only angular velocity XYZ is physically
        // shared with HumanoidState. Linear acceleration != linear velocity.
        assert_eq!(report.shared_role_count, 3);
        assert_eq!(report.physical_role_matches, 3);
    }

    #[test]
    fn non_finite_shared_value_fails_measurement_instead_of_becoming_zero() {
        let error = measure_imu_humanoid_angular_velocity_alignment_v2([f32::NAN, 0.0, 0.0])
            .expect_err("missing IMU observation must not be treated as a measured zero");
        assert!(error.contains("observation is missing"));
    }
}
