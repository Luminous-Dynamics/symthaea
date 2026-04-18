// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sensor Fusion — real-sensor → ContinuousHV
//!
//! Bridges physical (or simulated) sensor readings into the cognitive
//! loop's perception phase. The `SensorFusion` trait takes a reading and
//! returns a 16,384-dim `ContinuousHV` suitable for bundling into the
//! perception encoding alongside STT audio and radio spectrum (matches
//! the Phase 2 additive-sensor-blend pattern).
//!
//! ## S1 scope (this module)
//!
//! - `ImuReading` — 6-axis accelerometer + gyroscope + timestamp
//! - `SensorFusion` trait
//! - `RoleFillerImuFusion` — default deterministic fusion using the same
//!   role-filler HDC pattern as `SpectrumManager::perception_hv` (3ac2af34c6).
//!   Identical readings produce identical HVs; different bucket values
//!   produce HDC-orthogonal HVs.
//!
//! ## Later arcs
//!
//! - S2: `CameraReading` + patch-based fusion via `symthaea-vision-manifold`
//! - S3: `GpsReading` + dead-reckoning cross-fusion
//! - S4: PX4 MAVLink adapter feeding `ImuReading`s from real autopilots

#![cfg(feature = "sensor-fusion")]

use symthaea_core::hdc::{BinaryHV, ContinuousHV};

/// 6-axis inertial measurement unit reading.
///
/// Body-frame accelerations and angular velocities, plus a monotonic
/// capture timestamp so fusion modules can detect stale data.
#[derive(Debug, Clone, Copy)]
pub struct ImuReading {
    /// Linear acceleration in body frame (m/s²). Includes gravity.
    pub accel: [f32; 3],
    /// Angular velocity in body frame (rad/s).
    pub gyro: [f32; 3],
    /// Capture timestamp, microseconds since epoch.
    pub timestamp_us: u64,
}

impl ImuReading {
    /// A zeroed reading at t=0. Useful as a perception-phase default when
    /// no IMU is attached yet (fusion returns None rather than a fake HV).
    pub const fn zero() -> Self {
        Self {
            accel: [0.0; 3],
            gyro: [0.0; 3],
            timestamp_us: 0,
        }
    }
}

/// Converts sensor readings into a ContinuousHV for the perception phase.
///
/// Impls should be deterministic (same reading → same HV) and cheap
/// (perception phase runs every ~5 ms at 200 Hz). Send + Sync required
/// because the cognitive loop service must remain Sync for its
/// MetricsProvider bound (see `feedback_sync_bound_mpsc.md`).
pub trait ImuFusion: Send + Sync {
    /// Produce a 16,384-dim ContinuousHV from an IMU reading.
    fn fuse(&self, reading: &ImuReading) -> ContinuousHV;
}

// ─── Role seeds for IMU role-filler binding ─────────────────────────
// Stable u64s identify each IMU dimension. Role XOR Value gives a bound
// pair; bundling six pairs (3 accel axes + 3 gyro axes) yields the IMU
// perception HV. Seeds share the high-bit prefix with the radio role
// seeds (0xF0E0...) so the two subsystems never collide semantically.
const IMU_SEED_ACCEL_X: u64 = 0xF0E0_FEDC_BA98_0001;
const IMU_SEED_ACCEL_Y: u64 = 0xF0E0_FEDC_BA98_0002;
const IMU_SEED_ACCEL_Z: u64 = 0xF0E0_FEDC_BA98_0003;
const IMU_SEED_GYRO_X: u64 = 0xF0E0_FEDC_BA98_0004;
const IMU_SEED_GYRO_Y: u64 = 0xF0E0_FEDC_BA98_0005;
const IMU_SEED_GYRO_Z: u64 = 0xF0E0_FEDC_BA98_0006;

/// Quantize one accelerometer axis into a bucket ID.
///
/// Range ±40 m/s² (covers 4g linear + 1g gravity offset) with 0.5 m/s²
/// resolution → 161 buckets centered on 0. Values outside the range
/// saturate rather than wrap, so extreme readings still produce a stable
/// (if boundary-pinned) HV.
fn accel_bucket(v: f32) -> u64 {
    let clamped = v.clamp(-40.0, 40.0);
    ((clamped / 0.5).round() as i32 + 80) as u64
}

/// Quantize one gyroscope axis into a bucket ID.
///
/// Range ±20 rad/s (~1150 deg/s, covers high-rate maneuvers) with
/// 0.1 rad/s resolution → 401 buckets centered on 0.
fn gyro_bucket(v: f32) -> u64 {
    let clamped = v.clamp(-20.0, 20.0);
    ((clamped / 0.1).round() as i32 + 200) as u64
}

/// Role-filler HDC fusion of a 6-axis IMU reading.
///
/// Mirrors the radio `SpectrumManager::perception_hv` pattern
/// (3ac2af34c6): for each axis, bind a stable role HV to a bucketed
/// value HV, then bundle the six pairs.
///
/// Determinism: same reading → identical HV. Different bucket values
/// produce HDC-orthogonal HVs (by the random-projection property of
/// `BinaryHV::random`).
#[derive(Debug, Clone, Default)]
pub struct RoleFillerImuFusion;

impl RoleFillerImuFusion {
    pub fn new() -> Self {
        Self
    }
}

impl ImuFusion for RoleFillerImuFusion {
    fn fuse(&self, reading: &ImuReading) -> ContinuousHV {
        let pairs = [
            BinaryHV::random(IMU_SEED_ACCEL_X)
                .bind(&BinaryHV::random(accel_bucket(reading.accel[0]))),
            BinaryHV::random(IMU_SEED_ACCEL_Y)
                .bind(&BinaryHV::random(accel_bucket(reading.accel[1]))),
            BinaryHV::random(IMU_SEED_ACCEL_Z)
                .bind(&BinaryHV::random(accel_bucket(reading.accel[2]))),
            BinaryHV::random(IMU_SEED_GYRO_X).bind(&BinaryHV::random(gyro_bucket(reading.gyro[0]))),
            BinaryHV::random(IMU_SEED_GYRO_Y).bind(&BinaryHV::random(gyro_bucket(reading.gyro[1]))),
            BinaryHV::random(IMU_SEED_GYRO_Z).bind(&BinaryHV::random(gyro_bucket(reading.gyro[2]))),
        ];
        let bundled = BinaryHV::bundle(&pairs);
        bundled.to_continuous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(accel: [f32; 3], gyro: [f32; 3]) -> ImuReading {
        ImuReading {
            accel,
            gyro,
            timestamp_us: 0,
        }
    }

    #[test]
    fn imu_zero_is_finite() {
        let f = RoleFillerImuFusion::new();
        let hv = f.fuse(&ImuReading::zero());
        assert_eq!(hv.values.len(), 16_384, "IMU HV must match CORE_HDC_DIM");
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn identical_readings_produce_identical_hvs() {
        let f = RoleFillerImuFusion::new();
        let a = f.fuse(&reading([1.2, 0.0, 9.8], [0.0, 0.1, 0.0]));
        let b = f.fuse(&reading([1.2, 0.0, 9.8], [0.0, 0.1, 0.0]));
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn different_readings_produce_different_hvs() {
        // Bucket width is 0.5 m/s² on accel; 2.0 m/s² apart is guaranteed
        // to land in different buckets.
        let f = RoleFillerImuFusion::new();
        let a = f.fuse(&reading([1.0, 0.0, 9.8], [0.0, 0.0, 0.0]));
        let b = f.fuse(&reading([3.0, 0.0, 9.8], [0.0, 0.0, 0.0]));
        assert_ne!(a.values, b.values);
    }

    #[test]
    fn saturation_beyond_range_still_finite() {
        // Readings outside ±40 m/s² / ±20 rad/s clamp rather than wrap.
        // Extreme inputs still produce a finite, stable HV.
        let f = RoleFillerImuFusion::new();
        let hv = f.fuse(&reading([1e6, -1e6, f32::MAX], [1e6, -1e6, f32::MIN]));
        assert_eq!(hv.values.len(), 16_384);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn saturated_extremes_are_stable() {
        // Two "extreme" readings that both clamp to the same bound must
        // produce identical HVs (clamping must be deterministic).
        let f = RoleFillerImuFusion::new();
        let a = f.fuse(&reading([100.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
        let b = f.fuse(&reading([1000.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
        assert_eq!(a.values, b.values, "above-range readings should saturate");
    }

    #[test]
    fn accel_bucket_center_is_80() {
        // Zero accel → bucket 80 (the middle of the 161-bucket range).
        assert_eq!(accel_bucket(0.0), 80);
        // ±40 m/s² saturates at the endpoints.
        assert_eq!(accel_bucket(40.0), 160);
        assert_eq!(accel_bucket(-40.0), 0);
    }

    #[test]
    fn gyro_bucket_center_is_200() {
        assert_eq!(gyro_bucket(0.0), 200);
        assert_eq!(gyro_bucket(20.0), 400);
        assert_eq!(gyro_bucket(-20.0), 0);
    }
}
