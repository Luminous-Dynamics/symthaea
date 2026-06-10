// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ranging models: convert raw sensor data to distance + uncertainty.
//!
//! Each model maps a physical measurement (RSSI, time-of-flight, etc.)
//! to a range estimate with an explicit uncertainty model. These are
//! first-pass envelopes for fusion, not guarantees of field accuracy.

use serde::{Deserialize, Serialize};

/// Speed of light in meters per second.
const C_M_S: f64 = 299_792_458.0;

/// A range estimate with uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeEstimate {
    pub distance_m: f64,
    pub sigma_m: f64,
    pub method: RangingMethod,
}

/// Ranging technology used for the measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RangingMethod {
    /// Ultra-Wideband time-of-flight (σ ≈ 0.05-0.30 m)
    UltraWideband,
    /// LoRa time-of-arrival (σ ≈ 50-200 m)
    LoRaToA,
    /// LoRa time-difference-of-arrival (σ ≈ 30-100 m)
    LoRaTDoA,
    /// RSSI log-distance path loss (σ ≈ 3-8 m)
    RssiPathLoss,
    /// WiFi Fine Timing Measurement / RTT (σ ≈ 1-3 m, IEEE 802.11mc)
    WifiRtt,
    /// Manual survey with instruments (σ = instrument accuracy)
    ManualSurvey,
    /// Meshtastic hop-count estimate (σ ≈ 500-2000 m)
    MeshtasticHops,
    /// BLE Channel Sounding (Bluetooth 6.0, σ often ≈ 0.2-1.0 m on commodity devices)
    BleChannelSounding,
    /// Acoustic time-of-flight through water or air in low-visibility environments.
    AcousticTimeOfFlight,
    /// Optical time-of-flight or laser ranging.
    OpticalTimeOfFlight,
    /// Deep-space RF round-trip timing.
    DeepSpaceRadioRtt,
}

// ============================================================================
// RSSI PATH-LOSS MODEL
// ============================================================================

/// Estimate distance from Received Signal Strength Indicator (RSSI).
///
/// Uses the log-distance path loss model:
///   RSSI = RSSI_ref - 10 * n * log10(d / d_ref)
///
/// Solving for d:
///   d = d_ref * 10^((RSSI_ref - RSSI) / (10 * n))
///
/// # Parameters
/// - `rssi_dbm`: Received signal strength (dBm), typically -30 to -100
/// - `ref_rssi_dbm`: Reference RSSI at 1 meter (typically -40 to -60 dBm)
/// - `path_loss_exponent`: Environment factor (2.0=free space, 2.5-3.0=indoor, 3.5-4.5=dense)
///
/// # Returns
/// Range estimate with σ derived from environment (higher n = more uncertainty).
pub fn rssi_to_range(rssi_dbm: f64, ref_rssi_dbm: f64, path_loss_exponent: f64) -> RangeEstimate {
    let d_ref = 1.0; // reference distance: 1 meter
    let exponent = (ref_rssi_dbm - rssi_dbm) / (10.0 * path_loss_exponent);
    let distance = d_ref * 10.0_f64.powf(exponent);
    // Uncertainty scales with distance and path loss variability
    // Empirical: σ ≈ 0.2 * d for indoor, 0.1 * d for outdoor
    let sigma_fraction = 0.1 + 0.05 * (path_loss_exponent - 2.0);
    let sigma = (distance * sigma_fraction).max(0.5);
    RangeEstimate {
        distance_m: distance.max(0.1),
        sigma_m: sigma,
        method: RangingMethod::RssiPathLoss,
    }
}

// ============================================================================
// LoRa TIME-OF-ARRIVAL
// ============================================================================

/// Estimate distance from LoRa packet time-of-arrival.
///
/// # Parameters
/// - `toa_ns`: One-way time of arrival in nanoseconds
/// - `bandwidth_hz`: LoRa bandwidth (125_000, 250_000, or 500_000 Hz)
///
/// Uncertainty depends on bandwidth: higher BW = better time resolution.
pub fn lora_toa_to_range(toa_ns: f64, bandwidth_hz: f64) -> RangeEstimate {
    let distance = C_M_S * toa_ns * 1e-9;
    // Time resolution ≈ 1/BW. At 125 kHz: 8 μs → 2.4 km resolution
    // Practical accuracy much better due to correlation: ~50-200m
    let time_resolution_s = 1.0 / bandwidth_hz;
    let sigma = (C_M_S * time_resolution_s * 0.1).max(50.0); // min 50m
    RangeEstimate {
        distance_m: distance.max(0.0),
        sigma_m: sigma,
        method: RangingMethod::LoRaToA,
    }
}

// ============================================================================
// UWB TIME-OF-FLIGHT
// ============================================================================

/// Estimate distance from Ultra-Wideband two-way time-of-flight.
///
/// UWB uses 500 MHz+ bandwidth for sub-nanosecond time resolution.
/// IEEE 802.15.4z specifies centimeter-class accuracy.
///
/// # Parameters
/// - `tof_ns`: Two-way time of flight in nanoseconds (round-trip / 2)
pub fn uwb_tof_to_range(tof_ns: f64) -> RangeEstimate {
    let distance = C_M_S * tof_ns * 1e-9;
    // UWB achieves 10-30 cm accuracy in line-of-sight
    let sigma = 0.10 + distance * 0.002; // 10 cm base + 0.2% of range
    RangeEstimate {
        distance_m: distance.max(0.0),
        sigma_m: sigma.max(0.05),
        method: RangingMethod::UltraWideband,
    }
}

// ============================================================================
// WiFi ROUND-TRIP TIME (IEEE 802.11mc)
// ============================================================================

/// Estimate distance from WiFi Fine Timing Measurement.
///
/// # Parameters
/// - `rtt_ns`: Round-trip time in nanoseconds
pub fn wifi_rtt_to_range(rtt_ns: f64) -> RangeEstimate {
    let distance = C_M_S * rtt_ns * 1e-9 / 2.0;
    // WiFi RTT: typically 1-3m accuracy
    let sigma = 1.0 + distance * 0.01;
    RangeEstimate {
        distance_m: distance.max(0.0),
        sigma_m: sigma.max(0.5),
        method: RangingMethod::WifiRtt,
    }
}

// ============================================================================
// MANUAL SURVEY
// ============================================================================

/// Range from a manual survey measurement with known instrument accuracy.
pub fn manual_survey(distance_m: f64, instrument_accuracy_m: f64) -> RangeEstimate {
    RangeEstimate {
        distance_m: distance_m.max(0.0),
        sigma_m: instrument_accuracy_m.max(0.001),
        method: RangingMethod::ManualSurvey,
    }
}

// ============================================================================
// BLE CHANNEL SOUNDING (Bluetooth 6.0)
// ============================================================================

/// Estimate distance from BLE Channel Sounding phase measurement.
///
/// BLE Channel Sounding (Bluetooth 6.0, Android 16+) combines phase-based
/// ranging and round-trip timing, but indoor multipath on commodity devices
/// can easily push uncertainty into the decimeter-to-meter regime.
///
/// # Parameters
/// - `round_trip_time_ns`: Round-trip time in nanoseconds (from Channel Sounding)
///
/// # Accuracy
/// - Favorable line-of-sight: roughly 0.2-0.5 m
/// - Cluttered indoor / NLOS: often worse than 1 m
///
/// # References
/// - Bluetooth SIG Core Specification v6.0 (2024)
/// - Nordic Semiconductor Channel Sounding documentation
pub fn ble_channel_sounding_to_range(round_trip_time_ns: f64) -> RangeEstimate {
    let distance = C_M_S * round_trip_time_ns * 1e-9 / 2.0;
    // Conservative envelope for commodity hardware in real environments.
    let sigma = 0.25 + distance * 0.025;
    RangeEstimate {
        distance_m: distance.max(0.0),
        sigma_m: sigma.max(0.20),
        method: RangingMethod::BleChannelSounding,
    }
}

// ============================================================================
// ACOUSTIC TIME-OF-FLIGHT
// ============================================================================

/// Estimate distance from an acoustic time-of-flight measurement.
///
/// Suitable for underwater acoustics and cave environments where RF is poor
/// and line-of-sight optical links may be unavailable.
///
/// `sound_speed_m_s` should be environment-specific:
/// - air in caves: roughly 330-350 m/s depending on temperature/humidity
/// - seawater: roughly 1450-1550 m/s depending on salinity/temperature/pressure
pub fn acoustic_tof_to_range(
    round_trip_time_s: f64,
    sound_speed_m_s: f64,
    timing_sigma_s: f64,
) -> RangeEstimate {
    let sound_speed_m_s = sound_speed_m_s.max(1.0);
    let distance = sound_speed_m_s * round_trip_time_s.max(0.0) / 2.0;
    let sigma = (sound_speed_m_s * timing_sigma_s.max(1e-6) / 2.0).max(0.05);
    RangeEstimate {
        distance_m: distance,
        sigma_m: sigma,
        method: RangingMethod::AcousticTimeOfFlight,
    }
}

// ============================================================================
// OPTICAL / LASER TIME-OF-FLIGHT
// ============================================================================

/// Estimate distance from an optical or laser round-trip timing measurement.
///
/// `clock_sigma_ns` represents aggregate timing uncertainty from clocks,
/// timestamping electronics, and detection jitter.
pub fn optical_tof_to_range(round_trip_time_ns: f64, clock_sigma_ns: f64) -> RangeEstimate {
    let distance = C_M_S * round_trip_time_ns * 1e-9 / 2.0;
    let sigma = (C_M_S * clock_sigma_ns.max(0.001) * 1e-9 / 2.0).max(0.001);
    RangeEstimate {
        distance_m: distance.max(0.0),
        sigma_m: sigma,
        method: RangingMethod::OpticalTimeOfFlight,
    }
}

// ============================================================================
// DEEP-SPACE RF RTT
// ============================================================================

/// Estimate distance from RF round-trip timing with a known turnaround jitter.
pub fn deep_space_radio_rtt_to_range(
    round_trip_time_ns: f64,
    turnaround_jitter_ns: f64,
) -> RangeEstimate {
    let distance = C_M_S * round_trip_time_ns * 1e-9 / 2.0;
    let sigma = (C_M_S * turnaround_jitter_ns.max(1.0) * 1e-9 / 2.0).max(0.5);
    RangeEstimate {
        distance_m: distance.max(0.0),
        sigma_m: sigma,
        method: RangingMethod::DeepSpaceRadioRtt,
    }
}

// ============================================================================
// MESHTASTIC HOP COUNT
// ============================================================================

/// Rough distance estimate from Meshtastic mesh hop count.
///
/// Very coarse: each hop ≈ 1-5 km depending on terrain and antenna.
///
/// # Parameters
/// - `hop_count`: Number of mesh hops
/// - `avg_hop_distance_m`: Average per-hop distance (default: ~2000m)
pub fn meshtastic_hops_to_range(hop_count: u32, avg_hop_distance_m: f64) -> RangeEstimate {
    let distance = hop_count as f64 * avg_hop_distance_m;
    let sigma = distance * 0.5; // 50% uncertainty (very rough)
    RangeEstimate {
        distance_m: distance.max(0.0),
        sigma_m: sigma.max(500.0),
        method: RangingMethod::MeshtasticHops,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rssi_at_reference_distance() {
        let est = rssi_to_range(-50.0, -50.0, 2.0);
        assert!(
            (est.distance_m - 1.0).abs() < 0.1,
            "At ref RSSI, distance should be ~1m"
        );
    }

    #[test]
    fn rssi_farther_is_weaker() {
        let near = rssi_to_range(-55.0, -50.0, 2.0);
        let far = rssi_to_range(-70.0, -50.0, 2.0);
        assert!(far.distance_m > near.distance_m);
        assert!(far.sigma_m > near.sigma_m);
    }

    #[test]
    fn uwb_centimeter_accuracy() {
        let est = uwb_tof_to_range(33.36); // ~10m: 10/c = 33.36 ns
        assert!((est.distance_m - 10.0).abs() < 0.01);
        assert!(est.sigma_m < 0.5, "UWB should be sub-meter");
    }

    #[test]
    fn lora_reasonable_accuracy() {
        let est = lora_toa_to_range(3336.0, 125_000.0); // ~1000m
        assert!((est.distance_m - 1000.0).abs() < 1.0);
        assert!(est.sigma_m >= 50.0, "LoRa uncertainty >= 50m");
    }

    #[test]
    fn wifi_rtt_meter_class() {
        let est = wifi_rtt_to_range(66.7); // 10m round trip: 20/c ≈ 66.7 ns
        assert!((est.distance_m - 10.0).abs() < 0.1);
        assert!(est.sigma_m < 5.0);
    }

    #[test]
    fn manual_survey_preserves_accuracy() {
        let est = manual_survey(100.0, 0.01);
        assert_eq!(est.distance_m, 100.0);
        assert_eq!(est.sigma_m, 0.01);
    }

    #[test]
    fn meshtastic_scales_with_hops() {
        let one = meshtastic_hops_to_range(1, 2000.0);
        let three = meshtastic_hops_to_range(3, 2000.0);
        assert!((one.distance_m - 2000.0).abs() < 1.0);
        assert!((three.distance_m - 6000.0).abs() < 1.0);
        assert!(three.sigma_m > one.sigma_m);
    }

    #[test]
    fn ble_channel_sounding_uses_conservative_sigma() {
        // 10m: round trip = 2 * 10 / c = 66.7 ns
        let est = ble_channel_sounding_to_range(66.7);
        assert!((est.distance_m - 10.0).abs() < 0.1);
        assert!(
            est.sigma_m >= 0.2,
            "BLE CS should not be modeled as cm-class: {}",
            est.sigma_m
        );
        assert_eq!(est.method, RangingMethod::BleChannelSounding);
    }

    #[test]
    fn ble_channel_sounding_short_range() {
        // 1m: round trip = 6.67 ns
        let est = ble_channel_sounding_to_range(6.67);
        assert!((est.distance_m - 1.0).abs() < 0.01);
        assert!(
            est.sigma_m >= 0.2,
            "Short range BLE CS should still stay conservative: {}",
            est.sigma_m
        );
    }

    #[test]
    fn optical_tof_supports_sub_meter_sigma() {
        let est = optical_tof_to_range(66.7, 0.02);
        assert!((est.distance_m - 10.0).abs() < 0.1);
        assert!(
            est.sigma_m < 0.01,
            "Optical timing should support cm/mm-class sigma"
        );
    }

    #[test]
    fn acoustic_tof_supports_underwater_ranges() {
        let est = acoustic_tof_to_range(0.0267, 1500.0, 0.0001);
        assert!(
            (est.distance_m - 20.0).abs() < 0.2,
            "Acoustic range: {}",
            est.distance_m
        );
        assert!(est.sigma_m >= 0.05);
        assert_eq!(est.method, RangingMethod::AcousticTimeOfFlight);
    }

    #[test]
    fn acoustic_tof_supports_cave_air_ranges() {
        let est = acoustic_tof_to_range(0.0583, 343.0, 0.0002);
        assert!(
            (est.distance_m - 10.0).abs() < 0.2,
            "Acoustic cave range: {}",
            est.distance_m
        );
    }

    #[test]
    fn deep_space_radio_rtt_has_coarser_sigma() {
        let est = deep_space_radio_rtt_to_range(66.7, 10.0);
        assert!((est.distance_m - 10.0).abs() < 0.1);
        assert!(est.sigma_m >= 0.5);
    }

    #[test]
    fn all_methods_produce_positive_distance() {
        assert!(rssi_to_range(-90.0, -40.0, 3.0).distance_m > 0.0);
        assert!(lora_toa_to_range(0.1, 125_000.0).distance_m >= 0.0);
        assert!(uwb_tof_to_range(0.1).distance_m >= 0.0);
        assert!(wifi_rtt_to_range(0.1).distance_m >= 0.0);
        assert!(acoustic_tof_to_range(0.1, 1500.0, 0.001).distance_m >= 0.0);
        assert!(manual_survey(0.0, 1.0).distance_m >= 0.0);
        assert!(meshtastic_hops_to_range(0, 2000.0).distance_m >= 0.0);
    }
}
