// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RTL-SDR hardware, CCSDS space packets, and interplanetary functions.

use super::hardware::{RadioError, RadioHardware};
use super::tier::RadioTier;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// RTL-SDR RADIO HARDWARE — Wideband spectrum scanning (24 MHz - 1.766 GHz)
// ═══════════════════════════════════════════════════════════════════════════════

/// RTL-SDR spectrum scanner — implements `RadioHardware` for receive-only SDR.
///
/// Uses an RTL-SDR V3/V4 dongle (~$30) to scan the electromagnetic spectrum
/// from 24 MHz to 1.766 GHz. The SpectrumManager treats radio waves as a
/// sensory modality: prediction error when new signals appear, attention
/// drawn to anomalies, memory of known signal patterns.
///
/// ## What it detects
///
/// - **Survivor signals**: FM broadcast (88-108 MHz), CB (27 MHz), FRS/GMRS (462 MHz),
///   marine VHF (156 MHz), amateur (144/430 MHz), aviation (118-136 MHz)
/// - **Mesh frequencies**: LoRa ISM (868/915 MHz), WiFi (2.4 GHz via harmonic)
/// - **Threats**: Jamming patterns, suspicious transmissions, interference sources
/// - **Signal classification**: CW, voice, digital, noise (via spectral features)
///
/// ## Limitations
///
/// RTL-SDR is **receive-only**. `transmit()` always returns `Unavailable`.
/// For TX capability, use HackRF One ($350) with the same trait impl pattern.
///
/// ## Hardware
///
/// - RTL-SDR V4 ($30): 24-1766 MHz, 8-bit ADC, 2.4 MS/s
/// - RTL-SDR V3 ($25): Same range, slightly worse dynamic range
/// - Airspy Mini ($100): 24-1700 MHz, 12-bit ADC, better sensitivity
pub struct RtlSdrHardware {
    /// Current center frequency (Hz) per tier.
    frequencies: HashMap<usize, u64>,
    /// Last measured SNR per frequency band.
    snr_cache: HashMap<u64, f32>,
    /// Whether the SDR device is connected and operational.
    pub(super) available: bool,
    /// Device path or identifier.
    device_id: String,
    /// Scan sweep state: current frequency index in the sweep plan.
    sweep_index: usize,
    /// Sweep plan: frequencies to scan in order (Hz).
    pub(super) sweep_plan: Vec<u64>,
    /// Power spectrum density buffer from last FFT.
    psd_buffer: Vec<f32>,
    /// FFT size (number of bins).
    fft_size: usize,
    /// Sample rate (Hz).
    sample_rate: u32,
    /// Detected signals from last scan.
    detected_signals: Vec<DetectedSignal>,
}

/// A signal detected during spectrum scanning.
#[derive(Debug, Clone)]
pub struct DetectedSignal {
    /// Center frequency (Hz).
    pub frequency_hz: u64,
    /// Signal strength relative to noise floor (dB).
    pub snr_db: f32,
    /// Estimated bandwidth (Hz).
    pub bandwidth_hz: u32,
    /// Signal classification.
    pub classification: SignalClass,
    /// Cycle when detected.
    pub detected_cycle: u64,
}

/// Classification of detected radio signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalClass {
    /// Narrow-band continuous carrier (CW beacon, telemetry).
    ContinuousWave,
    /// Voice-modulated signal (FM/AM/SSB).
    Voice,
    /// Digital data signal (FSK, PSK, OFDM, LoRa chirp).
    Digital,
    /// Wideband noise or interference.
    Noise,
    /// Pulsed signal (radar, time beacon).
    Pulsed,
    /// Unknown / insufficient data to classify.
    Unknown,
}

/// Standard frequency bands for sweep planning.
pub mod bands {
    /// FM broadcast band (88-108 MHz) — survivor radio stations.
    pub const FM_BROADCAST: (u64, u64) = (88_000_000, 108_000_000);
    /// CB radio (26.965-27.405 MHz) — emergency comms.
    pub const CB_RADIO: (u64, u64) = (26_965_000, 27_405_000);
    /// FRS/GMRS (462-467 MHz) — family/general mobile.
    pub const FRS_GMRS: (u64, u64) = (462_000_000, 467_000_000);
    /// Marine VHF (156-162 MHz) — coastal/river comms.
    pub const MARINE_VHF: (u64, u64) = (156_000_000, 162_000_000);
    /// Aviation (118-136 MHz) — air traffic.
    pub const AVIATION: (u64, u64) = (118_000_000, 136_000_000);
    /// Amateur 2m (144-148 MHz) — ham radio.
    pub const AMATEUR_2M: (u64, u64) = (144_000_000, 148_000_000);
    /// Amateur 70cm (430-440 MHz) — ham radio.
    pub const AMATEUR_70CM: (u64, u64) = (430_000_000, 440_000_000);
    /// ISM 868 MHz (Europe LoRa).
    pub const ISM_868: (u64, u64) = (863_000_000, 870_000_000);
    /// ISM 915 MHz (US LoRa).
    pub const ISM_915: (u64, u64) = (902_000_000, 928_000_000);
    /// Weather satellite NOAA APT (137 MHz).
    pub const NOAA_APT: (u64, u64) = (137_000_000, 138_000_000);
    /// S-band satellite downlink (2.2-2.3 GHz) — above RTL-SDR range but
    /// included for completeness with upconverters.
    pub const S_BAND_DOWN: (u64, u64) = (2_200_000_000, 2_300_000_000);
    /// UHF satellite (400-402 MHz) — CubeSat, ISS.
    pub const UHF_SATELLITE: (u64, u64) = (400_000_000, 402_000_000);
}

impl Default for RtlSdrHardware {
    fn default() -> Self {
        // Default sweep plan: key emergency + mesh + satellite bands
        let sweep_plan = vec![
            27_185_000,  // CB channel 19 (truckers/emergency)
            88_500_000,  // FM broadcast low end
            108_000_000, // FM broadcast high end
            121_500_000, // Aviation emergency (ELT)
            137_500_000, // NOAA weather satellite
            144_390_000, // APRS (ham packet radio)
            146_520_000, // 2m calling frequency
            156_800_000, // Marine channel 16 (distress)
            400_500_000, // UHF satellite
            433_920_000, // ISM 433 MHz (sensors, remotes)
            446_000_000, // PMR446 (Europe walkie-talkies)
            462_562_500, // FRS channel 1
            868_100_000, // LoRa EU
            915_000_000, // LoRa US
        ];

        Self {
            frequencies: HashMap::new(),
            snr_cache: HashMap::new(),
            available: false,
            device_id: "rtlsdr-0".into(),
            sweep_index: 0,
            sweep_plan,
            psd_buffer: vec![0.0; 1024],
            fft_size: 1024,
            sample_rate: 2_400_000, // 2.4 MS/s
            detected_signals: Vec::new(),
        }
    }
}

impl RtlSdrHardware {
    /// Create with a specific device identifier.
    pub fn new(device_id: &str) -> Self {
        Self {
            device_id: device_id.into(),
            ..Self::default()
        }
    }

    /// Connect to the RTL-SDR device.
    ///
    /// In production, this calls `rtlsdr_open()` via the `rtlsdr` crate.
    /// Currently stubbed — the integration point is ready, hardware access
    /// requires the `rtlsdr` system library.
    pub fn connect(&mut self) -> Result<(), RadioError> {
        // TODO: rtlsdr::open(device_index) when rtlsdr crate is added
        tracing::info!("RTL-SDR connecting to device {}", self.device_id);
        self.available = true;
        Ok(())
    }

    /// Advance the sweep to the next frequency in the plan.
    /// Returns the frequency that was just scanned.
    pub fn advance_sweep(&mut self) -> u64 {
        let freq = self.sweep_plan[self.sweep_index];
        self.sweep_index = (self.sweep_index + 1) % self.sweep_plan.len();
        freq
    }

    /// Inject a simulated spectrum observation (for testing without hardware).
    pub fn inject_observation(&mut self, freq_hz: u64, snr_db: f32) {
        self.snr_cache.insert(freq_hz, snr_db);
    }

    /// Classify a signal based on spectral features.
    ///
    /// Uses simple heuristics on bandwidth and frequency:
    /// - Narrow (<1 kHz) → CW
    /// - 3-15 kHz → Voice (AM/FM/SSB)
    /// - 10-500 kHz → Digital
    /// - >500 kHz → Noise/wideband
    pub fn classify_signal(bandwidth_hz: u32, _freq_hz: u64) -> SignalClass {
        match bandwidth_hz {
            0..=1_000 => SignalClass::ContinuousWave,
            1_001..=15_000 => SignalClass::Voice,
            15_001..=500_000 => SignalClass::Digital,
            _ => SignalClass::Noise,
        }
    }

    /// Get all signals detected in the most recent sweep.
    pub fn detected_signals(&self) -> &[DetectedSignal] {
        &self.detected_signals
    }

    /// Count of distinct frequencies with detected signals.
    pub fn active_frequency_count(&self) -> usize {
        self.snr_cache.values().filter(|&&snr| snr > 0.0).count()
    }

    /// Best SNR across all scanned frequencies.
    pub fn peak_snr(&self) -> f32 {
        self.snr_cache
            .values()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Worst SNR (most jammed band).
    pub fn worst_snr(&self) -> f32 {
        self.snr_cache
            .values()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }
}

impl RadioHardware for RtlSdrHardware {
    fn transmit(&mut self, _tier: RadioTier, _payload: &[u8]) -> Result<usize, RadioError> {
        // RTL-SDR is receive-only. Use HackRF for TX.
        Err(RadioError::Unavailable)
    }

    fn receive(&mut self, _tier: RadioTier) -> Result<Option<(Vec<u8>, f32)>, RadioError> {
        if !self.available {
            return Err(RadioError::Unavailable);
        }
        // RTL-SDR doesn't demodulate — it provides raw IQ samples.
        // Spectrum observations are the primary output, not demodulated data.
        // For demodulated receive, use a dedicated modem (Meshtastic, etc.).
        Ok(None)
    }

    fn current_snr(&self, tier: RadioTier) -> Option<f32> {
        if !self.available {
            return None;
        }
        // Return SNR for the frequency associated with this tier
        let freq = self.frequencies.get(&(tier as usize))?;
        self.snr_cache.get(freq).copied()
    }

    fn is_available(&self, _tier: RadioTier) -> bool {
        self.available
    }

    fn set_tx_power(&mut self, _tier: RadioTier, _power_dbm: f32) -> Result<(), RadioError> {
        Err(RadioError::Unavailable) // Receive-only
    }

    fn current_frequency(&self, tier: RadioTier) -> Option<u64> {
        self.frequencies.get(&(tier as usize)).copied()
    }

    fn tune(&mut self, tier: RadioTier, frequency_hz: u64) -> Result<(), RadioError> {
        if !self.available {
            return Err(RadioError::Unavailable);
        }
        // Validate frequency range (RTL-SDR: 24 MHz - 1.766 GHz)
        if frequency_hz < 24_000_000 || frequency_hz > 1_766_000_000 {
            return Err(RadioError::FrequencyOutOfBand {
                freq_hz: frequency_hz,
                band: "RTL-SDR 24-1766 MHz".into(),
            });
        }
        self.frequencies.insert(tier as usize, frequency_hz);
        // TODO: rtlsdr_set_center_freq(frequency_hz) when hardware connected
        Ok(())
    }

    fn hardware_id(&self) -> &str {
        "RTL-SDR Spectrum Scanner"
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CCSDS SPACE PACKET — Spacecraft communication framing
// ═══════════════════════════════════════════════════════════════════════════════

/// CCSDS Space Packet Protocol (CCSDS 133.0-B-2) simplified framing.
///
/// The Consultative Committee for Space Data Systems defines the standard
/// packet format used by NASA, ESA, JAXA, and most space agencies.
///
/// Our implementation maps WisdomPackets and consciousness deltas into
/// CCSDS-compatible frames for transmission via satellite radio.
///
/// ## Framing
///
/// ```text
/// ┌──────────────────────────────────────────────────────────┐
/// │ Primary Header (6 bytes)                                  │
/// │ ├─ Version (3 bits): 000                                  │
/// │ ├─ Type (1 bit): 0=telemetry, 1=telecommand              │
/// │ ├─ Sec Header Flag (1 bit)                                │
/// │ ├─ APID (11 bits): Application Process ID                 │
/// │ ├─ Sequence Flags (2 bits): 11=standalone                 │
/// │ ├─ Sequence Count (14 bits): 0-16383                      │
/// │ └─ Data Length (16 bits): payload_len - 1                  │
/// ├──────────────────────────────────────────────────────────┤
/// │ Secondary Header (optional, variable)                      │
/// │ └─ Timestamp (8 bytes CDS format)                         │
/// ├──────────────────────────────────────────────────────────┤
/// │ Data Field (variable, max 65,535 bytes)                    │
/// └──────────────────────────────────────────────────────────┘
/// ```
pub struct CcsdsPacket {
    /// Application Process Identifier (0-2047).
    pub apid: u16,
    /// Sequence count (0-16383, wrapping).
    pub sequence: u16,
    /// Packet type: false = telemetry (TM), true = telecommand (TC).
    pub is_command: bool,
    /// Payload data.
    pub data: Vec<u8>,
}

/// APID assignments for Mycelix/Symthaea space operations.
pub mod apid {
    /// Consciousness telemetry (Phi, neuromodulators, harmony).
    pub const CONSCIOUSNESS_TM: u16 = 100;
    /// Compressed consciousness delta.
    pub const CONSCIOUSNESS_DELTA: u16 = 101;
    /// Governance relay (TEND, emergency, etc.).
    pub const GOVERNANCE_RELAY: u16 = 200;
    /// Threat signature for distributed immunity.
    pub const THREAT_SIGNATURE: u16 = 201;
    /// Consolidated wisdom (dream-processed store-and-forward).
    pub const CONSOLIDATED_WISDOM: u16 = 300;
    /// Heartbeat / node health beacon.
    pub const HEARTBEAT: u16 = 400;
    /// Sensor telemetry (IoT readings).
    pub const SENSOR_TM: u16 = 500;
}

/// CCSDS primary header size.
const CCSDS_HEADER_SIZE: usize = 6;

impl CcsdsPacket {
    /// Encode into CCSDS wire format.
    pub fn encode(&self) -> Vec<u8> {
        let data_len = self.data.len();
        let mut buf = Vec::with_capacity(CCSDS_HEADER_SIZE + data_len);

        // Word 1: Version(3) + Type(1) + SecHdrFlag(1) + APID(11)
        #[allow(clippy::identity_op)]
        let word1: u16 = (0b000 << 13)                           // Version = 0
            | (if self.is_command { 1 } else { 0 } << 12)        // Type
            | (0 << 11)                                            // No secondary header
            | (self.apid & 0x7FF); // APID
        buf.extend_from_slice(&word1.to_be_bytes());

        // Word 2: SeqFlags(2) + SeqCount(14)
        let word2: u16 = (0b11 << 14)                             // Standalone packet
            | (self.sequence & 0x3FFF); // Sequence count
        buf.extend_from_slice(&word2.to_be_bytes());

        // Word 3: Data length - 1 (CCSDS convention)
        let word3: u16 = if data_len > 0 {
            (data_len - 1) as u16
        } else {
            0
        };
        buf.extend_from_slice(&word3.to_be_bytes());

        // Data field
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Decode from CCSDS wire format.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < CCSDS_HEADER_SIZE {
            return None;
        }

        let word1 = u16::from_be_bytes([buf[0], buf[1]]);
        let word2 = u16::from_be_bytes([buf[2], buf[3]]);
        let word3 = u16::from_be_bytes([buf[4], buf[5]]);

        let version = (word1 >> 13) & 0x7;
        if version != 0 {
            return None; // Only version 0 supported
        }

        let is_command = (word1 >> 12) & 1 == 1;
        let apid = word1 & 0x7FF;
        let sequence = word2 & 0x3FFF;
        let data_len = (word3 as usize) + 1;

        if buf.len() < CCSDS_HEADER_SIZE + data_len {
            return None;
        }

        let data = buf[CCSDS_HEADER_SIZE..CCSDS_HEADER_SIZE + data_len].to_vec();

        Some(Self {
            apid,
            sequence,
            is_command,
            data,
        })
    }
}

/// Interplanetary latency calculator based on orbital mechanics.
///
/// Returns one-way light-time in seconds for common destinations.
/// Used by ConsciousnessAwareRouter to set sharing cadence for space links.
pub fn light_time_seconds(destination: &str) -> f64 {
    match destination {
        "LEO" | "leo" | "iss" => 0.003,    // ~400 km: 1.3 ms
        "GEO" | "geo" => 0.12,             // ~36,000 km: 120 ms
        "Moon" | "moon" | "lunar" => 1.28, // 384,400 km: 1.28 s
        "Mars" | "mars" => 750.0,          // 4-24 min, avg 12.5 min
        "Jupiter" | "jupiter" => 2_580.0,  // 33-53 min, avg 43 min
        "Saturn" | "saturn" => 4_800.0,    // ~80 min
        "deep" | "voyager" => 72_000.0,    // ~20 hours (Voyager 1)
        _ => 1.0,                          // Default: 1 second
    }
}

/// Calculate minimum sharing cadence (cycles) for a given light-time.
///
/// There's no point sharing consciousness state more frequently than
/// the round-trip light time. Dream consolidation fills the gap.
pub fn min_sharing_cadence_for_latency(light_time_s: f64, cycle_hz: f32) -> u32 {
    // Round-trip = 2 * one-way. Add 20% margin for processing.
    let round_trip = light_time_s * 2.0 * 1.2;
    let cycles = (round_trip * cycle_hz as f64) as u32;
    cycles.max(1)
}
