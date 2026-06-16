// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RadioHardware trait, mock/null implementations, and regulatory database.

use super::tier::{RadioTier, RegulatoryConstraints};
use std::collections::{HashMap, VecDeque};

// ═══════════════════════════════════════════════════════════════════════════════
// RADIO HARDWARE TRAIT — Physical radio abstraction
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors from radio hardware operations.
#[derive(Debug, Clone)]
pub enum RadioError {
    /// Hardware not available or powered off.
    Unavailable,
    /// Payload exceeds tier MTU.
    PayloadTooLarge { max: usize, got: usize },
    /// Regulatory constraint would be violated.
    RegulatoryViolation(String),
    /// Hardware-specific error.
    HardwareError(String),
    /// Channel busy (carrier sense).
    ChannelBusy,
    /// Frequency out of allowed band.
    FrequencyOutOfBand { freq_hz: u64, band: String },
}

impl std::fmt::Display for RadioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RadioError::Unavailable => write!(f, "radio hardware unavailable"),
            RadioError::PayloadTooLarge { max, got } => {
                write!(f, "payload too large: {} bytes (max {})", got, max)
            }
            RadioError::RegulatoryViolation(msg) => write!(f, "regulatory violation: {}", msg),
            RadioError::HardwareError(msg) => write!(f, "hardware error: {}", msg),
            RadioError::ChannelBusy => write!(f, "channel busy"),
            RadioError::FrequencyOutOfBand { freq_hz, band } => {
                write!(f, "frequency {} Hz out of band {}", freq_hz, band)
            }
        }
    }
}

/// Abstraction over physical radio hardware for mesh networking.
///
/// Implementations handle actual RF transmission/reception while
/// `SpectrumManager` handles cognitive-level decisions.
///
/// Basis: Clark & Chalmers (1998) — extended mind via radio as cognitive prosthesis.
pub trait RadioHardware: Send + Sync {
    /// Transmit a payload on the specified tier. Returns bytes actually sent.
    fn transmit(&mut self, tier: RadioTier, payload: &[u8]) -> Result<usize, RadioError>;
    /// Receive pending data from a tier. Returns `(payload, snr_db)`.
    fn receive(&mut self, tier: RadioTier) -> Result<Option<(Vec<u8>, f32)>, RadioError>;
    /// Query current signal-to-noise ratio for a tier.
    fn current_snr(&self, tier: RadioTier) -> Option<f32>;
    /// Query whether a tier's hardware is available/powered.
    fn is_available(&self, tier: RadioTier) -> bool;
    /// Set transmit power (dBm) for a tier, respecting regulatory limits.
    fn set_tx_power(&mut self, tier: RadioTier, power_dbm: f32) -> Result<(), RadioError>;
    /// Get current frequency (Hz) for a tier.
    fn current_frequency(&self, tier: RadioTier) -> Option<u64>;
    /// Tune to a specific frequency (Hz), respecting regulatory constraints.
    fn tune(&mut self, tier: RadioTier, frequency_hz: u64) -> Result<(), RadioError>;
    /// Hardware-specific name/identifier (e.g., "HackRF One", "RFM95W LoRa").
    fn hardware_id(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK RADIO HARDWARE — Testing implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Mock radio hardware for testing. Configurable SNR, availability per tier,
/// collects transmitted payloads for assertion.
pub struct MockRadioHardware {
    available: [bool; 3],
    snr: [f32; 3],
    frequency: [u64; 3],
    tx_power: [f32; 3],
    transmitted: Vec<(usize, Vec<u8>)>,
    receive_queue: VecDeque<(usize, Vec<u8>, f32)>,
    regulatory_db: Option<RegulatoryDatabase>,
}

impl MockRadioHardware {
    /// Create a mock with all tiers available and default SNR.
    pub fn new() -> Self {
        Self {
            available: [true, true, true],
            snr: [30.0, 15.0, 5.0],
            frequency: [2_450_000_000, 915_000_000, 7_100_000],
            tx_power: [20.0, 14.0, 30.0],
            transmitted: Vec::new(),
            receive_queue: VecDeque::new(),
            regulatory_db: None,
        }
    }

    pub fn set_available(&mut self, tier: RadioTier, available: bool) {
        self.available[tier as usize] = available;
    }

    pub fn set_snr(&mut self, tier: RadioTier, snr_db: f32) {
        self.snr[tier as usize] = snr_db;
    }

    pub fn transmitted_payloads(&self) -> &[(usize, Vec<u8>)] {
        &self.transmitted
    }

    pub fn inject_receive(&mut self, tier: RadioTier, payload: Vec<u8>, snr_db: f32) {
        self.receive_queue
            .push_back((tier as usize, payload, snr_db));
    }

    pub fn set_regulatory_db(&mut self, db: RegulatoryDatabase) {
        self.regulatory_db = Some(db);
    }
}

impl RadioHardware for MockRadioHardware {
    fn transmit(&mut self, tier: RadioTier, payload: &[u8]) -> Result<usize, RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        let mtu = tier.profile().mtu;
        if payload.len() > mtu {
            return Err(RadioError::PayloadTooLarge {
                max: mtu,
                got: payload.len(),
            });
        }
        self.transmitted.push((tier as usize, payload.to_vec()));
        Ok(payload.len())
    }

    fn receive(&mut self, tier: RadioTier) -> Result<Option<(Vec<u8>, f32)>, RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        let idx = tier as usize;
        if let Some(pos) = self.receive_queue.iter().position(|(t, _, _)| *t == idx) {
            let Some((_, payload, snr)) = self.receive_queue.remove(pos) else {
                // position() found it, remove() should succeed — but guard defensively
                return Ok(None);
            };
            Ok(Some((payload, snr)))
        } else {
            Ok(None)
        }
    }

    fn current_snr(&self, tier: RadioTier) -> Option<f32> {
        if self.available[tier as usize] {
            Some(self.snr[tier as usize])
        } else {
            None
        }
    }

    fn is_available(&self, tier: RadioTier) -> bool {
        self.available[tier as usize]
    }

    fn set_tx_power(&mut self, tier: RadioTier, power_dbm: f32) -> Result<(), RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        if let Some(ref db) = self.regulatory_db {
            let freq = self.frequency[tier as usize];
            if let Some(max_power) = db.max_power_for_frequency(freq) {
                if power_dbm > max_power {
                    return Err(RadioError::RegulatoryViolation(format!(
                        "power {} dBm exceeds max {} dBm for frequency {} Hz",
                        power_dbm, max_power, freq
                    )));
                }
            }
        }
        self.tx_power[tier as usize] = power_dbm;
        Ok(())
    }

    fn current_frequency(&self, tier: RadioTier) -> Option<u64> {
        if self.available[tier as usize] {
            Some(self.frequency[tier as usize])
        } else {
            None
        }
    }

    fn tune(&mut self, tier: RadioTier, frequency_hz: u64) -> Result<(), RadioError> {
        if !self.available[tier as usize] {
            return Err(RadioError::Unavailable);
        }
        if let Some(ref db) = self.regulatory_db {
            if !db.is_frequency_allowed(frequency_hz, tier) {
                return Err(RadioError::FrequencyOutOfBand {
                    freq_hz: frequency_hz,
                    band: format!("{:?} bands", db.region()),
                });
            }
        }
        self.frequency[tier as usize] = frequency_hz;
        Ok(())
    }

    fn hardware_id(&self) -> &str {
        "MockRadioHardware v1.0"
    }
}

/// No-op radio hardware that always returns `Unavailable`.
/// Used when mesh feature is enabled but no physical radio exists.
pub struct NullRadioHardware;

impl RadioHardware for NullRadioHardware {
    fn transmit(&mut self, _: RadioTier, _: &[u8]) -> Result<usize, RadioError> {
        Err(RadioError::Unavailable)
    }
    fn receive(&mut self, _: RadioTier) -> Result<Option<(Vec<u8>, f32)>, RadioError> {
        Err(RadioError::Unavailable)
    }
    fn current_snr(&self, _: RadioTier) -> Option<f32> {
        None
    }
    fn is_available(&self, _: RadioTier) -> bool {
        false
    }
    fn set_tx_power(&mut self, _: RadioTier, _: f32) -> Result<(), RadioError> {
        Err(RadioError::Unavailable)
    }
    fn current_frequency(&self, _: RadioTier) -> Option<u64> {
        None
    }
    fn tune(&mut self, _: RadioTier, _: u64) -> Result<(), RadioError> {
        Err(RadioError::Unavailable)
    }
    fn hardware_id(&self) -> &str {
        "NullRadioHardware"
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REGULATORY DATABASE — Region-aware frequency allocations
// ═══════════════════════════════════════════════════════════════════════════════

/// ITU regulatory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegulatoryRegion {
    /// FCC Part 15 (unlicensed) + Part 97 (amateur). US/Canada.
    FccUs,
    /// ETSI EN 300 220 (SRD) + EN 301 893 (5 GHz). EU/EEA/UK.
    EtsiEu,
    /// ARIB STD-T108. Japan.
    AribJp,
    /// Generic ISM (international fallback).
    IsmGlobal,
}

/// License requirement for a frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicenseType {
    Unlicensed,
    LightLicense,
    Amateur,
    Licensed,
}

/// A specific frequency allocation within a regulatory region.
#[derive(Debug, Clone)]
pub struct BandAllocation {
    pub name: String,
    pub freq_min_hz: u64,
    pub freq_max_hz: u64,
    pub max_eirp_dbm: f32,
    pub duty_cycle_max: Option<f32>,
    pub channel_bw_hz: u32,
    pub tier: RadioTier,
    pub license: LicenseType,
}

/// Regional regulatory database with band allocations.
///
/// Pre-populated with real-world spectrum allocations.
/// Basis: FCC Part 15/97, ETSI EN 300 220, ITU Radio Regulations.
#[derive(Debug, Clone)]
pub struct RegulatoryDatabase {
    region: RegulatoryRegion,
    bands: Vec<BandAllocation>,
}

impl RegulatoryDatabase {
    pub fn new(region: RegulatoryRegion) -> Self {
        let bands = match region {
            RegulatoryRegion::FccUs => Self::fcc_us_bands(),
            RegulatoryRegion::EtsiEu => Self::etsi_eu_bands(),
            RegulatoryRegion::AribJp => Self::arib_jp_bands(),
            RegulatoryRegion::IsmGlobal => Self::ism_global_bands(),
        };
        Self { region, bands }
    }

    pub fn region(&self) -> RegulatoryRegion {
        self.region
    }
    pub fn bands(&self) -> &[BandAllocation] {
        &self.bands
    }

    pub fn bands_for_tier(&self, tier: RadioTier) -> Vec<&BandAllocation> {
        self.bands.iter().filter(|b| b.tier == tier).collect()
    }

    pub fn is_frequency_allowed(&self, freq_hz: u64, tier: RadioTier) -> bool {
        self.bands
            .iter()
            .any(|b| b.tier == tier && freq_hz >= b.freq_min_hz && freq_hz <= b.freq_max_hz)
    }

    pub fn max_power_for_frequency(&self, freq_hz: u64) -> Option<f32> {
        self.bands
            .iter()
            .filter(|b| freq_hz >= b.freq_min_hz && freq_hz <= b.freq_max_hz)
            .map(|b| b.max_eirp_dbm)
            .fold(None, |acc, p| Some(acc.map_or(p, |a: f32| a.max(p))))
    }

    pub fn duty_cycle_for_band(&self, freq_hz: u64) -> Option<f32> {
        self.bands
            .iter()
            .find(|b| freq_hz >= b.freq_min_hz && freq_hz <= b.freq_max_hz)
            .and_then(|b| b.duty_cycle_max)
    }

    pub fn available_bandwidth(&self, tier: RadioTier) -> u64 {
        self.bands
            .iter()
            .filter(|b| b.tier == tier)
            .map(|b| b.freq_max_hz - b.freq_min_hz)
            .sum()
    }

    /// Convert to legacy `RegulatoryConstraints` for backward compatibility.
    pub fn to_legacy_constraints(&self) -> RegulatoryConstraints {
        let allowed_bands: Vec<(u64, u64)> = self
            .bands
            .iter()
            .filter(|b| b.license == LicenseType::Unlicensed)
            .map(|b| (b.freq_min_hz, b.freq_max_hz))
            .collect();
        let max_power = self
            .bands
            .iter()
            .filter(|b| b.license == LicenseType::Unlicensed)
            .map(|b| b.max_eirp_dbm)
            .fold(f32::NEG_INFINITY, f32::max);
        let region = match self.region {
            RegulatoryRegion::FccUs => "US",
            RegulatoryRegion::EtsiEu => "EU",
            RegulatoryRegion::AribJp => "JP",
            RegulatoryRegion::IsmGlobal => "GLOBAL",
        }
        .to_string();
        RegulatoryConstraints {
            allowed_bands,
            max_power_dbm: if max_power.is_finite() {
                max_power
            } else {
                0.0
            },
            region,
        }
    }

    fn fcc_us_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "ISM 915 MHz".into(),
                freq_min_hz: 902_000_000,
                freq_max_hz: 928_000_000,
                max_eirp_dbm: 30.0,
                duty_cycle_max: None,
                channel_bw_hz: 500_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_483_500_000,
                max_eirp_dbm: 36.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "U-NII-3 5.8 GHz".into(),
                freq_min_hz: 5_725_000_000,
                freq_max_hz: 5_850_000_000,
                max_eirp_dbm: 36.0,
                duty_cycle_max: None,
                channel_bw_hz: 20_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "HF 80m Amateur".into(),
                freq_min_hz: 3_500_000,
                freq_max_hz: 4_000_000,
                max_eirp_dbm: 61.76,
                duty_cycle_max: Some(0.5),
                channel_bw_hz: 3_000,
                tier: RadioTier::Regional,
                license: LicenseType::Amateur,
            },
            BandAllocation {
                name: "HF 40m Amateur".into(),
                freq_min_hz: 7_000_000,
                freq_max_hz: 7_300_000,
                max_eirp_dbm: 61.76,
                duty_cycle_max: Some(0.5),
                channel_bw_hz: 3_000,
                tier: RadioTier::Regional,
                license: LicenseType::Amateur,
            },
            BandAllocation {
                name: "HF 20m Amateur (NVIS)".into(),
                freq_min_hz: 14_000_000,
                freq_max_hz: 14_350_000,
                max_eirp_dbm: 61.76,
                duty_cycle_max: Some(0.5),
                channel_bw_hz: 3_000,
                tier: RadioTier::Regional,
                license: LicenseType::Amateur,
            },
        ]
    }

    fn etsi_eu_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "SRD 868 MHz (1%)".into(),
                freq_min_hz: 868_000_000,
                freq_max_hz: 868_600_000,
                max_eirp_dbm: 14.0,
                duty_cycle_max: Some(0.01),
                channel_bw_hz: 125_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "SRD 869 MHz (10%)".into(),
                freq_min_hz: 869_400_000,
                freq_max_hz: 869_650_000,
                max_eirp_dbm: 27.0,
                duty_cycle_max: Some(0.10),
                channel_bw_hz: 125_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_483_500_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "RLAN 5 GHz".into(),
                freq_min_hz: 5_150_000_000,
                freq_max_hz: 5_350_000_000,
                max_eirp_dbm: 23.0,
                duty_cycle_max: None,
                channel_bw_hz: 20_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
        ]
    }

    fn arib_jp_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "ARIB 920 MHz".into(),
                freq_min_hz: 920_000_000,
                freq_max_hz: 928_000_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: Some(0.10),
                channel_bw_hz: 200_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_483_500_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
        ]
    }

    fn ism_global_bands() -> Vec<BandAllocation> {
        vec![
            BandAllocation {
                name: "ISM 433 MHz".into(),
                freq_min_hz: 433_050_000,
                freq_max_hz: 434_790_000,
                max_eirp_dbm: 10.0,
                duty_cycle_max: Some(0.10),
                channel_bw_hz: 25_000,
                tier: RadioTier::Metro,
                license: LicenseType::Unlicensed,
            },
            BandAllocation {
                name: "ISM 2.4 GHz".into(),
                freq_min_hz: 2_400_000_000,
                freq_max_hz: 2_500_000_000,
                max_eirp_dbm: 20.0,
                duty_cycle_max: None,
                channel_bw_hz: 22_000_000,
                tier: RadioTier::Local,
                license: LicenseType::Unlicensed,
            },
        ]
    }
}
