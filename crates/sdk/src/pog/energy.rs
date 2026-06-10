// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Energy Infrastructure PoG
//!
//! Renewable energy generation verification for Proof of Grounding.

use super::{GeoCoordinate, PogScore, VerificationLevel};
use serde::{Deserialize, Serialize};

/// Energy source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyType {
    /// Solar photovoltaic
    SolarPV,
    /// Wind turbine
    Wind,
    /// Micro-hydro
    MicroHydro,
    /// Biogas
    Biogas,
    /// Grid-tied renewable
    GridTiedRenewable,
}

impl EnergyType {
    /// Get PoG multiplier for this energy type
    pub fn multiplier(&self) -> f64 {
        match self {
            EnergyType::SolarPV => 1.4,
            EnergyType::Wind => 1.4,
            EnergyType::MicroHydro => 1.3,
            EnergyType::Biogas => 1.2,
            EnergyType::GridTiedRenewable => 1.1,
        }
    }
}

/// Energy source registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySource {
    /// Source identifier
    pub source_id: String,
    /// Owner DID
    pub owner_did: String,
    /// Energy type
    pub source_type: EnergyType,
    /// Geographic location
    pub location: GeoCoordinate,
    /// Rated capacity in kW
    pub rated_capacity_kw: f64,
    /// Hardware attestation if available
    pub hardware_attestation: Option<HardwareAttestation>,
    /// Verification level
    pub verification_level: VerificationLevel,
}

/// Hardware attestation for smart inverters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAttestation {
    /// Device manufacturer
    pub manufacturer: String,
    /// Device model
    pub model: String,
    /// Serial number
    pub serial_number: String,
    /// Cryptographic signature from device
    pub device_signature: Vec<u8>,
    /// Attestation timestamp
    pub attested_at: u64,
}

/// Energy production attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyAttestation {
    /// Source ID
    pub source_id: String,
    /// Production in kWh for period
    pub production_kwh: f64,
    /// Period start timestamp
    pub period_start: u64,
    /// Period end timestamp
    pub period_end: u64,
    /// Verification proofs
    pub proofs: Vec<EnergyProof>,
}

/// Proof of energy production
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnergyProof {
    /// Smart meter reading
    SmartMeter {
        /// Meter reading value
        reading: f64,
        /// Cryptographic signature
        signature: Vec<u8>,
    },
    /// Inverter telemetry
    InverterTelemetry {
        /// Telemetry data payload
        data: Vec<u8>,
        /// Cryptographic signature
        signature: Vec<u8>,
    },
    /// Utility API confirmation
    UtilityApi {
        /// Confirmation identifier
        confirmation_id: String,
    },
    /// Community witness attestations
    CommunityWitness {
        /// List of witness DIDs
        witnesses: Vec<String>,
    },
}

/// Energy oracle for verification
#[derive(Debug, Clone)]
pub struct EnergyOracle {
    /// Registered sources
    pub sources: Vec<EnergySource>,
    /// Minimum daily production threshold (kWh)
    pub min_daily_kwh: f64,
}

impl Default for EnergyOracle {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            min_daily_kwh: 1.0, // 1 kWh/day minimum
        }
    }
}

impl EnergyOracle {
    /// Register new energy source
    pub fn register_source(&mut self, source: EnergySource) -> Result<(), String> {
        if source.rated_capacity_kw <= 0.0 {
            return Err("Rated capacity must be positive".to_string());
        }
        self.sources.push(source);
        Ok(())
    }

    /// Calculate PoG score for energy attestation
    pub fn calculate_pog(&self, attestation: &EnergyAttestation) -> PogScore {
        let source = match self
            .sources
            .iter()
            .find(|s| s.source_id == attestation.source_id)
        {
            Some(s) => s,
            None => return PogScore::zero(),
        };

        let period_days =
            (attestation.period_end - attestation.period_start) as f64 / (24.0 * 60.0 * 60.0);
        if period_days <= 0.0 {
            return PogScore::zero();
        }

        let daily_average = attestation.production_kwh / period_days;

        // Check minimum threshold
        if daily_average < self.min_daily_kwh {
            return PogScore::zero();
        }

        // Logarithmic scaling prevents mega-farm dominance
        let scaled_production = (daily_average.ln() + 1.0).min(5.0);

        // Apply source multiplier and verification weight
        let base_score = scaled_production
            * source.source_type.multiplier()
            * source.verification_level.weight()
            / 7.0; // Normalize to 0-1 range

        PogScore::new(base_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_pog_calculation() {
        let mut oracle = EnergyOracle::default();

        let source = EnergySource {
            source_id: "solar-1".to_string(),
            owner_did: "did:test:1".to_string(),
            source_type: EnergyType::SolarPV,
            location: GeoCoordinate::new(35.0, -100.0).unwrap(),
            rated_capacity_kw: 5.0,
            hardware_attestation: None,
            verification_level: VerificationLevel::OracleVerified,
        };

        oracle.register_source(source).unwrap();

        let attestation = EnergyAttestation {
            source_id: "solar-1".to_string(),
            production_kwh: 100.0, // 100 kWh over 30 days = 3.33 kWh/day
            period_start: 0,
            period_end: 30 * 24 * 60 * 60,
            proofs: vec![],
        };

        let pog = oracle.calculate_pog(&attestation);
        assert!(pog.value() > 0.0);
    }
}
