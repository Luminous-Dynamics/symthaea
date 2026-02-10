//! # Proof of Grounding (PoG)
//!
//! Implementation of MIP-E-003: Proof of Grounding Integration
//!
//! PoG binds Mycelix economic value to verified physical infrastructure:
//! - Energy: Renewable generation (solar, wind, hydro)
//! - Storage: Decentralized data storage (IPFS, Holochain DHT)
//! - Compute: Processing capacity with TEE attestation
//! - Bandwidth: Network connectivity

pub mod bandwidth;
pub mod compute;
pub mod energy;
pub mod storage;

pub use bandwidth::{BandwidthAttestation, PeerMeasurement};
pub use compute::{BenchmarkResults, ComputeAttestation, TeeType};
pub use energy::{EnergyAttestation, EnergyOracle, EnergySource};
pub use storage::{ReplicationProof, StorageAttestation, StorageType};

use serde::{Deserialize, Serialize};

/// Proof of Grounding score (0.0 - 1.0)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PogScore(f64);

impl PogScore {
    /// Create new PoG score, clamped to valid range
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Create zero score (no physical grounding)
    pub fn zero() -> Self {
        Self(0.0)
    }

    /// Get raw value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if above minimum threshold for validator eligibility
    pub fn validator_eligible(&self) -> bool {
        self.0 >= 0.3
    }
}

/// Verification level for infrastructure claims
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationLevel {
    /// Self-reported only (30% weight)
    SelfReported,
    /// Community witnessed (60% weight)
    CommunityWitnessed,
    /// External oracle verified (80% weight)
    OracleVerified,
    /// Hardware attestation (100% weight)
    HardwareAttested,
}

impl VerificationLevel {
    /// Get weight multiplier for this verification level
    pub fn weight(&self) -> f64 {
        match self {
            VerificationLevel::SelfReported => 0.3,
            VerificationLevel::CommunityWitnessed => 0.6,
            VerificationLevel::OracleVerified => 0.8,
            VerificationLevel::HardwareAttested => 1.0,
        }
    }
}

/// Infrastructure category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InfraCategory {
    /// Renewable energy generation
    Energy,
    /// Decentralized storage
    Storage,
    /// Compute capacity
    Compute,
    /// Network bandwidth
    Bandwidth,
}

impl InfraCategory {
    /// Get weight in composite PoG calculation
    pub fn weight(&self) -> f64 {
        match self {
            InfraCategory::Energy => 0.35,
            InfraCategory::Storage => 0.25,
            InfraCategory::Compute => 0.25,
            InfraCategory::Bandwidth => 0.15,
        }
    }
}

/// Composite PoG from multiple infrastructure categories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompositePoG {
    /// Energy infrastructure score
    pub energy: Option<PogScore>,
    /// Storage infrastructure score
    pub storage: Option<PogScore>,
    /// Compute infrastructure score
    pub compute: Option<PogScore>,
    /// Bandwidth infrastructure score
    pub bandwidth: Option<PogScore>,
}

impl CompositePoG {
    /// Calculate weighted total PoG score
    pub fn calculate_total(&self) -> PogScore {
        let mut total = 0.0;
        let mut count = 0;

        if let Some(e) = &self.energy {
            total += e.value() * InfraCategory::Energy.weight();
            count += 1;
        }
        if let Some(s) = &self.storage {
            total += s.value() * InfraCategory::Storage.weight();
            count += 1;
        }
        if let Some(c) = &self.compute {
            total += c.value() * InfraCategory::Compute.weight();
            count += 1;
        }
        if let Some(b) = &self.bandwidth {
            total += b.value() * InfraCategory::Bandwidth.weight();
            count += 1;
        }

        // Diversity bonus for multi-category contribution
        let diversity_bonus = match count {
            0 => 1.0,
            1 => 1.0,
            2 => 1.1,
            3 => 1.2,
            4 => 1.3,
            _ => 1.3,
        };

        PogScore::new(total * diversity_bonus)
    }

    /// Check if any infrastructure is contributed
    pub fn has_contribution(&self) -> bool {
        self.energy.is_some()
            || self.storage.is_some()
            || self.compute.is_some()
            || self.bandwidth.is_some()
    }

    /// Count categories with contribution
    pub fn category_count(&self) -> usize {
        let mut count = 0;
        if self.energy.is_some() {
            count += 1;
        }
        if self.storage.is_some() {
            count += 1;
        }
        if self.compute.is_some() {
            count += 1;
        }
        if self.bandwidth.is_some() {
            count += 1;
        }
        count
    }
}

/// Geographic coordinate for infrastructure location
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoCoordinate {
    /// Latitude (-90 to 90)
    pub latitude: f64,
    /// Longitude (-180 to 180)
    pub longitude: f64,
}

impl GeoCoordinate {
    /// Create new coordinate with validation
    pub fn new(latitude: f64, longitude: f64) -> Option<Self> {
        if (-90.0..=90.0).contains(&latitude) && (-180.0..=180.0).contains(&longitude) {
            Some(Self {
                latitude,
                longitude,
            })
        } else {
            None
        }
    }

    /// Calculate distance in kilometers using Haversine formula
    pub fn distance_km(&self, other: &GeoCoordinate) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6371.0;

        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let dlat = (other.latitude - self.latitude).to_radians();
        let dlon = (other.longitude - self.longitude).to_radians();

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_KM * c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pog_score() {
        let score = PogScore::new(0.75);
        assert!((score.value() - 0.75).abs() < 0.001);
        assert!(score.validator_eligible());

        let low = PogScore::new(0.2);
        assert!(!low.validator_eligible());

        // Clamping
        let over = PogScore::new(1.5);
        assert!((over.value() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_composite_pog() {
        let mut composite = CompositePoG::default();
        composite.energy = Some(PogScore::new(0.8));
        composite.storage = Some(PogScore::new(0.6));

        let total = composite.calculate_total();

        // 0.8*0.35 + 0.6*0.25 = 0.28 + 0.15 = 0.43
        // With 1.1x diversity bonus = 0.473
        assert!(total.value() > 0.4 && total.value() < 0.6);
        assert_eq!(composite.category_count(), 2);
    }

    #[test]
    fn test_verification_weight() {
        assert!((VerificationLevel::SelfReported.weight() - 0.3).abs() < 0.01);
        assert!((VerificationLevel::HardwareAttested.weight() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_geo_distance() {
        // New York to London approximately
        let ny = GeoCoordinate::new(40.7128, -74.0060).unwrap();
        let london = GeoCoordinate::new(51.5074, -0.1278).unwrap();

        let distance = ny.distance_km(&london);
        // Should be approximately 5570 km
        assert!(distance > 5500.0 && distance < 5700.0);
    }
}
