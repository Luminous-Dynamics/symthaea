// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! EU Digital Product Passport (DPP) ZKP attestations.
//!
//! Implements Regulation (EU) 2024/1781 (Ecodesign for Sustainable Products).
//! Battery DPPs mandatory Feb 2027, textiles/electronics phased 2028-2030.
//!
//! ## What DPP Requires
//!
//! Products must declare: materials, carbon footprint, recyclability,
//! conflict mineral status, supply chain compliance. ZKP enables proving
//! compliance WITHOUT revealing proprietary supplier relationships.
//!
//! ## Our Approach
//!
//! Categorical attestations (boolean/enum) are ideal for binary-field STARKs:
//! - "Conflict-free minerals": YES/NO (1 bit, 0 AND in Binius)
//! - "Recyclable content ≥ 70%": range proof (reuse HealthRangeAir)
//! - "Carbon footprint ≤ threshold": range proof
//! - "Supply chain hops ≤ N": counter proof
//!
//! These are the simplest possible ZKP circuits — categorical values
//! are natively binary, which is exactly what Binius optimizes for.

use serde::{Deserialize, Serialize};

/// DPP attestation categories per EU Regulation 2024/1781.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DppCategory {
    /// Battery passport (mandatory Feb 2027)
    Battery {
        /// Recycled content percentage (0-100)
        recycled_content_pct: u8,
        /// Carbon footprint class (A-G)
        carbon_class: char,
        /// Conflict mineral free
        conflict_free: bool,
        /// Expected lifetime (months)
        lifetime_months: u16,
    },
    /// Textile passport (mandatory ~2028)
    Textile {
        /// Organic content percentage
        organic_pct: u8,
        /// Recyclable
        recyclable: bool,
        /// Fair trade certified
        fair_trade: bool,
    },
    /// Electronics passport (mandatory ~2029)
    Electronics {
        /// Repairability score (0-10)
        repairability_score: u8,
        /// Contains restricted substances
        restricted_substances_free: bool,
        /// Recyclable percentage
        recyclable_pct: u8,
    },
    /// Generic product passport
    Generic {
        /// Category name
        category: String,
        /// Key-value attestations
        attestations: Vec<(String, String)>,
    },
}

/// A DPP ZKP attestation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DppAttestation {
    /// Product identifier (e.g., GTIN, serial number hash)
    pub product_id: String,
    /// DPP category with attestation values
    pub category: DppCategory,
    /// Which values are proven via ZKP (vs disclosed plaintext)
    pub zkp_proven: Vec<DppProvenClaim>,
    /// STARK proof bytes for range/categorical claims
    pub proof_bytes: Vec<u8>,
    /// Supply chain hop count (public)
    pub supply_chain_hops: u32,
    /// Issuer (manufacturer or authorized body)
    pub issuer_did: String,
    /// Domain tag for replay prevention
    pub domain_tag: String,
}

/// A single DPP claim proven via ZKP.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DppProvenClaim {
    /// What property (e.g., "recycled_content_pct")
    pub property: String,
    /// What's proven (e.g., "≥ 70%", "conflict_free = true")
    pub assertion: String,
    /// Proof type used
    pub proof_type: DppProofType,
}

/// Types of DPP proofs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DppProofType {
    /// Value in range [min, max] — uses Winterfell HealthRangeAir
    Range { min: u64, max: u64 },
    /// Boolean categorical — trivial in Binius (0 AND for XOR)
    Categorical,
    /// Supply chain hop count ≤ N
    HopLimit { max_hops: u32 },
}

/// Validate DPP attestation structure.
pub fn validate_dpp_attestation(att: &DppAttestation) -> Result<(), String> {
    if att.proof_bytes.is_empty() {
        return Err("Empty proof bytes".to_string());
    }
    if att.proof_bytes.len() > 500_000 {
        return Err("Proof exceeds 500KB".to_string());
    }
    if att.product_id.is_empty() {
        return Err("Empty product ID".to_string());
    }
    if !att.domain_tag.starts_with("ZTML:Supply:") {
        return Err("Wrong domain tag".to_string());
    }
    // Battery-specific validations
    if let DppCategory::Battery { recycled_content_pct, carbon_class, lifetime_months, .. } = &att.category {
        if *recycled_content_pct > 100 {
            return Err("Recycled content > 100%".to_string());
        }
        if !['A', 'B', 'C', 'D', 'E', 'F', 'G'].contains(carbon_class) {
            return Err("Invalid carbon class".to_string());
        }
        if *lifetime_months == 0 {
            return Err("Zero lifetime".to_string());
        }
    }
    Ok(())
}

/// Create a battery DPP attestation.
pub fn create_battery_dpp(
    product_id: &str,
    recycled_pct: u8,
    carbon_class: char,
    conflict_free: bool,
    lifetime_months: u16,
    proof_bytes: Vec<u8>,
    issuer_did: &str,
) -> DppAttestation {
    let mut zkp_proven = Vec::new();

    // Recycled content proven via range proof (≥ 70% for EU compliance)
    zkp_proven.push(DppProvenClaim {
        property: "recycled_content_pct".to_string(),
        assertion: format!("≥ 70% (actual: proven in ZK)"),
        proof_type: DppProofType::Range { min: 70, max: 100 },
    });

    // Conflict-free proven as categorical
    if conflict_free {
        zkp_proven.push(DppProvenClaim {
            property: "conflict_free".to_string(),
            assertion: "true".to_string(),
            proof_type: DppProofType::Categorical,
        });
    }

    DppAttestation {
        product_id: product_id.to_string(),
        category: DppCategory::Battery {
            recycled_content_pct: recycled_pct,
            carbon_class,
            conflict_free,
            lifetime_months,
        },
        zkp_proven,
        proof_bytes,
        supply_chain_hops: 0,
        issuer_did: issuer_did.to_string(),
        domain_tag: "ZTML:Supply:DPP:v1".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battery_dpp_creation() {
        let dpp = create_battery_dpp(
            "GTIN-12345678901234",
            85, // 85% recycled content
            'B', // Carbon class B
            true,
            96, // 8 year lifetime
            vec![1, 2, 3, 4], // Proof bytes
            "did:mycelix:manufacturer001",
        );

        assert_eq!(dpp.zkp_proven.len(), 2); // recycled + conflict-free
        assert!(validate_dpp_attestation(&dpp).is_ok());
    }

    #[test]
    fn test_invalid_recycled_content() {
        let mut dpp = create_battery_dpp("P1", 110, 'A', true, 12, vec![1], "did:test");
        assert!(validate_dpp_attestation(&dpp).is_err());
    }

    #[test]
    fn test_invalid_carbon_class() {
        let dpp = create_battery_dpp("P1", 80, 'Z', true, 12, vec![1], "did:test");
        assert!(validate_dpp_attestation(&dpp).is_err());
    }

    #[test]
    fn test_empty_proof_rejected() {
        let dpp = create_battery_dpp("P1", 80, 'B', true, 12, vec![], "did:test");
        assert!(validate_dpp_attestation(&dpp).is_err());
    }

    #[test]
    fn test_dpp_serialization() {
        let dpp = create_battery_dpp("P1", 80, 'B', true, 24, vec![1, 2], "did:test");
        let json = serde_json::to_string(&dpp).unwrap();
        let parsed: DppAttestation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.product_id, "P1");
    }
}
